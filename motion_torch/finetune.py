"""Minimal fine-tuning of a converted MOTION model in PyTorch (pretraining stays in TensorFlow).

Objective = v3/train.py: free-running AR rollout with stop-gradient feedback, one masked mean of the clamped
per-frame relative error over all segments' valid frames (`rollout_loss`); single-shot families use
`singleshot_loss`, receding-horizon families `singleshot_rh_loss` (lead-1 hops). Relative L2 per channel
(--per_channel 1, the training default) or joint, rel-L1 with --l1 1 (published MOTION-IVP loss). Per-segment backward
(the feedback is detached, so summing per-segment gradients equals the full-rollout gradient and keeps the
activation memory at one segment). Optimizer Adam (Keras eps 1e-7) or AdamW, schedule constant-with-warmup or
cosine (Keras CosineDecay with linear warmup from 2% of the peak, as --lr_sched cosine), gradient clipping per
tensor (Keras `clipnorm`, as v3/train.py) or global.

  python -m motion_torch.finetune --ckpt m1f_ckpt_best.npz --preset motion_s --units units.pkl \
      --families incom_ns --steps 200 --lr 1e-4 --save ft.pt
  python -m motion_torch.finetune --preset motion_s --ckpt ... --synthetic incom_ns --steps 20   # smoke
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
import time

import numpy as np
import torch

from .config import PRESETS, MotionConfig
from .convert import build_model, save_pt
from .data.example import box_dims_for, load_local_units, make_example_ar, stack_batch, synthetic_sample
from .inference import rel, to_torch
from .model.v3model import to_grid


def _common(b):
    qall = torch.cat([b["coords_node"], b["coords_q"]], 1)
    gall = torch.cat([b["geom_node"], b["geom_q"]], 1)
    u_prev = torch.cat([b["x_win"][:, -1], b["ic_q"]], 1)
    return qall, gall, u_prev


def rollout_loss_backward(model, b, clamp=4.0, per_channel=True, l1=False, k_seg=None, backward=True):
    """v3/train.rollout_loss with per-segment backward. Returns the loss value (float)."""
    K, steady, dense2d = b["K"], b["steady"], b["dense2d"]
    roles, bd = list(b["roles"]), tuple(b["box_dims"])
    gd = tuple(b["dims"]) if b.get("dims") else None
    k_seg = int(k_seg or b["k_seg"])
    Bn = b["coords_node"].shape[0]
    Nn = b["coords_node"].shape[1]
    qall, gall, u_prev = _common(b)
    win, fmask = b["x_win"], b["fmask"]
    prev_grid = prev2_grid = None
    if dense2d:
        prev_grid = to_grid(win[:, -1], gd, model.S)
        prev2_grid = to_grid(win[:, -2], gd, model.S) if win.shape[1] >= 2 else None
    if steady:
        wsum = sum(float(k + 1) * Bn for k in range(k_seg))
    else:
        wsum = float(b["tstep_mask"][:, :k_seg].sum())
    wsum = max(wsum, 1e-6)
    total = 0.0
    for k in range(k_seg):
        with torch.set_grad_enabled(backward):
            pred = model.step(b["coords_node"], win, fmask, b["geom_node"], K, roles, b["cond"], b["op_multihot"],
                              qall, gall, u_prev, steady=steady, prev_grid=prev_grid, grid_dims=gd,
                              fam_id=b["fam_id"], sdf_box=b["sdf_box"], box_dims=bd, fbmask=b["fbmask"],
                              prev2_grid=prev2_grid)
            kt = 0 if steady else k
            y_all = torch.cat([b["y_node"][:, kt], b["y_q"][:, kt]], 1)
            r = torch.clamp(rel(pred, y_all, b["cmask"], per_channel, l1=l1), max=clamp)
            wk = (float(k + 1) * torch.ones_like(r)) if steady else b["tstep_mask"][:, k]
            L = (wk * r).sum() / wsum
        if backward:
            L.backward()
        total += float(L.detach())
        u_prev = pred.detach()
        pred_n = u_prev[:, :Nn]
        win = torch.cat([win[:, 1:], pred_n[:, None]], 1)
        fmask = torch.cat([fmask[:, 1:], torch.ones_like(fmask[:, :1])], 1)
        if dense2d:
            prev2_grid, prev_grid = prev_grid, to_grid(pred_n, gd, model.S)
    return total


def singleshot_loss_backward(model, b, clamp=4.0, per_channel=True, l1=False, k_seg=None, backward=True,
                             rh_roll=0, rh_leads=1):
    """v3/train.singleshot_loss (rh_roll=0) or singleshot_rh_loss (rh_roll calls, leads 1..rh_leads)."""
    k_seg = int(k_seg or b["k_seg"])
    gd = tuple(b["dims"]) if b.get("dims") else None
    Nn = b["coords_node"].shape[1]
    qall, gall, u_prev = _common(b)
    win, fmask = b["x_win"], b["fmask"]
    use_pg = gd is not None and model.ss_warp and b["dense2d"]   # (TF: grid_dims only; a sub-sampled 3-D window cannot be gridded)
    pg = to_grid(win[:, -1], gd, model.S) if use_pg else None
    calls = [(r, [l for l in range(1, rh_leads + 1) if r + l <= k_seg]) for r in range(min(rh_roll, k_seg))] \
        if rh_roll else [(0, None)]
    wsum = 0.0
    for r, leads in calls:
        ks = list(range(1, k_seg + 1)) if leads is None else [r + l for l in leads]
        wsum += float(sum(b["tstep_mask"][:, k - 1].sum() for k in ks))
    wsum = max(wsum, 1e-6)
    total = 0.0
    for r, leads in calls:
        with torch.set_grad_enabled(backward):
            P = model.step_ss(b["coords_node"], win, fmask, b["geom_node"], b["K"], list(b["roles"]), b["cond"],
                              b["op_multihot"], qall, gall, u_prev, kf=k_seg, fam_id=b["fam_id"],
                              sdf_box=b["sdf_box"], box_dims=tuple(b["box_dims"]), prev_grid=pg, grid_dims=gd,
                              leads=leads)
            ks = list(range(1, k_seg + 1)) if leads is None else [r + l for l in leads]
            L = 0.0
            for j, k in enumerate(ks):
                y_all = torch.cat([b["y_node"][:, k - 1], b["y_q"][:, k - 1]], 1)
                rr = torch.clamp(rel(P[:, j], y_all, b["cmask"], per_channel, l1=l1), max=clamp)
                L = L + (b["tstep_mask"][:, k - 1] * rr).sum() / wsum
        if backward:
            L.backward()
        total += float(L.detach())
        if rh_roll:
            u_prev = P[:, 0].detach()
            pred_n = u_prev[:, :Nn]
            win = torch.cat([win[:, 1:], pred_n[:, None]], 1)
            fmask = torch.cat([fmask[:, 1:], torch.ones_like(fmask[:, :1])], 1)
            if use_pg:
                pg = to_grid(pred_n, gd, model.S)
    return total


def loss_backward(model, b, a, backward=True):
    kw = dict(clamp=a.clamp, per_channel=bool(a.per_channel), l1=bool(a.l1), backward=backward)
    if b["mode"] == "ss":
        return singleshot_loss_backward(model, b, **kw)
    if b["mode"] == "rh":
        return singleshot_rh_loss_backward(model, b, a, **kw)
    return rollout_loss_backward(model, b, **kw)


def singleshot_rh_loss_backward(model, b, a, **kw):
    return singleshot_loss_backward(model, b, rh_roll=a.rh_roll, rh_leads=a.rh_leads, **kw)


def lr_lambda(a):
    """Multiplier of a.lr at optimizer step s (Keras _WarmupConst / CosineDecay(warmup) of v3/train.py)."""
    def f(s):
        if a.lr_sched == "cosine":
            if s < a.warmup:
                return 0.02 + (1.0 - 0.02) * s / max(1, a.warmup)
            done = min(max(s - a.warmup, 0) / max(1, a.steps), 1.0)
            return (1.0 - a.lr_alpha) * 0.5 * (1.0 + math.cos(math.pi * done)) + a.lr_alpha
        return min(1.0, s / max(1, a.warmup))
    return f


def clip_grads(params, max_norm, mode):
    if max_norm <= 0:
        return
    if mode == "global":
        torch.nn.utils.clip_grad_norm_(params, max_norm)
        return
    for p in params:                                   # Keras `clipnorm`: each gradient tensor separately
        if p.grad is not None:
            n = p.grad.norm()
            if n > max_norm:
                p.grad.mul_(max_norm / (n + 1e-12))


def example_stream(a, cfg, seed):
    rng = np.random.default_rng(seed)
    if a.synthetic:
        fams = a.synthetic.split(",")
        pool = {f: [synthetic_sample(f, rng, T=20) for _ in range(4)] for f in fams}
    else:
        U = load_local_units(a.units)
        fams = a.families.split(",") if a.families else list(U)
        pool = {f: U[f][a.skip_units:a.skip_units + a.n_units] if a.n_units else U[f] for f in fams}
    for it in itertools.count():
        f = fams[it % len(fams)]
        s = pool[f][int(rng.integers(len(pool[f])))]
        exs = [make_example_ar(s, cfg.t_in, a.k_fut, 4, a.n_colloc, a.enc_n, rng, cfg,
                               box_dims=box_dims_for(f, cfg), ic_rand=a.ic_rand) for _ in range(a.batch)]
        yield stack_batch(exs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="TF npz or converted .pt")
    ap.add_argument("--preset", default="motion_s", choices=sorted(PRESETS))
    ap.add_argument("--units", default=os.environ.get("LOCAL_UNITS", ""), help="LOCAL_UNITS pickle")
    ap.add_argument("--families", default="")
    ap.add_argument("--n_units", type=int, default=0, help="first N units per family (0 = all)")
    ap.add_argument("--skip_units", type=int, default=0)
    ap.add_argument("--synthetic", default="", help="comma list of families: train on random synthetic samples")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--k_fut", type=int, default=10)
    ap.add_argument("--n_colloc", type=int, default=2048)
    ap.add_argument("--enc_n", type=int, default=8192)
    ap.add_argument("--ic_rand", type=int, default=1, help="random IC position within the trajectory (train)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--lr_sched", default="cosine", choices=["const", "cosine"])
    ap.add_argument("--lr_alpha", type=float, default=0.05)
    ap.add_argument("--opt", default="adam", choices=["adam", "adamw"])
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--clip_mode", default="per_tensor", choices=["per_tensor", "global"])
    ap.add_argument("--clamp", type=float, default=4.0)
    ap.add_argument("--per_channel", type=int, default=1)
    ap.add_argument("--l1", type=int, default=0)
    ap.add_argument("--rh_roll", type=int, default=4)
    ap.add_argument("--rh_leads", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=0, help="loss on a fixed held-in batch every N steps")
    ap.add_argument("--overfit", type=int, default=0, help="1: train on the fixed batch itself (sanity check)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = PRESETS[a.preset]
    model = build_model(cfg, a.ckpt, device=a.device, seed=a.seed)
    cfg = model.cfg
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    if a.opt == "adamw":
        opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, a.beta2), eps=1e-7, weight_decay=a.wd)
    else:
        opt = torch.optim.Adam(params, lr=a.lr, betas=(0.9, a.beta2), eps=1e-7, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(a))
    stream = example_stream(a, cfg, a.seed)
    fixed = to_torch(next(example_stream(a, cfg, a.seed + 99)), a.device)
    L0 = loss_backward(model, fixed, a, backward=False)
    print(f"[ft] {cfg.name}: {sum(p.numel() for p in params) / 1e6:.2f}M trainable | fixed-batch loss {L0:.5f}",
          flush=True)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        b = fixed if a.overfit else to_torch(next(stream), a.device)
        opt.zero_grad(set_to_none=True)
        L = loss_backward(model, b, a)
        clip_grads(params, a.clip, a.clip_mode)
        opt.step()
        sched.step()
        msg = f"[ft] step {step:5d} loss {L:.5f} lr {sched.get_last_lr()[0]:.2e} ({b['families'][0]}, {b['mode']})"
        if a.eval_every and step % a.eval_every == 0:
            msg += f" | fixed-batch loss {loss_backward(model, fixed, a, backward=False):.5f}"
        print(msg + f"  {time.time() - t0:.0f}s", flush=True)
    L1 = loss_backward(model, fixed, a, backward=False)
    print(f"[ft] fixed-batch loss {L0:.5f} -> {L1:.5f}", flush=True)
    if a.save:
        save_pt(model, a.save)
        print(f"[ft] saved {a.save}")


if __name__ == "__main__":
    main()
