"""GPU timing of MOTION-IVP on one Poseidon downstream task (NS-PwC, 128^2), reference port vs accelerated execution
(fast.compile_accelerated), and of the MOTION-NS training step on the training path of the accelerated code
(revision R1, 2026-11-08).

  python -m motion_torch.bench_ivp --ckpt ckpt_70000.npz [--ns_ckpt zsc_ckpt_160000.npz]

MOTION-IVP (stage-3 option C configuration, d 384, depth 8, n_p 32, t_in 10, latent box 64^2) in the downstream
recipe of jobs/run_ft15_motion_v4.sh: hop-1 receding-horizon inference over the 20 frames of a test trajectory
(model.step_ss, lead 1, fed back), one observed frame; fine-tuning update = 10 micro-batches of 4 trajectories
(40 samples, as the Poseidon recipe's batch of 40), each a 10-call receding-horizon loss (rh_roll 10) with
backward, 8,192 encoder points and 2,048 collocation points, then one Adam step. Inference: batch 8, all 16,384
grid nodes queried. Random-valued inputs with the task's channel layout (timings do not depend on the values).
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch

from . import PRESETS
from .convert import build_model
from .data.example import box_dims_for, make_example_ar, stack_batch, synthetic_sample
from .fast import compile_accelerated
from .finetune import rollout_loss_backward, singleshot_loss_backward
from .inference import predict, to_torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--ns_ckpt", default="")
ap.add_argument("--task", default="NS-PwC")
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--micro", type=int, default=4)
ap.add_argument("--accum", type=int, default=10)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--out", default="bench_ivp.json")
a = ap.parse_args()
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
import torch._inductor.config as _icfg
if hasattr(_icfg, "size_asserts"):
    _icfg.size_asserts = False

cfg = PRESETS["motion_ivp_s"].replace(t_in=10, n_p=32, rh_fams=(a.task,))
rng = np.random.default_rng(0)


def batch_of(nb, train):
    exs = []
    for _ in range(nb):
        s = synthetic_sample(a.task, rng, dims=(a.n, a.n), T=31)
        exs.append(make_example_ar(s, cfg.t_in, 20, 4, 2048 if train else 0, 8192 if train else a.n * a.n, rng, cfg,
                                   box_dims=box_dims_for(a.task, cfg), t_in_min=1, ti_fixed=0 if train else 1,
                                   global_norm={a.task: (np.zeros(cfg.S, np.float32), np.ones(cfg.S, np.float32))}))
    return to_torch(stack_batch(exs), dev)


bi, bt = batch_of(a.batch, False), batch_of(a.micro, True)
print("mode", bi["mode"], "box", bi["box_dims"], "roles", bi["roles"], "nodes", tuple(bi["coords_node"].shape),
      "window", tuple(bi["x_win"].shape), flush=True)
base = build_model(cfg, a.ckpt, device=dev).eval()
roles, box = list(bi["roles"]), tuple(bi["box_dims"])


def sync():
    torch.cuda.synchronize()


def t_infer(m, reps):
    def run():
        with torch.no_grad():
            return predict(m, bi, with_metrics=False)["pred"]
    for _ in range(3):
        out = run()
    sync(); ts = []
    for _ in range(reps):
        sync(); t = time.perf_counter(); run(); sync(); ts.append(time.perf_counter() - t)
    return float(np.median(ts)), out.float()


def t_update(m, b, loss_fn, bf16, reps):
    m.train()
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-6)

    def micro():
        with torch.autocast(dev, torch.bfloat16, enabled=bf16):
            return loss_fn(m, b)
    for _ in range(3):
        micro(); opt.zero_grad(set_to_none=True)
    sync(); ts = []
    for _ in range(reps):
        sync(); t = time.perf_counter(); micro(); sync(); ts.append(time.perf_counter() - t)
        opt.zero_grad(set_to_none=True)
    tm = float(np.median(ts))
    micro(); sync(); t = time.perf_counter(); opt.step(); sync(); topt = time.perf_counter() - t
    opt.zero_grad(set_to_none=True)
    m.eval()
    return tm, topt


ivp_loss = lambda m, b: singleshot_loss_backward(m, b, clamp=4.0, per_channel=False, l1=True, k_seg=20,
                                                 rh_roll=10, rh_leads=1)
res = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "task": a.task, "n": a.n,
       "infer_batch": a.batch, "micro_batch": a.micro, "accum": a.accum, "ivp": {}, "ns": {}}


def record(dst, name, ti=None, p=None, p_ref=None, tm=None, topt=None, nb=1):
    r = {}
    if ti is not None:
        r.update(infer_s_per_traj=ti / nb, rel_diff_vs_ref_f32=float((p - p_ref).norm() / p_ref.norm()))
    if tm is not None:
        r.update(micro_s=tm, opt_step_s=topt, update_s=a.accum * tm + topt)
    dst[name] = r
    print(json.dumps({name: r}), flush=True)


# ---------------- MOTION-IVP: inference (20-frame hop-1 receding horizon) and fine-tuning update
ti, p_ref = t_infer(base, a.reps)
record(res["ivp"], "reference f32 (inference)", ti, p_ref, p_ref, nb=a.batch)
try:
    tm, to = t_update(copy.deepcopy(base), bt, ivp_loss, True, a.reps)
    record(res["ivp"], "reference eager, bf16 autocast (update)", tm=tm, topt=to)
except Exception as ex:
    print("CASE_ERROR ref update", repr(ex)[:400], flush=True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
for md, label in (("f32", "accelerated f32/TF32 (inference)"), ("infer_bf16", "accelerated bf16 weights (inference)")):
    try:
        torch._dynamo.reset()
        mi = compile_accelerated(copy.deepcopy(base), roles, box, mode=md)
        ti, p = t_infer(mi, a.reps)
        record(res["ivp"], label, ti, p, p_ref, nb=a.batch)
        res["ivp"][label]["rel_diff_per_hop"] = [float((p[:, k] - p_ref[:, k]).norm() / p_ref[:, k].norm())
                                                 for k in range(p.shape[1])]
    except Exception as ex:
        import traceback
        print("CASE_ERROR acc infer", md, traceback.format_exc()[-1500:], flush=True)
try:
    torch._dynamo.reset()
    mt = compile_accelerated(copy.deepcopy(base), roles, box, mode="train")
    tm, to = t_update(mt, bt, ivp_loss, True, a.reps)
    record(res["ivp"], "accelerated, bf16 autocast (update)", tm=tm, topt=to)
except Exception as ex:
    import traceback
    print("CASE_ERROR acc update", traceback.format_exc()[-1500:], flush=True)

# ---------------- MOTION-NS training step on the training path (fused weights rebuilt in the forward)
if a.ns_ckpt:
    torch._dynamo.reset()
    cfg_ns = PRESETS["motion_ns"]
    exs = [make_example_ar(synthetic_sample("incom_ns", rng, dims=(128, 128), T=20), cfg_ns.t_in, 10, 4, 0, 128 * 128,
                           rng, cfg_ns, box_dims=box_dims_for("incom_ns", cfg_ns)) for _ in range(8)]
    bn = to_torch(stack_batch(exs), dev)
    ns = build_model(cfg_ns, a.ns_ckpt, device=dev).eval()
    ns_loss = lambda m, b: rollout_loss_backward(m, b, k_seg=10)
    def t_inf_ns(m):
        def run():
            with torch.no_grad():
                return predict(m, bn, with_metrics=False)["pred"]
        for _ in range(3):
            out = run()
        sync(); ts = []
        for _ in range(a.reps * 2):
            sync(); t = time.perf_counter(); run(); sync(); ts.append(time.perf_counter() - t)
        return float(np.median(ts)), out.float()
    torch.backends.cuda.matmul.allow_tf32 = False
    ti, pn_ref = t_inf_ns(ns)
    record(res["ns"], "MOTION-NS reference f32 (inference, batch 8)", ti, pn_ref, pn_ref, nb=8)
    torch.backends.cuda.matmul.allow_tf32 = True
    for md in ("f32", "infer_bf16"):
        try:
            torch._dynamo.reset()
            ti, p = t_inf_ns(compile_accelerated(copy.deepcopy(ns), list(bn["roles"]), tuple(bn["box_dims"]), mode=md))
            record(res["ns"], f"MOTION-NS accelerated {md} (inference, batch 8)", ti, p, pn_ref, nb=8)
        except Exception as ex:
            import traceback
            print("CASE_ERROR ns infer", md, traceback.format_exc()[-1500:], flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    tm, to = t_update(copy.deepcopy(ns), bn, ns_loss, True, a.reps)
    record(res["ns"], "MOTION-NS reference eager, bf16 autocast (step, batch 8)", tm=tm, topt=to)
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch._dynamo.reset()
        mt = compile_accelerated(copy.deepcopy(ns), list(bn["roles"]), tuple(bn["box_dims"]), mode="train")
        tm, to = t_update(mt, bn, ns_loss, True, a.reps)
        record(res["ns"], "MOTION-NS accelerated, bf16 autocast (step, batch 8)", tm=tm, topt=to)
    except Exception as ex:
        import traceback
        print("CASE_ERROR ns train", traceback.format_exc()[-1500:], flush=True)
json.dump(res, open(a.out, "w"), indent=1)
print("BENCH_IVP_DONE", json.dumps(res), flush=True)
