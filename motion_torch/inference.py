"""Inference API: AR rollout (K one-frame steps), single-shot (one trunk call, all leads), receding-horizon hop-1
rollout of the single-shot operator, and the relative-error metrics of v3/train.py `_rel` (normalized, physical,
fluctuation conventions) -- mirrors eval_stats.per_sample_ar / per_sample_ss and v3/train.singleshot_rh_loss.

    from motion_torch import PRESETS, build_model
    from motion_torch.data.example import make_example_ar, stack_batch, box_dims_for
    from motion_torch.inference import to_torch, predict
    model = build_model(PRESETS["motion_s"], "m1f_ckpt_best.npz", device="cuda")
    ex = make_example_ar(unit, 10, 10, 4, 2048, 8192, np.random.default_rng(0), cfg, box_dims=box_dims_for(fam, cfg))
    out = predict(model, to_torch(stack_batch([ex]), "cuda"))        # dispatches on the example's mode
    out["pred"]          # (B, k_seg, N_node + Q, S) normalized-space prediction at nodes + collocation points
    out["rel_phys"]      # (B, k_seg) physical-space relative L2 per frame (PROSE/BCAT joint convention)
"""
from __future__ import annotations

import numpy as np
import torch

from .data.example import BATCH_KEYS
from .model.v3model import to_grid

STATIC_KEYS = ("K", "steady", "dense2d", "roles", "box_dims", "surf", "mode", "k_seg", "dims", "families")


def to_torch(b, device="cpu"):
    """stack_batch() dict -> same dict with torch tensors on `device` (statics kept as python values)."""
    out = {k: v for k, v in b.items() if k in STATIC_KEYS}
    for k in BATCH_KEYS:
        if k in b:
            t = torch.from_numpy(np.ascontiguousarray(b[k]))
            out[k] = t.to(device=device, dtype=torch.long if k == "fam_id" else torch.float32)
    return out


def rel(pred, y, cm, per_channel=False, scale=None, fluct=False, l1=False):
    """Masked relative L2 (or L1) error, port of v3/train._rel. pred/y (B,Q,S), cm (B,S).
    scale (B,S) = per-channel std -> physical units; (B,2S) = [scale | shift] for z-scored models."""
    if scale is not None:
        S_ = pred.shape[-1]
        if scale.shape[-1] == 2 * S_:
            sc, sh = scale[:, None, :S_], scale[:, None, S_:]
            pred, y = pred * sc + sh, y * sc + sh
        else:
            sc = scale[:, None, :]
            pred, y = pred * sc, y * sc
    y_den = y - y.mean(1, keepdim=True) if fluct else y
    if l1:
        if not per_channel:
            w = cm[:, None, :]
            return (w * (pred - y).abs()).sum((1, 2)) / ((w * y_den.abs()).sum((1, 2)) + 1e-6)
        r = cm * ((pred - y).abs().sum(1) / (y_den.abs().sum(1) + 1e-6))
        return r.sum(-1) / torch.clamp(cm.sum(-1), min=1.0)
    if not per_channel:
        w = cm[:, None, :]
        num = torch.sqrt((w * torch.square(pred - y)).sum((1, 2)) + 1e-12)
        den = torch.sqrt((w * torch.square(y_den)).sum((1, 2)) + 1e-12)
        return num / den
    num = torch.sqrt(torch.square(pred - y).sum(1) + 1e-12)
    den = torch.sqrt(torch.square(y_den).sum(1) + 1e-12)
    if fluct:
        den = torch.maximum(den, 1e-3 * torch.sqrt(torch.square(y).sum(1) + 1e-12))
    r = cm * (num / den)
    return r.sum(-1) / torch.clamp(cm.sum(-1), min=1.0)


def phys_scale(b, cfg):
    """The `scale` argument of the physical metric: [scale | shift] for z-scored inputs (eval_stats.py)."""
    return torch.cat([b["scale"], b["shift"]], -1) if cfg.input_zscore else b["scale"]


def _metrics(pred, y_all, b, cfg):
    cm = b["cmask"]
    return (rel(pred, y_all, cm, False), rel(pred, y_all, cm, False, phys_scale(b, cfg)),
            rel(pred, y_all, cm, True, fluct=True))


def _common(b):
    qall = torch.cat([b["coords_node"], b["coords_q"]], 1)
    gall = torch.cat([b["geom_node"], b["geom_q"]], 1)
    u_prev = torch.cat([b["x_win"][:, -1], b["ic_q"]], 1)
    return qall, gall, u_prev


def rollout_ar(model, b, k_seg=None, with_metrics=True, grad=False, feedback_detach=True):
    """Free-running AR rollout (eval_stats.per_sample_ar / v3/train.rollout_loss): k_seg one-frame steps, the
    prediction at the nodes is fed back into the window, and on dense 2-D grids into the transport grid.
    Returns {"pred": (B,k,N+Q,S), "rel_norm"/"rel_phys"/"rel_fluct"/"w": (B,k)}."""
    cfg = model.cfg
    K, steady, dense2d = b["K"], b["steady"], b["dense2d"]
    roles, box_dims = list(b["roles"]), tuple(b["box_dims"])
    k_seg = int(k_seg or b["k_seg"])
    gd = tuple(b["dims"]) if b.get("dims") else None
    Nn = b["coords_node"].shape[1]
    qall, gall, u_prev = _common(b)
    win, fmask = b["x_win"], b["fmask"]
    prev_grid = prev2_grid = None
    if dense2d:
        prev_grid = to_grid(win[:, -1], gd, model.S)
        if win.shape[1] >= 2:
            prev2_grid = to_grid(win[:, -2], gd, model.S)
    preds, rn, rp, rf, wt = [], [], [], [], []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for k in range(k_seg):
            pred = model.step(b["coords_node"], win, fmask, b["geom_node"], K, roles, b["cond"], b["op_multihot"],
                              qall, gall, u_prev, steady=steady, prev_grid=prev_grid, grid_dims=gd,
                              fam_id=b["fam_id"], sdf_box=b["sdf_box"], box_dims=box_dims, fbmask=b["fbmask"],
                              prev2_grid=prev2_grid)
            preds.append(pred)
            if with_metrics:
                kt = 0 if steady else k
                y_all = torch.cat([b["y_node"][:, kt], b["y_q"][:, kt]], 1)
                a, p_, f = _metrics(pred, y_all, b, cfg)
                rn.append(a); rp.append(p_); rf.append(f)
                wt.append(torch.full_like(a, float(k + 1)) if steady else b["tstep_mask"][:, k])
            u_prev = pred.detach() if feedback_detach else pred
            pred_n = u_prev[:, :Nn]
            win = torch.cat([win[:, 1:], pred_n[:, None]], 1)
            fmask = torch.cat([fmask[:, 1:], torch.ones_like(fmask[:, :1])], 1)
            if dense2d:
                prev2_grid, prev_grid = prev_grid, to_grid(pred_n, gd, model.S)
    out = {"pred": torch.stack(preds, 1)}
    if with_metrics:
        out.update(rel_norm=torch.stack(rn, 1), rel_phys=torch.stack(rp, 1), rel_fluct=torch.stack(rf, 1),
                   w=torch.stack(wt, 1))
    return out


def singleshot(model, b, k_seg=None, leads=None, with_metrics=True):
    """Single-shot (eval_stats.per_sample_ss / v3/train.singleshot_loss): one trunk call decodes leads 1..k_seg.
    The IC grid is passed on dense 2-D grids when the model has the MOTION_SS_WARP carrier."""
    cfg = model.cfg
    k_seg = int(k_seg or b["k_seg"])
    gd = tuple(b["dims"]) if b.get("dims") else None
    qall, gall, u_prev = _common(b)
    pg = to_grid(b["x_win"][:, -1], gd, model.S) if (gd is not None and model.ss_warp and b["dense2d"]) else None
    with torch.no_grad():
        preds = model.step_ss(b["coords_node"], b["x_win"], b["fmask"], b["geom_node"], b["K"], list(b["roles"]),
                              b["cond"], b["op_multihot"], qall, gall, u_prev, kf=k_seg, fam_id=b["fam_id"],
                              sdf_box=b["sdf_box"], box_dims=tuple(b["box_dims"]), prev_grid=pg, grid_dims=gd,
                              leads=leads)
    out = {"pred": preds}
    if with_metrics:
        ks = list(range(1, k_seg + 1)) if leads is None else list(leads)
        rn, rp, rf, wt = [], [], [], []
        for j, k in enumerate(ks):
            y_all = torch.cat([b["y_node"][:, k - 1], b["y_q"][:, k - 1]], 1)
            a, p_, f = _metrics(preds[:, j], y_all, b, cfg)
            rn.append(a); rp.append(p_); rf.append(f); wt.append(b["tstep_mask"][:, k - 1])
        out.update(rel_norm=torch.stack(rn, 1), rel_phys=torch.stack(rp, 1), rel_fluct=torch.stack(rf, 1),
                   w=torch.stack(wt, 1))
    return out


def rollout_rh(model, b, k_seg=None, n_roll=None, with_metrics=True):
    """Receding-horizon inference (MOTION_RH_FAMS families, v3/train.singleshot_rh_loss eval form): the single-shot
    operator with its own time map (kf = k_seg) is applied to its own lead-1 output n_roll (= k_seg) times."""
    cfg = model.cfg
    k_seg = int(k_seg or b["k_seg"])
    n_roll = min(int(n_roll or k_seg), k_seg)
    gd = tuple(b["dims"]) if b.get("dims") else None
    Nn = b["coords_node"].shape[1]
    qall, gall, u_prev = _common(b)
    win, fmask = b["x_win"], b["fmask"]
    use_pg = gd is not None and model.ss_warp and b["dense2d"]   # (TF: grid_dims only; a sub-sampled 3-D window cannot be gridded)
    pg = to_grid(win[:, -1], gd, model.S) if use_pg else None
    preds, rn, rp, rf, wt = [], [], [], [], []
    with torch.no_grad():
        for r in range(n_roll):
            p = model.step_ss(b["coords_node"], win, fmask, b["geom_node"], b["K"], list(b["roles"]), b["cond"],
                              b["op_multihot"], qall, gall, u_prev, kf=k_seg, fam_id=b["fam_id"],
                              sdf_box=b["sdf_box"], box_dims=tuple(b["box_dims"]), prev_grid=pg, grid_dims=gd,
                              leads=[1])[:, 0]
            preds.append(p)
            if with_metrics:
                y_all = torch.cat([b["y_node"][:, r], b["y_q"][:, r]], 1)
                a, p_, f = _metrics(p, y_all, b, cfg)
                rn.append(a); rp.append(p_); rf.append(f); wt.append(b["tstep_mask"][:, r])
            u_prev = p
            pred_n = u_prev[:, :Nn]
            win = torch.cat([win[:, 1:], pred_n[:, None]], 1)
            fmask = torch.cat([fmask[:, 1:], torch.ones_like(fmask[:, :1])], 1)
            if use_pg:
                pg = to_grid(pred_n, gd, model.S)
    out = {"pred": torch.stack(preds, 1)}
    if with_metrics:
        out.update(rel_norm=torch.stack(rn, 1), rel_phys=torch.stack(rp, 1), rel_fluct=torch.stack(rf, 1),
                   w=torch.stack(wt, 1))
    return out


def predict(model, b, **kw):
    """Dispatch on the batch's temporal mode: 'ar'/'steady' -> rollout_ar, 'ss' -> singleshot, 'rh' -> rollout_rh."""
    mode = b.get("mode", "steady" if b["steady"] else "ar")
    if mode == "ss":
        return singleshot(model, b, **kw)
    if mode == "rh":
        return rollout_rh(model, b, **kw)
    return rollout_ar(model, b, **kw)


def trajectory_error(out, key="rel_phys"):
    """Per-trajectory error: frame-validity weighted mean over frames (eval_stats.py `traj`)."""
    w = out["w"]
    return (out[key] * w).sum(-1) / torch.clamp(w.sum(-1), min=1e-6)


def to_physical(pred, b, cfg):
    """Normalized-space prediction -> physical units (multiply the scale back, add the z-score mean back)."""
    sc = b["scale"][:, None, None, :] if pred.dim() == 4 else b["scale"][:, None, :]
    sh = b["shift"][:, None, None, :] if pred.dim() == 4 else b["shift"][:, None, :]
    return pred * sc + (sh if cfg.input_zscore else 0.0)
