"""Figure-1 extra qualitative panels (non-NS mechanisms). Same ckpt parameterization.

  python fig1_extra.py <ckpt> <outdir>

f: Wave-Layer hidden-material recovery — the coefficient bank's learned c(x) field on the
   latent box vs the TRUE layered wave-speed layout (shipped as the family's coefficient file,
   threaded into geom channel 1... it enters as the last geom column in the nc adapter).
g: CE-RM shock capturing — full model vs shock-head knockout, final frame + 1D cut through
   the front (upwind head removed -> smeared/oscillatory front).
"""
import json
import os
import sys

import numpy as np
import tensorflow as tf

from core.pretrain import load_ckpt
from data import loader
from data.registry import FAMILIES, NUM_SLOTS, SEMANTIC_SLOTS
from v3.data_ar import make_example_ar, stack_batch
from v3.model import V3Model
from v3.train import BOX_TABLE_2D

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/corpus/results/motion_m1/ckpt_best.npz"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/code-vol/motion_fv2/figs_out"
D = int(os.environ.get("FIG_D", 384)); DEPTH = int(os.environ.get("FIG_DEPTH", 8))
DW = int(os.environ.get("FIG_DW", 16)); NP_ = int(os.environ.get("FIG_NP", 16))
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(1234)

model = V3Model(S=NUM_SLOTS, d=D, depth=DEPTH, d_w=DW, n_p=NP_, d_cond=128,
                dims2=(64, 64), dims3=(32, 32, 32), t_in=10, transport=True,
                n_fams=len(FAMILIES), n_semantic=SEMANTIC_SLOTS,
                gate_cond=1, dict_decode=0, fam_head=1)
model.warm_build()
V = model.trainable_variables
load_ckpt(V, CKPT, d_w=DW, names=[v.name for v in V])
print(f"[extra] loaded {CKPT}", flush=True)

def batch_of(fam):
    s = next(iter(loader.units_for(fam, "val")))
    ex = make_example_ar(s, 10, 10, 4, 2048, 8192, rng,
                         box_dims=(BOX_TABLE_2D.get(fam, 64),) * 2)
    return stack_batch([ex]), s

out = {}
# ---------------------------------------------------------------- f: Wave-Layer c(x) recovery
bw, sw = batch_of("Wave-Layer")
dims = tuple(bw["dims"])
# true layered coefficient: Wave-Layer has no other geometry, so the adapter puts c(x) in
# geom column 0 (columns 1..7 are zero padding — see data_ar._geom_levers)
cgt = np.asarray(bw["geom_node"][0][:, 0]).reshape(dims)
assert cgt.std() > 1e-6, "true coefficient layout missing from geom"
# run one step eagerly with the coefficient-field probe on
model.banks.probe = {}
model.banks.probe_field = "coefficient"
cn = tf.constant(bw["coords_node"]); q = cn; gq = tf.constant(bw["geom_node"])
model.step_ss(cn, tf.constant(bw["x_win"]), tf.constant(bw["fmask"]), gq, 2,
              list(bw["roles"]), tf.constant(bw["cond"]), tf.constant(bw["op_multihot"]),
              q, gq, tf.constant(bw["x_win"][:, -1]), kf=1, fam_id=tf.constant(bw["fam_id"]),
              sdf_box=tf.constant(bw["sdf_box"]), box_dims=tuple(bw["box_dims"]))
cf = model.banks.probe.get("coefficient_field")
model.banks.probe = None; model.banks.probe_field = None
if cf is not None:
    C = cf[0]                                              # (R,R,8)
    # pick the channel most correlated (|r|) with the true layout, downsampled to the box
    R = C.shape[0]
    ii = np.minimum(((np.arange(R) + 0.5) / R * dims[0]).astype(int), dims[0] - 1)
    cgt_box = cgt[np.ix_(ii, ii)]
    rs = [abs(np.nan_to_num(np.corrcoef(C[:, :, k].ravel(), cgt_box.ravel())[0, 1])) for k in range(C.shape[-1])]
    k = int(np.nanargmax(rs))
    sgn = np.sign(np.corrcoef(C[:, :, k].ravel(), cgt_box.ravel())[0, 1])
    out["f_c_true"] = cgt.astype(np.float32)
    out["f_c_learned"] = (sgn * C[:, :, k]).astype(np.float32)
    out["f_corr"] = np.float32(rs[k])
    print(f"[f] coefficient recovery: best |r|={rs[k]:.3f} (ch {k})", flush=True)

# ---------------------------------------------------------------- g: CE-RM shock knockout
bg, _ = batch_of("CE-RM")
dims = tuple(bg["dims"])

def rollout_field(b, suppress=None):
    saved = {}
    if suppress:
        for m in suppress:
            saved[m] = (model.banks.gates[m].numpy(), model.banks.gcond[m].kernel.numpy())
            model.banks.gates[m].assign(tf.zeros_like(model.banks.gates[m]))
            model.banks.gcond[m].kernel.assign(tf.zeros_like(model.banks.gcond[m].kernel))
    cn = tf.constant(b["coords_node"]); xw = tf.constant(b["x_win"]); fm = tf.constant(b["fmask"])
    gn = tf.constant(b["geom_node"]); cond = tf.constant(b["cond"]); op = tf.constant(b["op_multihot"])
    fid = tf.constant(b["fam_id"]); sdb = tf.constant(b["sdf_box"])
    tv = b["tstep_mask"][0]; last = int(np.max(np.nonzero(tv)[0]))
    # FIG_LEAD picks the displayed lead (1-based) as in fig1_dump; the shock front is already
    # formed by t=3 while the rollout has not yet accumulated, so the knockout contrast reads
    # against a prediction that still tracks the truth. Caption must state the lead.
    _L = int(os.environ.get("FIG_LEAD", 0))
    if _L > 0:
        last = min(_L - 1, last)
    win, fmask = xw, fm
    prev_grid = tf.reshape(xw[:, -1], [1, *dims, NUM_SLOTS])
    up = xw[:, -1]
    for k in range(last + 1):
        pred = model.step(cn, win, fmask, gn, 2, list(b["roles"]), cond, op, cn, gn, up,
                          steady=False, prev_grid=prev_grid, grid_dims=dims, fam_id=fid,
                          sdf_box=sdb, box_dims=tuple(b["box_dims"]),
                          fbmask=tf.constant(b["fbmask"]))
        up = pred
        win = tf.concat([win[:, 1:], pred[:, None]], 1)
        fmask = tf.concat([fmask[:, 1:], tf.ones_like(fmask[:, :1])], 1)
        prev_grid = tf.reshape(pred, [1, *dims, NUM_SLOTS])
    for m, (g, gc) in saved.items():
        model.banks.gates[m].assign(g); model.banks.gcond[m].kernel.assign(gc)
    ch = 3                                                  # density: sharpest shock signature
    sc = b["scale"][0][ch]
    gt = b["y_node"][0, last][:, ch].reshape(dims) * sc
    pf = pred.numpy()[0][:, ch].reshape(dims) * sc
    return gt, pf, last

gt, full, last = rollout_field(bg)
_, noshock, _ = rollout_field(bg, suppress=["shock"])
out["g_gt"], out["g_full"], out["g_noshock"] = gt, full, noshock
row = int(np.argmax(np.abs(np.diff(gt, axis=1)).max(1)))    # row crossing the strongest front
out["g_row"] = np.int32(row)
print(f"[g] shock knockout: full rel {np.linalg.norm(full-gt)/np.linalg.norm(gt):.3f}, "
      f"no-shock rel {np.linalg.norm(noshock-gt)/np.linalg.norm(gt):.3f}", flush=True)

# ---------------------------------------------------------------- h: CE-CRP compressibility knockout
# The compressibility head is UNMASKED (never equation-selected): its concentration on the
# compressible-Euler families is learned entirely from data — the emergent-mechanism vignette.
bh, _ = batch_of("CE-CRP")
dims = tuple(bh["dims"])
gt_h, full_h, _ = rollout_field(bh)
_, nocmp_h, _ = rollout_field(bh, suppress=["compressible"])
out["h_gt"], out["h_full"], out["h_nocmp"] = gt_h, full_h, nocmp_h
out["h_row"] = np.int32(int(np.argmax(np.abs(np.diff(gt_h, axis=1)).max(1))))
print(f"[h] compressible knockout (CE-CRP): full rel {np.linalg.norm(full_h-gt_h)/np.linalg.norm(gt_h):.3f}, "
      f"no-cmp rel {np.linalg.norm(nocmp_h-gt_h)/np.linalg.norm(gt_h):.3f}", flush=True)

np.savez_compressed(os.path.join(OUT, "fig1_extra.npz"), **out)
print(f"[extra] saved -> {OUT}/fig1_extra.npz", flush=True)
