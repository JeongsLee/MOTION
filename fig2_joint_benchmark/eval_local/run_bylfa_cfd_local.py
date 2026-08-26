"""MOTION(bylfa) cfdbench conditional flip ensemble — LOCAL run.

Idea (user): detect per-case mean-flow direction from the INPUT window only
(no GT), then apply only the flow-perpendicular (spanwise) mirror — same
principle as the buoyant family's gravity-perpendicular K2 group.

Steps:
  1) reproduce id score (sanity vs job record)
  2) classify each of 100 cases: directional (dominant mean-flow axis) vs
     ambiguous (cavity-like recirculation) using fluid-region mean velocity
  3) per-g diagnostics on the directional subset under BOTH velocity-axis
     pairings (was the earlier all-mirror blowup a pairing artifact or real?)
  4) conditional K2 ensemble: directional cases avg(id, spanwise-mirror),
     ambiguous cases keep id; report overall + subset scores
"""
import sys, importlib, numpy as np
sys.path.insert(0, "/mnt/d/working/2026/neuraladaf/v6/fig3_local/bylfa_code")
sys.modules.pop("code", None)
import tensorflow as tf
CFG = "motion_tf.train.configs.prose_150M_fluid5_6x_unified_task2_advb2_arh5_bylfa"
cfg = importlib.import_module(CFG).Cfg
if bool(getattr(cfg, "bf16", False)):
    tf.keras.mixed_precision.set_global_policy("mixed_bfloat16"); print("bf16 on", flush=True)
from motion_tf.model import PhysicsOperatorMixture
from motion_tf.utils import ckpt as ckptlib
model = PhysicsOperatorMixture(cfg)
d = np.load("/mnt/d/working/2026/neuraladaf/v6/fig3_local/bylfa_cfd_test.npz")
xin, xtg = d["xin"], d["xtg"]
mean, std, dsc, cf = d["mean"], d["std"], d["desc"], d["coef"]
geom = d["geom"]
if geom.ndim == 3: geom = geom[..., None]
print("geom stats: min", geom.min(), "mean", geom.mean(), flush=True)
_AR = int(getattr(cfg, "ar_seg", 0)); _SL = int(cfg.Nt) - 1
_ = model.call_with_gate(tf.constant(xin[:1]), tf.constant(dsc[:1]), tf.constant(cf[:1]),
                         geom_mask=tf.constant(geom[:1]))
ckptlib.load(model, "/mnt/d/working/2026/neuraladaf/v6/pdefoundation/paper/weights/bylfa_final/ckpt_160000.npz")
print("loaded ckpt", flush=True)

@tf.function
def ev(u0, dd, cc, g):
    if _AR >= 2:
        preds = []; win = u0
        for _k in range(_AR):
            pk = tf.cast(model.call_with_gate(win, dd, cc, geom_mask=g)[0], tf.float32)
            preds.append(pk[:, 1:1 + _SL])
            if _k < _AR - 1:
                win = tf.concat([win[:, _SL:], tf.cast(pk[:, 1:1 + _SL], u0.dtype)], axis=1)
        return tf.concat(preds, axis=1)
    return tf.cast(model.call_with_gate(u0, dd, cc, geom_mask=g)[0], tf.float32)[:, 1:1 + _SL]

# flips on (B,T,H,W,C); PAIR=0: slot0<->H(row), PAIR=1: slot0<->W(col)
def T(a, gflip, pair, has_ch=True):
    fr, fc = gflip
    axH = a.ndim - 3; axW = a.ndim - 2
    if fr:
        a = np.flip(a, axis=axH).copy()
        if has_ch: a[..., 0 if pair == 0 else 1] *= -1
    if fc:
        a = np.flip(a, axis=axW).copy()
        if has_ch: a[..., 1 if pair == 0 else 0] *= -1
    return a

def predict(idx, gflip, pair):
    outs = []
    for i in range(0, len(idx), 4):
        j = idx[i:i + 4]
        xi = T(xin[j], gflip, pair)
        gm = T(geom[j], gflip, pair, has_ch=False)
        p = ev(tf.constant(xi), tf.constant(dsc[j]), tf.constant(cf[j]),
               tf.constant(gm)).numpy()[:, :10]
        outs.append(T(p, gflip, pair))
    return np.concatenate(outs, 0)

VT = 10
sc_all = std[:, None, None, None, :] + 1e-6; mc_all = mean[:, None, None, None, :]
gt_all = xtg[:, 1:1 + VT] * sc_all + mc_all
def rel(p, idx):
    pp = p * sc_all[idx] + mc_all[idx]
    num = np.sqrt(((pp - gt_all[idx]) ** 2).sum((2, 3, 4)))
    den = np.sqrt((gt_all[idx] ** 2).sum((2, 3, 4))) + 1e-12
    return 100 * (num / den).mean()

# ---- 2) flow-direction classification from input window only ----
# physical velocities in fluid region; slot0 = H-axis vel under PAIR=0 convention
phys_in = xin * (std[:, None, None, None, :] + 1e-6) + mean[:, None, None, None, :]
fl = (geom[:, None, ..., 0] > 0.5)  # (B,1,H,W) fluid mask broadcast over T
vH = np.array([phys_in[b, ..., 0][fl[b].repeat(xin.shape[1], 0)].mean() for b in range(len(xin))])
vW = np.array([phys_in[b, ..., 1][fl[b].repeat(xin.shape[1], 0)].mean() for b in range(len(xin))])
ratio = np.abs(vW) / (np.abs(vH) + 1e-9)
dirW = ratio > 3.0          # strong flow along W (cols) -> spanwise mirror = row flip
dirH = (1.0 / (ratio + 1e-12)) > 3.0
amb = ~(dirW | dirH)
print(f"classification: flow-along-W={dirW.sum()} flow-along-H={dirH.sum()} ambiguous={amb.sum()}", flush=True)
print("ratio quantiles:", np.round(np.quantile(ratio, [0, .25, .5, .75, 1]), 2), flush=True)

# ---- 1)+3) predictions ----
allidx = np.arange(len(xin))
P = {}
for pair in (0, 1):
    for g in [(0, 0), (1, 0), (0, 1)]:
        if g == (0, 0) and pair == 1: continue
        P[(g, pair)] = predict(allidx, g, pair)
        print(f"g={g} pair={pair}: rel(all)={rel(P[(g,pair)], allidx):.3f}", flush=True)
Pid = P[((0, 0), 0)]
print(f"id overall = {rel(Pid, allidx):.3f}  (job record 0.106-scale sanity)", flush=True)

for pair in (0, 1):
    for g in [(1, 0), (0, 1)]:
        if dirW.sum():
            print(f"  dirW subset g={g} pair={pair}: {rel(P[(g,pair)], np.where(dirW)[0]):.3f} vs id {rel(Pid, np.where(dirW)[0]):.3f}", flush=True)
        if dirH.sum():
            print(f"  dirH subset g={g} pair={pair}: {rel(P[(g,pair)], np.where(dirH)[0]):.3f} vs id {rel(Pid, np.where(dirH)[0]):.3f}", flush=True)

# ---- 4) conditional ensemble: spanwise mirror per detected flow axis ----
for pair in (0, 1):
    ens = Pid.copy()
    if dirW.sum(): ens[dirW] = (Pid[dirW] + P[((1, 0), pair)][dirW]) / 2  # flow||W -> flip rows
    if dirH.sum(): ens[dirH] = (Pid[dirH] + P[((0, 1), pair)][dirH]) / 2  # flow||H -> flip cols
    print(f"[pair={pair}] conditional K2 ENS overall = {rel(ens, allidx):.3f}  (single {rel(Pid, allidx):.3f})", flush=True)
    if dirW.sum():
        print(f"   directional-W subset: ens {rel(ens, np.where(dirW)[0]):.3f} vs id {rel(Pid, np.where(dirW)[0]):.3f}", flush=True)
np.savez("/mnt/d/working/2026/neuraladaf/v6/fig3_local/cfd_cond_sym_out.npz",
         dirW=dirW, dirH=dirH, ratio=ratio)
print("DONE", flush=True)
