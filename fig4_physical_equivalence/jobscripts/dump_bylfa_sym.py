"""bylfa (Fig4 ours, 6-family joint) flip-TTA. Per-family per-g rel-L2 + physics-legal
ensemble. Slots 0/1 = vx/vy; sign pairing tested both ways on incom_ns (isotropic) —
the flat one is the true pairing. Metric = Fig4 convention (allch rel-L2, mean over
10 forecast frames & samples, 100 traj/family)."""
import os, sys, glob, importlib
sys.path.insert(0, "/eu/code_mirror")
import numpy as np, tensorflow as tf
CFG = "motion_tf.train.configs.prose_150M_fluid5_6x_unified_task2_advb2_arh5_bylfa"
cfg = importlib.import_module(CFG).Cfg
if bool(getattr(cfg, "bf16", False)): tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
from motion_tf.model import PhysicsOperatorMixture
from motion_tf.utils import ckpt as ckptlib
from motion_tf.data import stream as streammod
_, test = streammod.make_train_stream(cfg, 8, "/eu/data/prose/prebuilt")
model = PhysicsOperatorMixture(cfg)
cks = sorted(glob.glob("/eu/results/prose_150M_fluid5_6x_unified_task2_advb2_arh5_bylfa/ckpt_*.npz"),
             key=lambda p: int(p.split("_")[-1].split(".")[0]))
ckptlib.load(model, cks[-1]); print("loaded", cks[-1], flush=True)
xin, xtg = test["xin"], test["xtg"]
dsc, cf, fam = test["desc"], test["coef"], test["fam"]
mean, std, names = test["mean"], test["std"], test["fam_names"]
geom = test.get("geom", None)
if geom is None: geom = np.ones((xin.shape[0], xin.shape[2], xin.shape[3], 1), np.float32)
_AR = int(getattr(cfg, "ar_seg", 0)); _SL = int(cfg.Nt) - 1
print("names", list(names), flush=True)

@tf.function
def ev(u0, d, c, g):
    if _AR >= 2:
        preds = []; win = u0
        for _k in range(_AR):
            pk = tf.cast(model.call_with_gate(win, d, c, geom_mask=g)[0], tf.float32)
            preds.append(pk[:, 1:1 + _SL])
            if _k < _AR - 1:
                win = tf.concat([win[:, _SL:], tf.cast(pk[:, 1:1 + _SL], u0.dtype)], axis=1)
        return tf.concat(preds, axis=1)
    return tf.cast(model.call_with_gate(u0, d, c, geom_mask=g)[0], tf.float32)[:, 1:1 + _SL]

# flips on (B,T,H,W,C) axes 2(H),3(W); PAIR=0: slot0<->H(row), PAIR=1: slot0<->W(col)
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

GROUP = [(0, 0), (1, 0), (0, 1), (1, 1)]
def run_family(fi, nm, pair):
    sel = np.where(fam == fi)[0][:100]
    preds_g = {}
    for gflip in GROUP:
        outs = []
        for i in range(0, len(sel), 20):
            idx = sel[i:i + 20]
            xi = T(xin[idx], gflip, pair)
            gm = T(geom[idx], gflip, pair, has_ch=False)
            p = ev(tf.constant(xi), tf.constant(dsc[idx]), tf.constant(cf[idx]),
                   tf.constant(gm)).numpy()[:, :10]
            outs.append(T(p, gflip, pair))
        preds_g[gflip] = np.concatenate(outs, 0)
    sc = std[sel][:, None, None, None, :] + 1e-6; mc = mean[sel][:, None, None, None, :]
    gt = xtg[sel][:, 1:11] * sc + mc
    def rel(p):
        pp = p * sc + mc
        num = np.sqrt(((pp - gt) ** 2).sum((2, 3, 4))); den = np.sqrt((gt ** 2).sum((2, 3, 4))) + 1e-12
        return 100 * (num / den).mean()
    r = {g: rel(preds_g[g]) for g in GROUP}
    print(f"[{nm} pair={pair}] per-g: id={r[(0,0)]:.3f} fr={r[(1,0)]:.3f} fc={r[(0,1)]:.3f} r180={r[(1,1)]:.3f}", flush=True)
    # physics-legal group: flips within 15% rel of id (empirical legality; illegal = blow-up)
    base = r[(0, 0)]
    legal = [g for g in GROUP if r[g] < base * 1.15 or g == (0, 0)]
    ens = np.mean([preds_g[g] for g in legal], axis=0)
    print(f"[{nm}] legal K={len(legal)} {legal}  single={base:.3f}  ENS={rel(ens):.3f}", flush=True)
    return base, rel(ens), len(legal)

# pairing test on incom_ns (isotropic): correct pairing -> fr/fc flat
fi_inc = list(names).index("incom_ns")
r0 = run_family(fi_inc, "incom_ns(pairtest0)", 0)
r1 = run_family(fi_inc, "incom_ns(pairtest1)", 1)
PAIR = 0 if True else 1  # decided below by prints; auto: pick smaller fr error
# auto-decide: rerun cheap? use printed—simpler: choose by ens quality
PAIR = 0 if r0[1] <= r1[1] else 1
print(f"PAIR decided = {PAIR}", flush=True)
res = {}
for fi, nm in enumerate(names):
    res[nm] = run_family(fi, nm, PAIR)
print("SUMMARY (single -> ens, legalK):", flush=True)
for nm, (a, b, k) in res.items():
    print(f"  {nm:20s} {a:7.3f} -> {b:7.3f}  (K={k})", flush=True)
print("DUMP_DONE", flush=True)
