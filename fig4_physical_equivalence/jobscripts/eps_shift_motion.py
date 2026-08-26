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

# D4 on (B,T,H,W,C); g=(t,fr,fc), forward = flips(transpose(x)); slot0<->row, slot1<->col
def T(a, g, inv=False, has_ch=True):
    t, fr, fc = g
    if t == 'sh':                                   # periodic translation by (fr, fc) cells
        sgn = 1 if inv else -1
        return np.roll(a, (sgn * fr, sgn * fc), axis=(a.ndim - 3, a.ndim - 2))
    def _fl(a):
        if fr:
            a = np.flip(a, axis=a.ndim - 3).copy()
            if has_ch: a[..., 0] *= -1
        if fc:
            a = np.flip(a, axis=a.ndim - 2).copy()
            if has_ch: a[..., 1] *= -1
        return a
    def _tr(a):
        if t:
            a = np.swapaxes(a, a.ndim - 3, a.ndim - 2).copy()
            if has_ch:
                sw = list(range(a.shape[-1])); sw[0], sw[1] = 1, 0
                a = a[..., sw]
        return a
    return _tr(_fl(a)) if inv else _fl(_tr(a))

GROUP = [(0, 0, 0), ('sh', 1, 1), ('sh', 8, 8)]   # identity + periodic shifts
def run_family(fi, nm):
    sel = np.where(fam == fi)[0][:100]
    preds_g = {}
    for g in GROUP:
        outs = []
        for i in range(0, len(sel), 20):
            idx = sel[i:i + 20]
            xi = T(xin[idx], g)
            gm = T(geom[idx], g, has_ch=False)
            p = ev(tf.constant(xi), tf.constant(dsc[idx]), tf.constant(cf[idx]),
                   tf.constant(gm)).numpy()[:, :10]
            outs.append(T(p, g, inv=True))
        preds_g[g] = np.concatenate(outs, 0)
    sc = std[sel][:, None, None, None, :] + 1e-6; mc = mean[sel][:, None, None, None, :]
    gt = xtg[sel][:, 1:11] * sc + mc
    def rel(p):
        pp = p * sc + mc
        num = np.sqrt(((pp - gt) ** 2).sum((2, 3, 4))); den = np.sqrt((gt ** 2).sum((2, 3, 4))) + 1e-12
        return 100 * (num / den).mean()
    # defect in the SAME convention as rel(): physical rel-L2, active channels,
    # frame-mean then sample-mean
    _m5 = (np.abs(std[sel]) > 0)[:, None, None, None, :].astype(np.float64)
    def _eps_of(Pg, P0):
        A = Pg.astype(np.float64) * sc + mc; B = P0.astype(np.float64) * sc + mc
        num = np.sqrt((((A - B) ** 2) * _m5).sum(axis=(2, 3, 4)))
        den = np.sqrt((((B) ** 2) * _m5).sum(axis=(2, 3, 4))) + 1e-12
        return float((num / den).mean() * 100)
    _lb = {("sh",1,1):"shift1", ("sh",8,8):"shift8"}
    _P0 = preds_g[(0, 0, 0)]
    _eps = {_lb[g]: _eps_of(preds_g[g], _P0) for g in GROUP if g != (0, 0, 0)}
    print(f"[{nm}] EPS_EQUIV(%, family metric):",
          {k: round(v, 3) for k, v in sorted(_eps.items(), key=lambda kv: kv[1])}, flush=True)
    r = {g: rel(preds_g[g]) for g in GROUP}
    lbl = {(0,0,0):"id", ("sh",1,1):"shift1", ("sh",8,8):"shift8"}
    print(f"[{nm}] per-g:", {lbl[g]: round(float(v), 3) for g, v in r.items()}, flush=True)
    ens4 = preds_g[(0, 0, 0)]
    ens8 = preds_g[(0, 0, 0)]
    print(f"[{nm}] single={r[(0,0,0)]:.3f} K4-ENS={rel(ens4):.3f} K8-ENS={rel(ens8):.3f}", flush=True)

for nm in [n for n in names if n != "cfdbench"]:
    run_family(list(names).index(nm), nm)
print("DUMP_DONE", flush=True)
