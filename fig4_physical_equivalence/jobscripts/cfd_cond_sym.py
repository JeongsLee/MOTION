"""bylfa cfdbench conditional flip-TTA (server).
Per-case mean-flow axis detected from INPUT window only (no GT); apply only the
flow-perpendicular mirror -> conditional K2, like the buoyant family's rule.
Both velocity-axis pairings diagnosed. Metric = Fig4 convention."""
import sys, glob, importlib
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
names = list(test["fam_names"]); fi = names.index("cfdbench")
sel = np.where(test["fam"] == fi)[0][:100]
xin, xtg = test["xin"][sel], test["xtg"][sel]
dsc, cf = test["desc"][sel], test["coef"][sel]
mean, std = test["mean"][sel], test["std"][sel]
geom = test.get("geom", None)
geom = (np.ones((100, xin.shape[2], xin.shape[3], 1), np.float32) if geom is None
        else geom[sel].astype(np.float32))
if geom.ndim == 3: geom = geom[..., None]
print("geom stats:", geom.min(), geom.mean(), flush=True)
_AR = int(getattr(cfg, "ar_seg", 0)); _SL = int(cfg.Nt) - 1

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

def predict(gflip, pair):
    outs = []
    for i in range(0, 100, 20):
        xi = T(xin[i:i+20], gflip, pair)
        gm = T(geom[i:i+20], gflip, pair, has_ch=False)
        p = ev(tf.constant(xi), tf.constant(dsc[i:i+20]), tf.constant(cf[i:i+20]),
               tf.constant(gm)).numpy()[:, :10]
        outs.append(T(p, gflip, pair))
    return np.concatenate(outs, 0)

sc = std[:, None, None, None, :] + 1e-6; mc = mean[:, None, None, None, :]
gt = xtg[:, 1:11] * sc + mc
def rel(p, m=None):
    pp = p[:, :10] * sc + mc
    num = np.sqrt(((pp - gt) ** 2).sum((2, 3, 4))); den = np.sqrt((gt ** 2).sum((2, 3, 4))) + 1e-12
    r = 100 * (num / den).mean(1)
    return (r if m is None else r[m]).mean()

# flow-axis classification from input window only (physical units, fluid region)
phys = xin * sc + mc
fmask = geom[:, :, :, 0] > 0.5
vH = np.array([phys[b, :, :, :, 0][:, fmask[b]].mean() for b in range(100)])
vW = np.array([phys[b, :, :, :, 1][:, fmask[b]].mean() for b in range(100)])
ratio = np.abs(vW) / (np.abs(vH) + 1e-9)
dirW = ratio > 3.0; dirH = (1.0 / (ratio + 1e-12)) > 3.0; amb = ~(dirW | dirH)
print(f"classify: alongW={dirW.sum()} alongH={dirH.sum()} ambiguous={amb.sum()}", flush=True)
print("ratio q:", np.round(np.quantile(ratio, [0, .25, .5, .75, 1.0]), 2), flush=True)
print("|vH| q:", np.round(np.quantile(np.abs(vH), [0, .5, 1.0]), 4),
      "|vW| q:", np.round(np.quantile(np.abs(vW), [0, .5, 1.0]), 4), flush=True)

P = {}
for pair in (0, 1):
    for g in [(0, 0), (1, 0), (0, 1)]:
        if g == (0, 0) and pair == 1: continue
        P[(g, pair)] = predict(g, pair)
Pid = P[((0, 0), 0)]
print(f"id overall = {rel(Pid):.3f}", flush=True)
for pair in (0, 1):
    for g in [(1, 0), (0, 1)]:
        line = f"g={g} pair={pair}: all={rel(P[(g,pair)]):.3f}"
        if dirW.sum(): line += f"  dirW={rel(P[(g,pair)], dirW):.3f}(id {rel(Pid, dirW):.3f})"
        if dirH.sum(): line += f"  dirH={rel(P[(g,pair)], dirH):.3f}(id {rel(Pid, dirH):.3f})"
        if amb.sum():  line += f"  amb={rel(P[(g,pair)], amb):.3f}(id {rel(Pid, amb):.3f})"
        print(line, flush=True)
for pair in (0, 1):
    ens = Pid.copy()
    if dirW.sum(): ens[dirW] = (Pid[dirW] + P[((1, 0), pair)][dirW]) / 2
    if dirH.sum(): ens[dirH] = (Pid[dirH] + P[((0, 1), pair)][dirH]) / 2
    print(f"[pair={pair}] conditional K2 ENS = {rel(ens):.3f} (single {rel(Pid):.3f})", flush=True)
    if dirW.sum(): print(f"   dirW subset: ens {rel(ens, dirW):.3f} vs id {rel(Pid, dirW):.3f}", flush=True)
print("DUMP_DONE", flush=True)
