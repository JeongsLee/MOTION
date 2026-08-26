"""D4 (K=8) symmetry-TTA pilot — MOTION side. argv: task cfgmod ckpt do_ar f32(0/1) wv0(0/1)
Group = dihedral D4: element g=(t,fr,fc), forward = flips(transpose(x)).
Transpose swaps H/W axes AND velocity slots 0<->1 (no sign); flips as verified (K4 layout).
Prints per-g, K4 ensemble (flip subgroup) and K8 ensemble medians."""
import os, sys, importlib, numpy as np
sys.path.insert(0, "/tmp/w"); sys.modules.pop("code", None)
import tensorflow as tf
task, cfgmod, ck, do_ar, use_f32, wv0 = (sys.argv[1], sys.argv[2], sys.argv[3],
    int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
m = importlib.import_module(f"motion_tf.train.configs.{cfgmod}"); cfg = m.Cfg
if not use_f32 and bool(getattr(cfg, "bf16", False)):
    tf.keras.mixed_precision.set_global_policy("mixed_bfloat16"); print("bf16 on", flush=True)
from motion_tf.model import PhysicsOperatorMixture
from motion_tf.data.poseidon import build_poseidon_stream
from motion_tf.utils import ckpt as ckptlib
model = PhysicsOperatorMixture(cfg)
_ = model.call_with_gate(tf.zeros((1, cfg.Nx, cfg.Nx, 8)), tf.zeros((1, cfg.desc_dim)),
                         tf.zeros((1, 4)), training=False)
ckptlib.load(model, ck); print("restored", ck, flush=True)
_, test = build_poseidon_stream(cfg, cfg.batch)
xt = np.asarray(test["traj"], np.float32); cm = np.asarray(test["cm"], np.float32)
mean = np.asarray(test["mean"], np.float32); std = np.asarray(test["std"], np.float32)
N = xt.shape[0]; NT = int(getattr(cfg, "nt_data", 0)) or xt.shape[1]
u0 = np.array(xt[:, 0])
if wv0:
    u0[..., int(cfg.v0_to_slot)] = 0.0
print("test", xt.shape, "NT", NT, flush=True)

GROUP = [(t, fr, fc) for t in (0, 1) for fr in (0, 1) for fc in (0, 1)]

# Which velocity slot pairs with which array axis is a property of how a family's file was
# written, and the compressible and incompressible releases do not agree on it.  VEL_SWAP
# exposes the pairing.  The double flip negates both components and is therefore
# pairing-independent, which is what makes it the reference for reading the single flips.
_SW = int(os.environ.get("VEL_SWAP", "0"))
_A, _B = (1, 0) if _SW else (0, 1)

def _flips(a, fr, fc):
    if fr: a = np.flip(a, axis=a.ndim - 3).copy(); a[..., _A] = -a[..., _A]
    if fc: a = np.flip(a, axis=a.ndim - 2).copy(); a[..., _B] = -a[..., _B]
    return a

def _tr(a, t):
    if t:
        a = np.swapaxes(a, a.ndim - 3, a.ndim - 2).copy()
        sw = list(range(a.shape[-1])); sw[0], sw[1] = 1, 0
        a = a[..., sw]
    return a

def T(a, g, inv=False):
    t, fr, fc = g
    return _tr(_flips(a, fr, fc), t) if inv else _flips(_tr(a, t), fr, fc)

def predict(u):
    o = []
    for i in range(0, len(u), 8):
        b = tf.constant(u[i:i+8])
        o.append(tf.cast(model.call_with_gate(b, tf.zeros((len(b), cfg.desc_dim)),
                                              tf.zeros((len(b), 4)), training=False)[0],
                         tf.float32).numpy())
    return np.concatenate(o)[:, :NT]

FAM = np.asarray(test["fam"], np.int32)
FNAMES = list(test.get("fam_names", []))
print("families in test:", {FNAMES[k] if k < len(FNAMES) else k: int((FAM == k).sum())
                            for k in sorted(set(FAM.tolist()))}, flush=True)

preds_g = {g: T(predict(T(u0, g)), g, inv=True) for g in GROUP}
_lb = {(0,1,0):"fr",(0,0,1):"fc",(0,1,1):"r180",(1,0,0):"d1",(1,1,0):"r90",(1,0,1):"r270",(1,1,1):"d2"}

def _rel(P, R, sel):
    """rel-L1 in the task convention, restricted to the samples of one family."""
    sc = std[sel][:, None, None, None, :]; mc = mean[sel][:, None, None, None, :]
    m5 = cm[sel][:, None, None, None, :]
    A = P[sel][:, 1:] * sc + mc; B = R[sel][:, 1:] * sc + mc
    num = np.sum(np.abs(A - B) * m5, axis=(2, 3, 4))
    den = np.sum(np.abs(B) * m5, axis=(2, 3, 4)) + 1e-8
    return float(np.median((num / den).mean(axis=1)) * 100)

P0 = preds_g[(0, 0, 0)]
for k in sorted(set(FAM.tolist())):
    nm = FNAMES[k] if k < len(FNAMES) else str(k)
    sel = np.where(FAM == k)[0]
    err = _rel(P0, xt[:, :NT], sel)
    eps = {_lb[g]: _rel(preds_g[g], P0, sel) for g in GROUP if g != (0, 0, 0)}
    perg = {_lb[g]: _rel(preds_g[g], xt[:, :NT], sel) for g in GROUP if g != (0, 0, 0)}
    print(f"[{nm}] n={len(sel)}  err_id={err:.3f}%", flush=True)
    print(f"[{nm}] EPS:", {a: round(b, 3) for a, b in sorted(eps.items(), key=lambda kv: kv[1])}, flush=True)
    print(f"[{nm}] ERR:", {a: round(b, 3) for a, b in sorted(perg.items(), key=lambda kv: kv[1])}, flush=True)
print("PERFAM_DONE", flush=True)
