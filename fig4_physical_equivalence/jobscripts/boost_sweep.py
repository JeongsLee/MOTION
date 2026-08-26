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

def _flips(a, fr, fc):
    if fr: a = np.flip(a, axis=a.ndim - 3).copy(); a[..., 0] = -a[..., 0]
    if fc: a = np.flip(a, axis=a.ndim - 2).copy(); a[..., 1] = -a[..., 1]
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

def metrics(pred, tag):
    sc = std[:, None, None, None, :]; mc = mean[:, None, None, None, :]
    m5 = cm[:, None, None, None, :]
    pT = pred[:, 1:]*sc + mc; rT = xt[:, 1:NT]*sc + mc
    num = np.sum(np.abs(pT-rT)*m5, axis=(2, 3, 4)); den = np.sum(np.abs(rT)*m5, axis=(2, 3, 4))+1e-8
    r = (num/den)
    ft = r.mean(axis=1)
    print(f"[{task} {tag}] full-traj median={np.median(ft)*100:.3f}%", flush=True)
    return ft

preds_g = {}
for g in GROUP:
    p = T(predict(T(u0, g)), g, inv=True)
    preds_g[g] = p
    metrics(p, f"g={g}")

# ================= Galilean boost (periodic, exact on the lattice) =================
# u'(x,t) = u(x + V t, t) + V, with V an integer number of cells per frame along AX.
# ---- Galilean boost sweep, exact lattice displacements only ------------------
# u'(x,t) = u(x + Vt, t) + V.  Input is the single frame t=0, so the boosted input
# is u0 with +V on the in-plane velocity slot; the prediction of frame j must be the
# original prediction rolled by -V*j and shifted by +V.  V = (p/q) cells/frame; a
# frame j is scored only when p*j/q is an integer, so every roll is exact.
AX = int(os.environ.get("BOOST_AX", "1"))
SLOT = 0 if AX == 0 else 1
_scq = std[:, None, None, None, :]; _mcq = mean[:, None, None, None, :]
_su = std[:, SLOT].reshape(-1, 1, 1, 1); _mu = mean[:, SLOT].reshape(-1, 1, 1, 1)
_uax = xt[..., SLOT] * _su + _mu
_urms = float(np.sqrt((_uax ** 2).mean()))
_dx = 1.0 / xt.shape[2]; _dt = 1.0 / (NT - 1)
P0q = predsg_id = preds_g[(0, 0, 0)]
_SOLq = {"NS-PwC": [0, 1, 2], "ACE": [2], "Wave-Layer": [2], "Poisson-Gauss": [2]}
_msq = np.zeros_like(cm)
for _c in _SOLq.get(task, list(range(cm.shape[-1]))):
    _msq[:, _c] = cm[:, _c]
_m5q = _msq[:, None, None, None, :]

def _rel_frames(P, R, fr):
    """rel-L1 in the task convention, restricted to output-frame indices fr."""
    Pp = P[:, fr] * _scq + _mcq; Rp = R[:, fr] * _scq + _mcq
    num = np.sum(np.abs(Pp - Rp) * _m5q, axis=(2, 3, 4))
    den = np.sum(np.abs(Rp) * _m5q, axis=(2, 3, 4)) + 1e-8
    return float(np.median((num / den).mean(axis=1)) * 100)

print(f"[{task}] u_rms={_urms:.4f}  (dx={_dx:.5f}, dt={_dt:.4f})", flush=True)
print(f"[{task}] {'V/u_rms':>8} {'cells/fr':>9} {'frames':>7} {'err_id':>8} "
      f"{'err_boost':>10} {'eps':>8} {'eps_noroll':>11}", flush=True)
for (p, q) in [(1, 4), (1, 2), (1, 1), (2, 1), (3, 1)]:
    V = (p / q) * _dx / _dt
    Vn = V / (std[:, SLOT] + 1e-9)
    u0b = np.array(u0); u0b[..., SLOT] += Vn[:, None, None]
    Pb = predict(u0b)
    # exact frames: output index j corresponds to global frame j
    fr = [j for j in range(1, Pb.shape[1]) if (p * j) % q == 0]
    if not fr:
        continue
    inv = np.array(Pb); inv[..., SLOT] -= Vn[:, None, None, None]
    noroll = np.array(inv)
    for j in fr:
        inv[:, j] = np.roll(inv[:, j], (p * j) // q, axis=AX + 1)
    e_id = _rel_frames(P0q, xt[:, :NT], fr)
    e_bo = _rel_frames(inv, xt[:, :NT], fr)
    eps = _rel_frames(inv, P0q, fr)
    eps0 = _rel_frames(noroll, P0q, fr)
    print(f"[{task}] {V/_urms:8.2f} {p/q:9.2f} {len(fr):7d} {e_id:8.3f} "
          f"{e_bo:10.3f} {eps:8.3f} {eps0:11.3f}", flush=True)
    # did the operator carry the drift at all?
    mu_b = float((Pb[:, fr][..., SLOT] * _su + _mu).mean())
    mu_0 = float((P0q[:, fr][..., SLOT] * _su + _mu).mean())
    mu_g = float((xt[:, fr][..., SLOT] * _su + _mu).mean())
    print(f"[{task}]   mean u_ax: GT={mu_g:+.4f} pred={mu_0:+.4f} "
          f"pred_boosted={mu_b:+.4f} expected={mu_g + V:+.4f} "
          f"carried={(mu_b - mu_0) / V * 100:.0f}%", flush=True)
print("BOOST_SWEEP_DONE", flush=True)
