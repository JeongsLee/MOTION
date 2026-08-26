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
# Input is the single frame t=0, so the boosted input is just u0 with +V on the
# in-plane velocity slot; the prediction of frame j must then be the original
# prediction rolled by V*j and shifted by +V.
AX = int(os.environ.get("BOOST_AX", "1"))          # 1 = column axis (slot1 component)
VS = int(os.environ.get("BOOST_CELLS", "3"))       # cells per frame
SLOT = 0 if AX == 0 else 1                         # velocity component along AX
_scq = std[:, None, None, None, :]; _mcq = mean[:, None, None, None, :]
_su = std[:, SLOT].reshape(-1, 1, 1, 1); _mu = mean[:, SLOT].reshape(-1, 1, 1, 1)
_uax = xt[..., SLOT] * _su + _mu                       # (B,T,H,W) physical velocity
_urms = float(np.sqrt((_uax ** 2).mean()))
_dx = 1.0 / xt.shape[2]; _dt = 1.0 / (NT - 1)
_V = VS * _dx / _dt
print(f"[{task}] boost {VS} cell/frame -> V={_V:.4f}, u_rms={_urms:.4f}, V/u_rms={_V/_urms:.2f}", flush=True)

_Vn = _V / (std[:, SLOT] + 1e-9)                        # V in normalised units, per sample
u0b = np.array(u0)
u0b[..., SLOT] = u0b[..., SLOT] + _Vn[:, None, None]
Pb = predict(u0b)                                   # G(T u)
# inverse: subtract V, un-roll frame j by -V*j (frame index j = 1..NT-1 of the output)
Pb_inv = np.array(Pb)
Pb_inv[..., SLOT] = Pb_inv[..., SLOT] - _Vn[:, None, None, None]
for j in range(Pb_inv.shape[1]):
    Pb_inv[:, j] = np.roll(Pb_inv[:, j], -VS * j, axis=AX + 1)
P0q = preds_g[(0, 0, 0)]
_SOLq = {"NS-PwC": [0, 1, 2], "ACE": [2], "Wave-Layer": [2], "Poisson-Gauss": [2]}
_msq = np.zeros_like(cm)
for _c in _SOLq.get(task, list(range(cm.shape[-1]))):
    _msq[:, _c] = cm[:, _c]
_m5q = _msq[:, None, None, None, :]
def _relq(P, R):
    num = np.sum(np.abs(P - R) * _m5q, axis=(2, 3, 4))
    den = np.sum(np.abs(R) * _m5q, axis=(2, 3, 4)) + 1e-8
    return float(np.median((num / den).mean(axis=1)) * 100)
A = Pb_inv[:, 1:] * _scq + _mcq; Bq = P0q[:, 1:] * _scq + _mcq
GTq = xt[:, 1:NT] * _scq + _mcq
print(f"[{task}] BOOST eps={_relq(A, Bq):.3f}%  err_id={_relq(Bq, GTq):.3f}%  "
      f"err_boost={_relq(A, GTq):.3f}%", flush=True)
# ---- equivariance defect, in the SAME convention as metrics(): rel-L1 on physical
# units over the scored solution slots, frames 1:, frame-mean then sample-median ----
_SOL = {"NS-PwC": [0, 1, 2], "ACE": [2], "Wave-Layer": [2], "Poisson-Gauss": [2]}
_ms = np.zeros_like(cm)
for _c in _SOL.get(task, list(range(cm.shape[-1]))):
    _ms[:, _c] = cm[:, _c]
_m5 = _ms[:, None, None, None, :]
_sc = std[:, None, None, None, :]; _mc = mean[:, None, None, None, :]
def _eps_of(Pg, P0):
    A = Pg[:, 1:] * _sc + _mc; B = P0[:, 1:] * _sc + _mc
    num = np.sum(np.abs(A - B) * _m5, axis=(2, 3, 4))
    den = np.sum(np.abs(B) * _m5, axis=(2, 3, 4)) + 1e-8
    return float(np.median((num / den).mean(axis=1)) * 100)
_lb = {(0,1,0):"fr",(0,0,1):"fc",(0,1,1):"r180",(1,0,0):"d1",(1,1,0):"r90",(1,0,1):"r270",(1,1,1):"d2"}
_P0 = preds_g[(0, 0, 0)]
_eps = {_lb[g]: _eps_of(preds_g[g], _P0) for g in GROUP if g != (0, 0, 0)}
print(f"[{task}] EPS_EQUIV(%, task metric):",
      {k: round(v, 3) for k, v in sorted(_eps.items(), key=lambda kv: kv[1])}, flush=True)
ens4 = np.mean([preds_g[(0, fr, fc)] for fr in (0, 1) for fc in (0, 1)], axis=0)
ens8 = np.mean(list(preds_g.values()), axis=0)
metrics(preds_g[(0, 0, 0)], "single")
metrics(ens4, "K4-ENSEMBLE")
metrics(ens8, "K8-ENSEMBLE")

if do_ar:
    ARS = 2; tg = list(range(ARS, NT, ARS)); cmk = cm[:, None, None, :]
    ar_g = {}
    for g in GROUP:
        cur = T(np.array(u0), g); u00 = np.array(cur); pl = []
        for t in tg:
            nx = predict(cur)[:, ARS]
            cur = nx*cmk + u00*(1.0-cmk); pl.append(cur)
        ar_g[g] = T(np.stack(pl, 1), g, inv=True)
    sc = std[:, None, None, None, :]; mc = mean[:, None, None, None, :]; m5 = cm[:, None, None, None, :]
    def armet(arr, tag):
        pA = arr*sc + mc; rA = xt[:, tg]*sc + mc
        num = np.sum(np.abs(pA-rA)*m5, axis=(2, 3, 4)); den = np.sum(np.abs(rA)*m5, axis=(2, 3, 4))+1e-8
        r = (num/den).mean(axis=1)
        print(f"[{task} {tag}] full-traj median={np.median(r)*100:.3f}%", flush=True)
    for g in GROUP: armet(ar_g[g], f"AR g={g}")
    armet(ar_g[(0, 0, 0)], "AR single")
    armet(np.mean([ar_g[(0, fr, fc)] for fr in (0, 1) for fc in (0, 1)], axis=0), "AR K4-ENSEMBLE")
    armet(np.mean(list(ar_g.values()), axis=0), "AR K8-ENSEMBLE")
print("DUMP_DONE", flush=True)
