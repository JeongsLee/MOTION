"""Fig3 symmetry-TTA ensemble — MOTION side. argv: task cfgmod ckpt do_ar f32(0/1) wv0(0/1) out
Group {id,fx,fy,r180}; slot0=vx pairs with row axis (verified layout), slot1=vy col axis.
Prints per-g and ensemble medians; dumps ens pred16 (+predAR16) for contours."""
import os, sys, importlib, numpy as np
sys.path.insert(0, "/tmp/w"); sys.modules.pop("code", None)
import tensorflow as tf
task, cfgmod, ck, do_ar, use_f32, wv0, out = (sys.argv[1], sys.argv[2], sys.argv[3],
    int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7])
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
    u0[..., int(cfg.v0_to_slot)] = 0.0                       # repaired anchor-0: v(0)=0
print("test", xt.shape, "NT", NT, flush=True)

# g = (flip_row, flip_col); axes of (B,H,W,C) or (B,T,H,W,C); slot0 <-> row, slot1 <-> col
GROUP = [(0, 0), (1, 0), (0, 1), (1, 1)]
def T(a, g, inv=False):                                      # involution: inv == fwd
    fr, fc = g
    ax_r = a.ndim - 3; ax_c = a.ndim - 2
    if fr: a = np.flip(a, axis=ax_r).copy(); a[..., 0] = -a[..., 0]
    if fc: a = np.flip(a, axis=ax_c).copy(); a[..., 1] = -a[..., 1]
    return a

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
    ft = r.mean(axis=1); f12 = r[:, :12].mean(axis=1)
    print(f"[{task} {tag}] full-traj median={np.median(ft)*100:.3f}%  lead12={np.median(f12)*100:.3f}%", flush=True)
    return ft

preds_g = []
for g in GROUP:
    p = predict(T(u0, g))
    p = T(p, g)                                               # inverse (involution)
    preds_g.append(p)
    metrics(p, f"g={g}")
ens = np.mean(preds_g, axis=0)
ft1 = metrics(preds_g[0], "single")
ftE = metrics(ens, "ENSEMBLE")
save = dict(pred16=ens[:16].astype(np.float32), gt16=xt[:16], mean=mean, std=std, cm=cm,
            ft_single=ft1.astype(np.float32), ft_ens=ftE.astype(np.float32))
if do_ar:
    ARS = 2; tg = list(range(ARS, NT, ARS)); cmk = cm[:, None, None, :]
    ar_g = []
    for g in GROUP:
        cur = T(np.array(u0), g); u00 = np.array(cur); pl = []
        for t in tg:
            nx = predict(cur)[:, ARS]
            cur = nx*cmk + u00*(1.0-cmk); pl.append(cur)
        ar_g.append(T(np.stack(pl, 1), g))
    arE = np.mean(ar_g, axis=0)
    sc = std[:, None, None, None, :]; mc = mean[:, None, None, None, :]; m5 = cm[:, None, None, None, :]
    for arr, tag in [(ar_g[0], "AR single"), (arE, "AR ENSEMBLE")]:
        pA = arr*sc + mc; rA = xt[:, tg]*sc + mc
        num = np.sum(np.abs(pA-rA)*m5, axis=(2, 3, 4)); den = np.sum(np.abs(rA)*m5, axis=(2, 3, 4))+1e-8
        r = (num/den).mean(axis=1)
        print(f"[{task} {tag}] full-traj median={np.median(r)*100:.3f}%", flush=True)
        if tag == "AR ENSEMBLE": save["ftAR_ens"] = r.astype(np.float32)
        else: save["ftAR_single"] = r.astype(np.float32)
    save["predAR16"] = arE[:16].astype(np.float32); save["ar_targets"] = np.array(tg)
np.savez_compressed(out, **save)
print("DUMP_DONE", out, flush=True)
