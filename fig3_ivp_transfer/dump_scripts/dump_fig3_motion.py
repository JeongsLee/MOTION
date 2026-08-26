"""Fig3 per-family qual dump — MOTION side (chunked for CPU-job RAM).
argv: task cfgmod ckpt do_ar(0/1) out.npz"""
import os, sys, importlib, numpy as np
sys.path.insert(0, "/tmp/w"); sys.modules.pop("code", None)
import tensorflow as tf
task, cfgmod, ck, do_ar, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
m = importlib.import_module(f"motion_tf.train.configs.{cfgmod}"); cfg = m.Cfg
if bool(getattr(cfg, "bf16", False)):
    tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
    print("mixed_bfloat16 policy enabled", flush=True)
from motion_tf.model import PhysicsOperatorMixture
from motion_tf.data.poseidon import build_poseidon_stream
from motion_tf.utils import ckpt as ckptlib
model = PhysicsOperatorMixture(cfg)
_ = model.call_with_gate(tf.zeros((1, cfg.Nx, cfg.Nx, 8)), tf.zeros((1, cfg.desc_dim)), tf.zeros((1, 4)))
ckptlib.load(model, ck); print("restored", ck, flush=True)
_, test = build_poseidon_stream(cfg, cfg.batch)
xt = np.asarray(test["traj"], np.float32); cm = np.asarray(test["cm"], np.float32)
mean = np.asarray(test["mean"], np.float32); std = np.asarray(test["std"], np.float32)
N, NT = xt.shape[0], xt.shape[1]; print("test", xt.shape, "cm0", cm[0], flush=True)

def predict(u0):
    o = []
    for i in range(0, len(u0), 8):
        b = tf.constant(u0[i:i+8])
        o.append(tf.cast(model.call_with_gate(b, tf.zeros((len(b), cfg.desc_dim)),
                                              tf.zeros((len(b), 4)),
                                              training=False)[0], tf.float32).numpy())
    return np.concatenate(o)[:, :NT]

CH = 16; ARS = 2; tg = list(range(ARS, NT, ARS))
acc = {k: [] for k in ["ft_sol", "ft_uv", "fin_sol", "fin_uv", "ftAR_sol", "ftAR_uv"]}
pred16 = predAR16 = None
for i in range(0, N, CH):
    sl = slice(i, i+CH)
    p = predict(xt[sl, 0])
    sc = std[sl, None, None, None, :]; mc = mean[sl, None, None, None, :]
    m5 = cm[sl, None, None, None, :]
    pT = p[:, 1:]*sc + mc; rT = xt[sl, 1:]*sc + mc
    num = np.sum(np.abs(pT-rT)*m5, axis=(2, 3, 4)); den = np.sum(np.abs(rT)*m5, axis=(2, 3, 4))+1e-8
    nu = np.sum(np.abs(pT[..., :2]-rT[..., :2]), axis=(2, 3, 4))
    du = np.sum(np.abs(rT[..., :2]), axis=(2, 3, 4))+1e-8
    acc["ft_sol"] += list((num/den).mean(axis=1)); acc["ft_uv"] += list((nu/du).mean(axis=1))
    acc["fin_sol"] += list((num/den)[:, -1]);      acc["fin_uv"] += list((nu/du)[:, -1])
    if i == 0: pred16 = p[:16].copy()
    del p, pT, rT
    if do_ar:
        cur = np.array(xt[sl, 0]); u00 = np.array(xt[sl, 0]); pl = []
        cmk = cm[sl, None, None, :]
        for t in tg:
            nx = predict(cur)[:, ARS]
            cur = nx*cmk + u00*(1.0-cmk); pl.append(cur)
        pAR = np.stack(pl, 1)
        pA = pAR*sc + mc; rA = xt[sl][:, tg]*sc + mc
        numA = np.sum(np.abs(pA-rA)*m5, axis=(2, 3, 4)); denA = np.sum(np.abs(rA)*m5, axis=(2, 3, 4))+1e-8
        nuA = np.sum(np.abs(pA[..., :2]-rA[..., :2]), axis=(2, 3, 4))
        duA = np.sum(np.abs(rA[..., :2]), axis=(2, 3, 4))+1e-8
        acc["ftAR_sol"] += list((numA/denA).mean(axis=1)); acc["ftAR_uv"] += list((nuA/duA).mean(axis=1))
        if i == 0: predAR16 = pAR[:16].copy()
        del pAR, pA, rA
    print(f"  chunk {i}..{i+CH} done", flush=True)
A = {k: np.array(v, np.float32) for k, v in acc.items() if v}
print(f"[VERIFY {task}] direct full-traj SOL median={np.median(A['ft_sol'])*100:.3f}%  "
      f"uv={np.median(A['ft_uv'])*100:.3f}%  @final SOL={np.median(A['fin_sol'])*100:.3f}%", flush=True)
if do_ar:
    print(f"[VERIFY {task}] AR hop2 SOL median={np.median(A['ftAR_sol'])*100:.3f}%  "
          f"uv={np.median(A['ftAR_uv'])*100:.3f}%", flush=True)
save = dict(pred16=pred16, gt16=xt[:16], mean=mean, std=std, cm=cm, **A)
if predAR16 is not None:
    save.update(predAR16=predAR16, ar_targets=np.array(tg))
np.savez_compressed(out, **save)
print("DUMP_DONE", out, flush=True)
