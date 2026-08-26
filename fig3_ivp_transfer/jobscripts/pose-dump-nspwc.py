import os, sys, numpy as np
sys.path.insert(0, "/tmp/w"); sys.modules.pop("code", None)
import importlib, tensorflow as tf
m = importlib.import_module("motion_tf.train.configs.poseidon_finetune_combo_158m")
cfg = m.Cfg
from motion_tf.model import PhysicsOperatorMixture
from motion_tf.data.poseidon import build_poseidon_stream
from motion_tf.utils import ckpt as ckptlib
model = PhysicsOperatorMixture(cfg)
_ = model.call_with_gate(tf.zeros((1, cfg.Nx, cfg.Nx, 8)),
                         tf.zeros((1, cfg.desc_dim)), tf.zeros((1, 4)))
import glob
cks = sorted(glob.glob("/eu/results/ftcombo_NS-PwC_N64_162m800000x4/ckpt_*.npz"))
print("ckpts:", [c.split("/")[-1] for c in cks], flush=True)
ckptlib.load(model, cks[-1]); print("restored", cks[-1], flush=True)
print("[flags] warp=", model.warp_head, flush=True)
_, test = build_poseidon_stream(cfg, cfg.batch)
xt = test["traj"]; n = 8
u0 = xt[:n, 0].astype(np.float32)
zd, zc = tf.zeros((n, cfg.desc_dim)), tf.zeros((n, 4))
Wf, Wl, mixf, icf = model.operator(tf.constant(u0), zd, zc)
pred = tf.cast(model.call_with_gate(tf.constant(u0), zd, zc)[0], tf.float32).numpy()
gt = xt[:128].astype(np.float32)                       # incl. calib rows
traj = model.evolve(Wf, Wl, mixf, icf)
D = model.n_latent
z = tf.reshape(traj, [n, cfg.Nt, cfg.Nx, cfg.Nx, D])
delta = tf.cast(model.decode_out(model.decode_h(z)), tf.float32).numpy()
dec_w = {}
for li, layer in enumerate([model.decode_h, model.decode_out]):
    for wi, w in enumerate(layer.weights):
        dec_w[f"L{li}_w{wi}"] = w.numpy()
gg = gt[:n, 1:, :, :, :3]
rn = np.linalg.norm(pred[:, 1:, :, :, :3]-gg)/np.linalg.norm(gg)
print("[native] 3ch rel-L2:", rn, flush=True)
np.savez_compressed("/seoul/pose_refine/nspwc_dump.npz",
    Wf=Wf.numpy().astype(np.float32), Wl=Wl.numpy().astype(np.float32),
    mix=mixf.numpy().astype(np.float32), ic=icf.numpy().astype(np.float32),
    pred=pred, gt=gt[:, :, :, :, :3], mean=test["mean"][:128], std=test["std"][:128],
    delta3=delta[..., :3].astype(np.float32), **dec_w)
print("DUMP_DONE", flush=True)
