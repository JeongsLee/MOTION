"""Latent semantics probe (revision R1-3): what does the trained MOTION latent encode?

For a few test trajectories of selected 2-D families, run ONE forecast step of the trained model and save
  * the latent box h fed to the mechanism bank (B, H, W, d),
  * the learned cell velocity u = banks.vel(h) used by the advection-type heads (B, H, W, 3),
  * the input frame and the target frame on the native grid, and the prediction,
  * the same latent from a randomly initialised model (baseline for linear probes).
Intervention ("latent arithmetic"): a ridge probe h -> v_x is fitted on the dumped cells; the latent is then shifted
along the unit probe direction by alpha * std(h.w) and the change of the predicted v_x is recorded for several alpha.
  python latent_dump.py --ckpt C --out OUT.npz [--fams incom_ns,NS-Sines,shallow_water,CE-RP,com_ns] [--n 8]
"""
import argparse, itertools, json, os, zlib
import numpy as np
import tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--fams", default="incom_ns,NS-Sines,shallow_water,CE-RP,com_ns")
ap.add_argument("--n", type=int, default=8)
a = ap.parse_args()

from core.pretrain import load_ckpt
from data import loader
from data.registry import FAMILIES, NUM_SLOTS, SEMANTIC_SLOTS
from v3.data_ar import make_example_ar, stack_batch
from v3.model import V3Model
from v3.train import BOX_TABLE_2D, _grid

ARCH = tuple(int(os.environ.get(k, v)) for k, v in (("ARCH_D", 384), ("ARCH_DEPTH", 8), ("ARCH_DW", 16)))  # MOTION: 896, 12, 48


def build():
    m = V3Model(S=NUM_SLOTS, d=ARCH[0], depth=ARCH[1], d_w=ARCH[2], n_p=16, d_cond=128, dims2=(64, 64), dims3=(32, 32, 32), t_in=10,
                transport=True, n_fams=len(FAMILIES), n_semantic=SEMANTIC_SLOTS, gate_cond=1, dict_decode=0, fam_head=1)
    m.warm_build(); return m


model = build()
V = model.trainable_variables
load_ckpt(V, a.ckpt, d_w=ARCH[2], names=[v.name for v in V])
tf.random.set_seed(1234); rand = build()                 # untrained baseline (same architecture)

CAP = {}
def hook(m, key):
    orig = m.banks.call
    def wrapped(h, *args, **kw):
        CAP[key] = h
        return orig(h, *args, **kw)
    m.banks.call = wrapped
hook(model, "h"); hook(rand, "h_rand")
DELTA = {"d": None}
orig_core = model.core
def core_shift(h, e, roles):
    out = orig_core(h, e, roles)
    return out + DELTA["d"] if DELTA["d"] is not None else out
model.core = core_shift


def run(m, b):
    T = lambda k: tf.constant(b[k])
    cn, xw, fm, gn = T("coords_node"), T("x_win"), T("fmask"), T("geom_node")
    qall = tf.concat([cn, T("coords_q")], 1); gall = tf.concat([gn, T("geom_q")], 1)
    u_prev = tf.concat([xw[:, -1], T("ic_q")], 1)
    K = int(b["K"]); roles = [int(r) for r in b["roles"]]; gd = tuple(b.get("dims", ())) or None
    box = tuple(int(v) for v in b["box_dims"])
    pg = _grid(xw[:, -1], gd, m.S)
    out = m.step(cn, xw, fm, gn, K, roles, T("cond"), T("op_multihot"), qall, gall, u_prev, steady=False,
                 prev_grid=pg, grid_dims=gd, fam_id=T("fam_id"), sdf_box=T("sdf_box"), box_dims=box,
                 fbmask=T("fbmask"), surf=False, prev2_grid=_grid(xw[:, -2], gd, m.S))
    pred = out[0] if isinstance(out, (tuple, list)) else out
    return pred.numpy()[:, :cn.shape[1]]                   # prediction at the node (native grid) points


res = {}
for fam in a.fams.split(","):
    units = list(itertools.islice(loader.units_for(fam, "test"), a.n))
    bd = (int(BOX_TABLE_2D.get(fam, 64)),) * 2
    exs = [make_example_ar(s, 10, 1, 4, 2048, 8192, np.random.default_rng(zlib.crc32(f"{fam}:{i}:0".encode())),
                           box_dims=bd) for i, s in enumerate(units)]
    b = stack_batch(exs)
    if not bool(b["dense2d"]):
        print("skip (not dense 2-D)", fam); continue
    pred = run(model, b); h = CAP["h"].numpy(); vel = model.banks.vel(CAP["h"]).numpy()
    run(rand, b); hr = CAP["h_rand"].numpy()
    dims = tuple(int(x) for x in b["dims"])
    x_in = np.asarray(b["x_win"])[:, -1].reshape(len(exs), *dims, -1)
    y_next = np.asarray(b["y_node"])[:, 0].reshape(len(exs), *dims, -1) if "y_node" in b else None
    # --- intervention: ridge probe h -> v_x (slot 0) on the box grid, then shift h along the probe direction
    Hb, Wb, d = h.shape[1:]
    fy, fx = dims[0] // Hb, dims[1] // Wb
    vx_box = x_in[..., 0].reshape(len(exs), Hb, fy, Wb, fx).mean((2, 4))
    X = h.reshape(-1, d).astype(np.float64); yv = vx_box.reshape(-1)
    Xm = X.mean(0); w = np.linalg.solve((X - Xm).T @ (X - Xm) + 1e-2 * len(X) * np.eye(d), (X - Xm).T @ (yv - yv.mean()))
    wdir = (w / np.linalg.norm(w)).astype(np.float32); s = float((X @ wdir).std())
    alphas = [-2.0, -1.0, 0.0, 1.0, 2.0]; dvx = []
    for al in alphas:
        DELTA["d"] = tf.constant(al * s * wdir[None, None, None, :], dtype=CAP["h"].dtype)
        p2 = run(model, b); dvx.append(float((p2[..., 0] - pred[..., 0]).mean()))
    DELTA["d"] = None
    res[fam] = dict(h=h.astype(np.float16), h_rand=hr.astype(np.float16), vel=vel.astype(np.float32), x_in=x_in.astype(np.float32),
                    y_next=None if y_next is None else y_next.astype(np.float32), pred=pred.reshape(len(exs), *dims, -1).astype(np.float32),
                    scale=np.asarray(b["scale"]), alphas=np.array(alphas), dvx=np.array(dvx), probe_std=s)
    print(f"[{fam}] h {h.shape} vel {vel.shape} native {dims}; intervention mean Δvx vs alpha: "
          + ", ".join(f"{al:+.0f}:{v:+.4f}" for al, v in zip(alphas, dvx)), flush=True)
flat = {}
for fam, r in res.items():
    for k, v in r.items():
        if v is not None: flat[f"{fam}|{k}"] = v
np.savez_compressed(a.out, **flat, __meta__=json.dumps({"ckpt": a.ckpt, "fams": list(res)}))
print("LATENT_OK", a.out, flush=True)
