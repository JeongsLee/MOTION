"""Functional identity of learned (V1a generic) axial stencils on the ACTUAL latent field.

For each family, the bank input h is captured from real forward passes (eager, first step). Each learned
stencil (width 3, per channel and axis role) is decomposed EXACTLY into the orthonormal basis
{3-point average (low-pass), first difference (gradient), second difference (Laplacian)}; the output
energy of each component is c_j^2 * E[(Q_j * h)^2] summed over channels (optionally weighted by how much
the head's Dense uses the channel). The share tells what the head computes on the data it actually sees:
a "diffusion" head whose output is dominated by the low-pass term acts as a pointwise/smoothing map,
not as a Laplacian.
  python eval_stencil_func.py --ckpt C --out OUT.json [--n 4] [--fams 2d]
"""
import argparse, itertools, json, os, zlib
import numpy as np
import tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--split", default="test")
ap.add_argument("--fams", default="2d")
a = ap.parse_args()

from core.pretrain import load_ckpt
from data import loader
from data.registry import FAMILIES, NUM_SLOTS, SEMANTIC_SLOTS, MOTION_FAMILIES
from v3.data_ar import make_example_ar, stack_batch
from v3.model import V3Model
from v3.train import BOX_TABLE_2D, _grid

model = V3Model(S=NUM_SLOTS, d=384, depth=8, d_w=16, n_p=16, d_cond=128, dims2=(64, 64),
                dims3=(32, 32, 32), t_in=10, transport=True, n_fams=len(FAMILIES),
                n_semantic=SEMANTIC_SLOTS, gate_cond=1, dict_decode=0, fam_head=1)
model.warm_build()
V = model.trainable_variables
load_ckpt(V, a.ckpt, d_w=16, names=[v.name for v in V])
B_ = model.banks
assert getattr(B_, "generic", False), "run with MOTION_BANK_GENERIC=1 on a V1a checkpoint"

CAP = []
_orig = B_.call
def _wrap(h, *args, **kw):
    CAP.append((h.numpy(), kw.get("roles", args[1] if len(args) > 1 else None)))
    return _orig(h, *args, **kw)
B_.call = _wrap

x = np.arange(3) - 1
Q, _ = np.linalg.qr(np.vander(x, 3, increasing=True).astype(float))   # cols: avg, grad, lap
for j in range(3):
    if Q[-1, j] < 0: Q[:, j] *= -1
LBL = ["lowpass", "grad", "lap"]
heads = [m for m in ("diffusion", "wave", "shear", "buoyancy", "gradvec", "advection", "shock",
                     "dilatation", "boundary_layer") if m in B_.gS]


def dense_importance(m):
    layer = B_.heads.get(m) if hasattr(B_, "heads") else None
    if layer is None or not layer.weights:
        return None
    K = layer.weights[0].numpy()
    d = B_.gS[m].k.shape[-1]
    return np.linalg.norm(K[:d], axis=1) if K.shape[0] >= d else None


fams = [f for f in MOTION_FAMILIES if FAMILIES[f].K == 2] if a.fams == "2d" else a.fams.split(",")
res = {}
for fam in fams:
    units = list(itertools.islice(loader.units_for(fam, a.split), a.n))
    exs = [make_example_ar(s, 10, 10, 4, 2048, 8192, np.random.default_rng(zlib.crc32(f"{fam}:{i}:0".encode())),
                           box_dims=(int(BOX_TABLE_2D.get(fam, 64)),) * 2) for i, s in enumerate(units)]
    CAP.clear()
    for i in range(0, len(exs), 2):
        b = stack_batch(exs[i:i + 2])
        T = lambda k: tf.constant(b[k])
        cn, xw, fm, gn = T("coords_node"), T("x_win"), T("fmask"), T("geom_node")
        qall = tf.concat([cn, T("coords_q")], 1); gall = tf.concat([gn, T("geom_q")], 1)
        u_prev = tf.concat([xw[:, -1], T("ic_q")], 1)
        K = int(b["K"]); roles = [int(r) for r in b["roles"]]; gd = tuple(b.get("dims", ())) or None
        box = tuple(int(v) for v in b["box_dims"])
        if b["mode"] == "ss":
            model.step_ss(cn, xw, fm, gn, K, roles, T("cond"), T("op_multihot"), qall, gall, u_prev,
                          kf=int(b["k_seg"]), fam_id=T("fam_id"), sdf_box=T("sdf_box"), box_dims=box)
        else:
            dense = bool(b["dense2d"])
            pg = _grid(xw[:, -1], gd, model.S) if dense else None
            model.step(cn, xw, fm, gn, K, roles, T("cond"), T("op_multihot"), qall, gall, u_prev,
                       steady=bool(b["steady"]), prev_grid=pg, grid_dims=gd, fam_id=T("fam_id"),
                       sdf_box=T("sdf_box"), box_dims=box, fbmask=T("fbmask"), surf=bool(b["surf"]),
                       prev2_grid=_grid(xw[:, -2], gd, model.S) if dense else None)
    if not CAP:
        continue
    res[fam] = {}
    for m in heads:
        W = B_.gS[m].k.numpy()                                  # (roles, 3, d)
        imp = dense_importance(m)
        E = np.zeros(3)
        for h, roles in CAP:
            h = h.astype(np.float64); K = h.ndim - 2; roles = roles or list(range(K))
            for ax in range(K):
                C = Q.T @ W[int(roles[ax]) % W.shape[0]]          # (3 basis, d) coefficients
                for j in range(3):
                    y = sum(Q[t, j] * np.roll(h, 1 - t, axis=1 + ax) for t in range(3))   # basis_j * h
                    e = (C[j][None, None, None, :] * y) ** 2 if K == 2 else (C[j] * y) ** 2
                    e = e.reshape(-1, e.shape[-1]).mean(0)       # per-channel output energy
                    E[j] += float((e * (imp / imp.sum() if imp is not None else 1.0 / len(e))).sum())
        res[fam][m] = (E / E.sum()).tolist()
    print(f"[{fam:16s}] " + " | ".join(f"{m}: " + "/".join(f"{v:.2f}" for v in res[fam][m])
                                     for m in ("diffusion", "advection", "wave", "shock") if m in res[fam]),
          flush=True)
agg = {m: np.mean([res[f][m] for f in res if m in res[f]], 0).round(3).tolist() for m in heads}
print("MEAN over families (lowpass/grad/lap):", json.dumps(agg), flush=True)
json.dump({"ckpt": a.ckpt, "basis": LBL, "per_family": res, "mean": agg}, open(a.out, "w"), indent=1)
print("STENCIL_FUNC_OK", a.out, flush=True)
