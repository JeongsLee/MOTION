"""[eval_redund] Head REDUNDANCY vs COLLAPSE (built from eval_contrib). Per-family mechanism-head contribution matrix (revision R1: V2 specialisation vs capacity).

For each family, the first forecast step of n test trajectories is run (graph mode, XLA) with the bank probe
on (TendencyBanks.probe), which records mean |sigma_k (.) head_k(h)| for every head (and H0) — the
actual contribution to W, not the gate value alone. Averaged over batches -> C[family, head].

Derived (printed and saved):
  share[f, k]     = C[f, k] / sum_k C[f, k]              (H0 excluded from the share)
  entropy / Gini  per family over the shares              (concentrated vs uniform use)
  mask AUC        how well contribution ranks the metadata labels (op token present / absent)
                  over the maskable heads (OP_INDEX), pooled over families. For B-R the masked
                  heads are exactly 0, so AUC = 1 by construction; the informative number is V2's.
  python eval_contrib.py --ckpt C --out OUT.json [--split test] [--n 8] [--batch 4]
"""
import argparse, json, os, time, zlib
import itertools
import numpy as np
import tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--fams", default="2d")
a = ap.parse_args()

from core.pretrain import load_ckpt
from core.banks import OP_INDEX
from data import loader
from data.registry import FAMILIES, NUM_SLOTS, SEMANTIC_SLOTS, MOTION_FAMILIES
from v3.data_ar import make_example_ar, stack_batch
from v3.model import V3Model
from v3.train import BOX_TABLE_2D, BOX_TABLE_3D, _grid

model = V3Model(S=NUM_SLOTS, d=384, depth=8, d_w=16, n_p=16, d_cond=128,
                dims2=(64, 64), dims3=(32, 32, 32), t_in=10, transport=True,
                n_fams=len(FAMILIES), n_semantic=SEMANTIC_SLOTS,
                gate_cond=1, dict_decode=0, fam_head=1)
model.warm_build()
V = model.trainable_variables
load_ckpt(V, a.ckpt, d_w=16, names=[v.name for v in V])
env = {k: v for k, v in os.environ.items() if k.startswith("MOTION_")}
print(f"[contrib] {a.ckpt} | env {env}", flush=True)


KEYS = ("coords_node", "x_win", "fmask", "geom_node", "cond", "op_multihot", "coords_q", "geom_q",
        "ic_q", "fam_id", "sdf_box", "fbmask")


def _probe_step(ts, K, roles, gd, box, mode, steady, dense, surf, k_seg):
    """First forecast step in GRAPH mode; the bank probe returns per-head mean |contribution| tensors."""
    cn, xw, fm, gn, cond, op, cq, gq, icq, fid, sdb, fbm = ts
    qall = tf.concat([cn, cq], 1); gall = tf.concat([gn, gq], 1)
    u_prev = tf.concat([xw[:, -1], icq], 1)
    model.banks.probe = {}
    model.banks.probe_all_fields = True
    if mode == "ss":
        model.step_ss(cn, xw, fm, gn, K, list(roles), cond, op, qall, gall, u_prev,
                      kf=k_seg, fam_id=fid, sdf_box=sdb, box_dims=box)
    else:
        pg = _grid(xw[:, -1], gd, model.S) if dense else None
        pg2 = _grid(xw[:, -2], gd, model.S) if dense else None
        model.step(cn, xw, fm, gn, K, list(roles), cond, op, qall, gall, u_prev,
                   steady=steady, prev_grid=pg, grid_dims=gd, fam_id=fid, sdf_box=sdb, box_dims=box,
                   fbmask=fbm, surf=surf, prev2_grid=pg2)
    out = dict(model.banks.probe)
    model.banks.probe = None
    return out


f_probe = tf.function(_probe_step, reduce_retracing=True, jit_compile=True)


def bdims(fam):
    spec = FAMILIES[fam]
    r = (BOX_TABLE_2D if spec.K == 2 else BOX_TABLE_3D).get(fam, 64 if spec.K == 2 else 32)
    return (int(r),) * spec.K



# ---------------- redundancy analysis
fams = ([f for f in MOTION_FAMILIES if FAMILIES[f].K == 2] if a.fams == "2d" else a.fams.split(","))
R = {}
for fam in fams:
    spec = FAMILIES[fam]
    units = list(itertools.islice(loader.units_for(fam, a.split), a.n))
    exs = [make_example_ar(s, 10, 10, 4, 2048, 8192,
                           np.random.default_rng(zlib.crc32(f"{fam}:{i}:0".encode())), box_dims=bdims(fam))
           for i, s in enumerate(units)]
    sims, norms, prs, names = [], [], [], None
    for i in range(0, len(exs), 2):
        b = stack_batch(exs[i:i + 2])
        ts = tuple(tf.constant(b[k]) for k in KEYS)
        st = (int(b["K"]), tuple(int(r) for r in b["roles"]), tuple(b.get("dims", ())) or None,
              tuple(int(v) for v in b["box_dims"]), str(b["mode"]), bool(b["steady"]),
              bool(b["dense2d"]), bool(b["surf"]), int(b["k_seg"]))
        pr = f_probe(ts, *st)
        F = {k[:-6]: v.numpy() for k, v in pr.items() if k.endswith("_field") and k[:-6] != "H0"}
        names = sorted(F)
        X = np.stack([F[n].reshape(F[n].shape[0], -1) for n in names], 1)      # (B, H, D)
        nrm = np.linalg.norm(X, axis=-1)                                        # (B, H)
        Xn = X / (nrm[..., None] + 1e-12)
        sims.append(np.einsum("bhd,bgd->bhg", Xn, Xn))                          # cosine
        norms.append(nrm)
        for bb in range(X.shape[0]):                                            # participation ratio
            act = nrm[bb] > 1e-6 * nrm[bb].max()
            G = X[bb][act] @ X[bb][act].T
            ev = np.clip(np.linalg.eigvalsh(G), 0, None)
            prs.append(float(ev.sum() ** 2 / ((ev ** 2).sum() + 1e-30)) if ev.sum() > 0 else 0.0)
    S_ = np.concatenate(sims).mean(0); N_ = np.concatenate(norms).mean(0)
    share = N_ / (N_.sum() + 1e-30)
    R[fam] = {"heads": names, "cos": S_.tolist(), "share": share.tolist(),
              "participation_ratio": float(np.mean(prs)), "n_active": int((share > 0.01).sum())}
    act = [j for j, s in enumerate(share) if s > 0.02]
    pairs = sorted(((abs(S_[p, q]), names[p], names[q]) for ii, p in enumerate(act) for q in act[ii + 1:]), reverse=True)[:4]
    print(f"[{fam:16s}] PR {np.mean(prs):.2f} | heads>1% {(share>0.01).sum():2d} | top |cos|: "
          + " ".join(f"{p}/{q}={c:.2f}" for c, p, q in pairs), flush=True)
json.dump({"ckpt": a.ckpt, "env": env, "R": R}, open(a.out, "w"), indent=1)
print("REDUND_OK", a.out, flush=True)
