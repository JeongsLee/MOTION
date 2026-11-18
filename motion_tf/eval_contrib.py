"""Per-family mechanism-head contribution matrix (revision R1: V2 specialisation vs capacity).

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
ap.add_argument("--fams", default="all")
a = ap.parse_args()

from core.pretrain import load_ckpt
from core.banks import OP_INDEX
from data import loader
from data.registry import FAMILIES, NUM_SLOTS, SEMANTIC_SLOTS, MOTION_FAMILIES
from v3.data_ar import make_example_ar, stack_batch
from v3.model import V3Model
from v3.train import BOX_TABLE_2D, BOX_TABLE_3D, _grid

# REVISION-R1 10-27: N_P env (MOTION-IVP: 32), KF env = example horizon (MOTION-IVP: 20); defaults unchanged
model = V3Model(S=NUM_SLOTS, d=384, depth=8, d_w=16, n_p=int(os.environ.get("N_P", "16")), d_cond=128,
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
    if mode == "ss":
        model.step_ss(cn, xw, fm, gn, K, list(roles), cond, op, qall, gall, u_prev,
                      kf=k_seg, fam_id=fid, sdf_box=sdb, box_dims=box)
    else:
        pg = _grid(xw[:, -1], gd, model.S) if dense else None
        pg2 = _grid(xw[:, -2], gd, model.S) if dense else None
        model.step(cn, xw, fm, gn, K, list(roles), cond, op, qall, gall, u_prev,
                   steady=steady, prev_grid=pg, grid_dims=gd, fam_id=fid, sdf_box=sdb, box_dims=box,
                   fbmask=fbm, surf=surf, prev2_grid=pg2)
    out = {k: v for k, v in model.banks.probe.items() if not k.endswith("_field")}
    model.banks.probe = None
    return out


_tko = [h for h in os.environ.get("MOTION_TRAIN_KO", "").split(",") if h]
if _tko:                                    # REVISION-R1: heads removed during training stay off (contribution 0)
    _nm = (list(model.banks.names) + list(getattr(model.banks, "_gen", []))   # + heads outside banks.names
           + ["bernoulli", "kinematic", "vortex", "reaction_nl", "spectral", "coefficient", "compressible"])
    model.banks.ko_index = {h: i for i, h in enumerate(_nm)}
    model.banks.ko_mask = tf.Variable([0.0 if h in _tko else 1.0 for h in _nm], trainable=False, name="train_ko_mask")
    print(f"[contrib] heads removed: {_tko}", flush=True)
f_probe = tf.function(_probe_step, reduce_retracing=True, jit_compile=True)


def bdims(fam):
    if int(os.environ.get("MOTION_BOX_UNIFORM", "0")):          # REVISION-R1: models trained with --box_table 0
        return (int(os.environ["MOTION_BOX_UNIFORM"]),) * FAMILIES[fam].K
    spec = FAMILIES[fam]
    r = (BOX_TABLE_2D if spec.K == 2 else BOX_TABLE_3D).get(fam, 64 if spec.K == 2 else 32)
    return (int(r),) * spec.K


fams = list(MOTION_FAMILIES) if a.fams == "all" else a.fams.split(",")
# REVISION-R1 10-27: MOTION_GLOBAL_NORM models (MOTION-IVP) need the per-family statistics before examples are built
# (same as score_ivp_l1.py: train split, first 64 trajectories)
import v3.data_ar as _DA
if _DA.GLOBAL_NORM_ON:
    for _f in fams:
        _DA.GLOBAL_NORM[_f] = _DA.compute_global_norm(loader.units_for(_f, "train"))[:2]
        print(f"[contrib] global norm {_f}", flush=True)
C, LAB = {}, {}
t0 = time.time()
for fam in fams:
    spec = FAMILIES[fam]
    B = 1 if spec.K == 3 else a.batch
    units = list(itertools.islice(loader.units_for(fam, a.split), a.n))
    exs = [make_example_ar(s, 10, int(os.environ.get("KF", "10")), 4, 2048, 8192,
                           np.random.default_rng(zlib.crc32(f"{fam}:{i}:0".encode())), box_dims=bdims(fam))
           for i, s in enumerate(units)]
    acc, nb = {}, 0
    for i in range(0, len(exs), B):
        b = stack_batch(exs[i:i + B])
        ts = tuple(tf.constant(b[k]) for k in KEYS)
        st = (int(b["K"]), tuple(int(r) for r in b["roles"]), tuple(b.get("dims", ())) or None,
              tuple(int(v) for v in b["box_dims"]), str(b["mode"]), bool(b["steady"]),
              bool(b["dense2d"]), bool(b["surf"]), int(b["k_seg"]))
        pr = f_probe(ts, *st)
        for k, v in pr.items():
            acc[k] = acc.get(k, 0.0) + float(v)
        nb += 1
        mh = np.asarray(b["op_multihot"])[0]
    model.banks.probe = None
    C[fam] = {k: v / nb for k, v in acc.items()}
    LAB[fam] = {m: int(mh[i] > 0) for m, i in OP_INDEX.items() if i < len(mh)}
    tot = sum(v for k, v in C[fam].items() if k != "H0") + 1e-30
    top = sorted(((v / tot, k) for k, v in C[fam].items() if k != "H0"), reverse=True)[:5]
    print(f"[{fam:44s}] H0 {C[fam].get('H0', 0):.4f} heads-sum {tot:.4f} | top " +
          " ".join(f"{k}={s:.2f}" for s, k in top) + f"  ({time.time()-t0:.0f}s)", flush=True)

# ---------------- derived statistics
heads = sorted({k for f in C for k in C[f] if k != "H0"})
ent, gini, share = {}, {}, {}
for f in C:
    x = np.array([C[f].get(k, 0.0) for k in heads]); s = x / (x.sum() + 1e-30)
    share[f] = dict(zip(heads, s.tolist()))
    p = s[s > 0]
    ent[f] = float(-(p * np.log(p)).sum() / np.log(len(heads)))      # normalised entropy (1 = uniform)
    xs = np.sort(s); n = len(xs)
    gini[f] = float((2 * np.arange(1, n + 1) - n - 1).dot(xs) / (n * xs.sum() + 1e-30))
# mask AUC over maskable heads, pooled (Mann-Whitney)
pos, neg = [], []
for f in C:
    for m, lab in LAB[f].items():
        if m in C[f]:
            (pos if lab else neg).append(share[f].get(m, 0.0))
auc = float("nan")
if pos and neg:
    P, N = np.array(pos), np.array(neg)
    auc = float(((P[:, None] > N[None, :]).sum() + 0.5 * (P[:, None] == N[None, :]).sum()) / (len(P) * len(N)))
print(f"[summary] mean normalised entropy {np.mean(list(ent.values())):.3f} | mean Gini {np.mean(list(gini.values())):.3f} "
      f"| mask AUC {auc:.3f} (pos {len(pos)}, neg {len(neg)}) | share on metadata-absent heads "
      f"{np.mean(neg) if neg else float('nan'):.3f} vs present {np.mean(pos) if pos else float('nan'):.3f}", flush=True)
json.dump({"ckpt": a.ckpt, "env": env, "C": C, "labels": LAB, "share": share, "entropy": ent,
           "gini": gini, "mask_auc": auc, "heads": heads}, open(a.out, "w"), indent=1)
print("CONTRIB_OK", a.out, flush=True)
