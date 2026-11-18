"""REVISION-R1: 3-D inference with point-sample averaging through the NODE decoder (the eval_stats readout).
For every test trajectory of a 3-D family, K examples of the same state are built with independent encoder point
samples; at every autoregressive step their latent rates W are averaged (model.w_ens hook in V3Model.step) and the
prediction is decoded once at the scoring points of sample 0 (its nodes + query points, exactly as eval_stats) and at
the node points of the other samples, whose windows advance with their own predictions. K=1 reproduces eval_stats.
  python eval_ens3d.py --ckpt C --out OUT.npz [--fams ...] [--K 1,8] [--n 128]
"""
import argparse, itertools, json, os, zlib
import numpy as np
import tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--fams", default="pdebench3d_cns_Rand_M0.1_Eta1e-08_Zeta1e-08,pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08,"
                                  "pdebench3d_cns_Turb_M1.0_Eta1e-08_Zeta1e-08")
ap.add_argument("--K", default="1,8"); ap.add_argument("--n", type=int, default=128); ap.add_argument("--k_fut", type=int, default=10)
ap.add_argument("--split", default="test")
a = ap.parse_args()

from core.pretrain import load_ckpt
from data import loader
from data.registry import FAMILIES, NUM_SLOTS, SEMANTIC_SLOTS
from v3.data_ar import make_example_ar, stack_batch
from v3.model import V3Model
from v3.train import BOX_TABLE_3D, _rel

model = V3Model(S=NUM_SLOTS, d=int(os.environ.get("ARCH_D", 384)), depth=int(os.environ.get("ARCH_DEPTH", 8)),
                d_w=int(os.environ.get("ARCH_DW", 16)), n_p=16, d_cond=128, dims2=(64, 64), dims3=(32, 32, 32), t_in=10,
                transport=True, n_fams=len(FAMILIES), n_semantic=SEMANTIC_SLOTS, gate_cond=1, dict_decode=0, fam_head=1)
model.warm_build(); V = model.trainable_variables
load_ckpt(V, a.ckpt, d_w=int(os.environ.get("ARCH_DW", 16)), names=[v.name for v in V])
T = lambda x: tf.constant(x)
XLA = int(os.environ.get("ENS_XLA", 1))


def make_step(Kd, roles, bx):
    """One compiled AR step per (family, K): the ensemble samples are explicit inputs and set model.w_ens at trace time
    (eager 32^3 runs out of memory; eval_stats compiles with XLA as well)."""
    def f(cn, win, fm, gn, cond, op, q, g, u, fid, sdb, fbm, ens):
        model.w_ens = list(ens) if ens else None
        out = model.step(cn, win, fm, gn, Kd, roles, cond, op, q, g, u, steady=False, fam_id=fid, sdf_box=sdb,
                         box_dims=bx, fbmask=fbm)
        model.w_ens = None
        return tf.cast(out[0] if isinstance(out, (tuple, list)) else out, tf.float32)
    return tf.function(f, jit_compile=bool(XLA))


_STEP = {}


def rollout(exs):
    """exs: K stacked single-example batches of the same state -> per-frame (rel_norm, rel_phys) for sample 0."""
    b0 = exs[0]; Kd = int(b0["K"]); roles = [int(r) for r in b0["roles"]]; bx = tuple(int(v) for v in b0["box_dims"])
    nN = [int(np.asarray(e["coords_node"]).shape[1]) for e in exs]
    q_all = np.concatenate([b0["coords_node"], b0["coords_q"]] + [e["coords_node"] for e in exs[1:]], 1)
    g_all = np.concatenate([b0["geom_node"], b0["geom_q"]] + [e["geom_node"] for e in exs[1:]], 1)
    u_prev = np.concatenate([b0["x_win"][:, -1], b0["ic_q"]] + [e["x_win"][:, -1] for e in exs[1:]], 1)
    wins = [np.array(e["x_win"]) for e in exs]; fms = [np.array(e["fmask"]) for e in exs]
    nq0 = nN[0] + int(np.asarray(b0["coords_q"]).shape[1])
    sc = T(b0["scale"]); cm = T(b0["cmask"]); rn, rp, w = [], [], []
    for k in range(int(b0["k_seg"])):
        key = (Kd, tuple(roles), bx, len(exs))
        if key not in _STEP: _STEP[key] = make_step(Kd, roles, bx)
        ens = tuple((T(exs[i]["coords_node"]), T(wins[i]), T(fms[i]), T(exs[i]["geom_node"])) for i in range(1, len(exs)))
        pred = _STEP[key](T(b0["coords_node"]), T(wins[0]), T(fms[0]), T(b0["geom_node"]), T(b0["cond"]),
                          T(b0["op_multihot"]), T(q_all), T(g_all), T(u_prev), T(b0["fam_id"]), T(b0["sdf_box"]),
                          T(b0["fbmask"]), ens)
        y_all = tf.concat([T(b0["y_node"])[:, k], T(b0["y_q"])[:, k]], 1)
        rn.append(float(_rel(pred[:, :nq0], y_all, cm, False)[0])); rp.append(float(_rel(pred[:, :nq0], y_all, cm, False, sc)[0]))
        w.append(float(b0["tstep_mask"][0, k]))
        P = pred.numpy(); u_prev = P
        off = [0, nq0] + list(np.cumsum([nq0] + nN[1:]))[1:]
        parts = [P[:, :nN[0]]] + [P[:, off[i]:off[i] + nN[i]] for i in range(1, len(exs))]
        for i in range(len(exs)):
            wins[i] = np.concatenate([wins[i][:, 1:], parts[i][:, None]], 1)
            fms[i] = np.concatenate([fms[i][:, 1:], np.ones_like(fms[i][:, :1])], 1)
    return np.array(rn), np.array(rp), np.array(w)


out, summ = {}, {}
for fam in a.fams.split(","):
    bd = BOX_TABLE_3D.get(fam, (32, 32, 32)); bd = (bd,) * 3 if isinstance(bd, int) else tuple(bd)
    units = list(itertools.islice(loader.units_for(fam, a.split), a.n))
    for Kn in [int(x) for x in a.K.split(",")]:
        E = []
        for i, u in enumerate(units):
            exs = [stack_batch([make_example_ar(u, 10, a.k_fut, 4, 2048, 8192,
                                                np.random.default_rng(zlib.crc32(f"{fam}:{i}:{j}".encode())), box_dims=bd)])
                   for j in range(Kn)]
            rn, rp, w = rollout(exs); E.append(((rp * w).sum() / w.sum(), rp))
        out[f"{fam}|K{Kn}|phys"] = np.array([e[1] for e in E], np.float32)
        m = 100 * float(np.mean([e[0] for e in E])); summ[f"{fam}|K{Kn}"] = m
        print(f"[ens3d] {fam} K={Kn} n={len(units)} phys {m:.3f}%", flush=True)
        np.savez_compressed(a.out, **out, __meta__=json.dumps({"args": vars(a), "summary": summ}))
print("ENS3D_DONE", json.dumps(summ), flush=True)
