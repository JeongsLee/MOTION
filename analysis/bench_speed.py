"""Synthetic-input speed benchmark for the MOTION (foundationv2 / v3) AR step.

No data: tensors are built at the per-family shapes used in training (box R from BOX_TABLE,
enc_n points, enc_n + n_colloc queries, dense 128^2 anchor grid for 2D). Weights are random
(warm_build), which does not affect timing.

Measured per case:
  - full AR step, forward and forward+backward, under bank configs
      all  : current implementation (every head computed, metadata mask applied afterwards)
      fam  : only heads whose metadata gate is open for the family (+ unmasked heads),
             vortex dropped for K=2  -> upper bound of the gating optimisation
      base : state head H0 only     -> total cost of the gated bank
  - forward component split: trunk (encode + core), bank, remainder (interp + decode)

Usage: PYTHONPATH=<release foundationv2> python bench_speed.py [--bf16 1] [--d 384 --depth 8]
"""
import argparse, json, os, sys, time
import numpy as np
import tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument("--d", type=int, default=384)
ap.add_argument("--depth", type=int, default=8)
ap.add_argument("--d_w", type=int, default=16)
ap.add_argument("--n_p", type=int, default=16)
ap.add_argument("--enc_n", type=int, default=8192)
ap.add_argument("--n_colloc", type=int, default=2048)
ap.add_argument("--b2d", type=int, default=8)
ap.add_argument("--b3d", type=int, default=1)
ap.add_argument("--cases", default="incom_ns,CE-RP,pdearena_ns,pdebench3d_cns_Rand_M0.1_Eta1e-08_Zeta1e-08,pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08")
ap.add_argument("--configs", default="all,fam,base")
ap.add_argument("--bwd", type=int, default=1)
ap.add_argument("--reps", type=int, default=15)
ap.add_argument("--bf16", type=int, default=0)
ap.add_argument("--xla", type=int, default=1)
ap.add_argument("--opt", type=int, default=0)   # 1: fast attention + static gating + vortex K=3 (needs fv2_opt)
ap.add_argument("--out", default="bench_out.json")
a = ap.parse_args()

for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
if a.bf16:
    tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")

from data.registry import FAMILIES, NUM_SLOTS
from data.symbolic import OPERATOR_VOCAB, OP2ID
from core.banks import OP_INDEX, STEADY_DROP
from v3.model import V3Model
from v3.cond import COND_DIM
from v3.train import BOX_TABLE_2D, BOX_TABLE_3D

if a.opt:
    from core.axops import AxialAttention
    os.environ["MOTION_VORTEX2D_LEGACY"] = "0"
    AxialAttention.FAST = False          # build through the legacy path, then switch
model = V3Model(S=NUM_SLOTS, d=a.d, depth=a.depth, d_w=a.d_w, n_p=a.n_p, d_cond=128,
                dims2=(64, 64), dims3=(32, 32, 32), t_in=10, transport=True,
                n_fams=len(FAMILIES), gate_cond=1, fam_head=1, dict_decode=0)
model.warm_build()
if a.opt:
    AxialAttention.FAST = os.environ.get("MOTION_FAST_ATTN", "1") == "1"
V = model.trainable_variables
print(f"[bench] params {sum(int(np.prod(v.shape)) for v in V)/1e6:.1f}M  gpu={tf.config.list_physical_devices('GPU')}", flush=True)

ALL_HEADS = list(model.banks.names) + ["bernoulli", "kinematic", "vortex", "reaction_nl",
                                        "spectral", "coefficient", "compressible"]


def fam_active(f):
    mh = {OP2ID[o] for o in f.operators if o in OP2ID}
    act = {m for m in ALL_HEADS if (m not in OP_INDEX) or (OP_INDEX[m] in mh)}
    if f.K == 2:
        act.discard("vortex")
    if getattr(f, "steady", False):
        act -= set(STEADY_DROP)
    return act


def inputs(fam):
    f = FAMILIES[fam]
    K = f.K
    R = (BOX_TABLE_2D if K == 2 else BOX_TABLE_3D).get(fam, 32)
    B = a.b2d if K == 2 else a.b3d
    N, Q = a.enc_n, a.enc_n + a.n_colloc
    S = NUM_SLOTS
    mh = np.zeros([B, len(OPERATOR_VOCAB)], np.float32)
    for o in f.operators:
        if o in OP2ID:
            mh[:, OP2ID[o]] = 1.0
    t = dict(cn=tf.random.uniform([B, N, K]), win=tf.random.normal([B, 10, N, S]),
             fm=tf.ones([B, 10]), gn=tf.zeros([B, N, 8]), cond=tf.zeros([B, COND_DIM]),
             op=tf.constant(mh), cq=tf.random.uniform([B, Q, K]), gq=tf.zeros([B, Q, 8]),
             up=tf.random.normal([B, Q, S]), fid=tf.zeros([B], tf.int32),
             sb=tf.zeros([B, *([R] * K), 1]))
    if K == 2:
        t["pg"] = tf.random.normal([B, 128, 128, S])
    t["ops"] = tuple(sorted(OP2ID[o] for o in f.operators if o in OP2ID))
    return f, K, R, B, t


def make_fns(f, K, R, t):
    roles = list(range(K))
    dims = (R,) * K
    grid = (128, 128) if K == 2 else None

    def step():
        return model.step(t["cn"], t["win"], t["fm"], t["gn"], K, roles, t["cond"], t["op"],
                          t["cq"], t["gq"], t["up"], steady=False, prev_grid=t.get("pg"),
                          grid_dims=grid, fam_id=t["fid"], sdf_box=t["sb"], box_dims=dims,
                          prev2_grid=t.get("pg"),
                          **({"op_static": t["ops"]} if a.opt else {}))

    def trunk():
        feats = model.feats_from_win(t["win"], t["fm"], t["gn"])
        box, geom_box, _ = model.enc(t["cn"], feats, dims=dims, roles=roles)
        box = box + model.sdf_gate(model.sdf_proj(t["sb"]))
        geom_box = tf.concat([tf.cast(t["sb"], geom_box.dtype), geom_box[..., 1:]], -1)
        e = tf.cast(model.cond(t["cond"]), box.dtype)
        return model.core(box, e, roles), geom_box, e

    fwd = tf.function(lambda: tf.reduce_sum(tf.cast(step(), tf.float32)), jit_compile=bool(a.xla))

    def _fb():
        with tf.GradientTape() as tape:
            L = tf.reduce_sum(tf.cast(step(), tf.float32))
        g = tape.gradient(L, V)
        return L + tf.add_n([tf.reduce_sum(tf.cast(x, tf.float32)) for x in g if x is not None]) * 0.0
    fb = tf.function(_fb, jit_compile=bool(a.xla))
    tr = tf.function(lambda: tf.reduce_sum(tf.cast(trunk()[0], tf.float32)), jit_compile=bool(a.xla))

    def _en():
        feats = model.feats_from_win(t["win"], t["fm"], t["gn"])
        box, _, _ = model.enc(t["cn"], feats, dims=dims, roles=roles)
        return tf.reduce_sum(tf.cast(box, tf.float32))
    en = tf.function(_en, jit_compile=bool(a.xla))

    def _bk():
        h, gb, e = trunk()
        return tf.reduce_sum(tf.cast(model.banks(h, K=K, roles=roles, op_multihot=t["op"],
                                                 geom_box=gb, steady=False, e=e,
                                                 **({"op_static": t["ops"]} if a.opt else {})), tf.float32))
    bk = tf.function(_bk, jit_compile=bool(a.xla))
    return fwd, fb, tr, bk, en


def timeit(fn):
    t0 = time.perf_counter(); fn().numpy(); comp = time.perf_counter() - t0
    for _ in range(2):
        fn().numpy()
    ts = []
    for _ in range(a.reps):
        t1 = time.perf_counter(); fn().numpy(); ts.append(time.perf_counter() - t1)
    return float(np.median(ts)) * 1e3, comp


res = []
for fam in a.cases.split(","):
    f, K, R, B, t = inputs(fam)
    act_f = fam_active(f)
    for cfg in a.configs.split(","):
        model.banks.active = {"all": None, "fam": act_f, "base": set()}[cfg]
        try:
            fwd, fb, tr, bk, en = make_fns(f, K, R, t)
            r = {"fam": fam, "K": K, "R": R, "B": B, "cfg": cfg,
                 "n_heads": len(ALL_HEADS) if cfg == "all" else len(act_f) if cfg == "fam" else 0}
            r["fwd_ms"], r["fwd_compile_s"] = timeit(fwd)
            r["enc_ms"], _ = timeit(en)
            r["trunk_ms"], _ = timeit(tr)
            r["trunk+bank_ms"], _ = timeit(bk)
            if a.bwd:
                r["fwdbwd_ms"], r["fwdbwd_compile_s"] = timeit(fb)
        except tf.errors.ResourceExhaustedError as ex:
            r = {"fam": fam, "K": K, "R": R, "B": B, "cfg": cfg, "error": "OOM"}
        res.append(r)
        print("[bench] " + json.dumps(r), flush=True)
        tf.keras.backend.clear_session() if False else None
model.banks.active = None
json.dump({"args": vars(a), "results": res}, open(a.out, "w"), indent=1)
print("[bench] saved", a.out, flush=True)
