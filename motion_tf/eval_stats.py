"""Batched per-trajectory evaluation for statistics (revision R1).

Writes PER-TRAJECTORY, PER-FRAME errors (normalized / physical / fluctuation rel-L2, same
conventions as rollout_loss / singleshot_loss in v3/train.py) so that bootstrap CIs, paired
model comparisons, Wilcoxon tests, horizon curves etc. can be computed offline
(stats_compare.py) without re-running the model.

Pairing: every trajectory of every family gets a FIXED point-sampling seed (crc32 of
"family:index:draw"), so two models evaluated with this script see identical inputs.

Configurations evaluated in one pass (same draws):
  base               the model as trained
  ko:<head>          one mechanism head switched off (TendencyBanks.active minus <head>)
  warp_off           transport carrier removed (model.transport = False)

  python eval_stats.py --ckpt C --out OUT.npz [--split test] [--n 128] [--draws 1]
         [--configs base,warp_off,ko:advection,...] [--fams all] [--batch 8] [--batch3d 2]
         [--d 384 --depth 8 --d_w 16] [--rng_mode fixed|eval19] [--xla 1]
v2 (2026-10-15): knockouts / warp_off are RUNTIME switches (banks.ko_mask, model.warp_scale variables)
so the rollout is compiled ONCE per input shape instead of once per configuration, families are the
outer loop (examples built once and reused by every configuration), and the next family's examples
are built in a background thread. Outputs are identical to v1 (eval_stats_v1.py).
--rng_mode eval19 reproduces eval19_fam.py exactly (single global rng 1234, batch 1, val split):
use it once to validate this scorer against the per-family errors of the submitted version.
"""
import argparse, json, os, sys, time, zlib
import itertools
import numpy as np
import tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--draws", type=int, default=1)
ap.add_argument("--configs", default="base")
ap.add_argument("--fams", default="all")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--batch3d", type=int, default=2)
ap.add_argument("--d", type=int, default=384)
ap.add_argument("--depth", type=int, default=8)
ap.add_argument("--d_w", type=int, default=16)
ap.add_argument("--n_p", type=int, default=16)
ap.add_argument("--rng_mode", default="fixed")
ap.add_argument("--xla", type=int, default=1)
ap.add_argument("--bf16", type=int, default=0)
ap.add_argument("--t_obs", type=int, default=0, help="observed history frames (0 = full t_in); horizon test")
ap.add_argument("--k_fut", type=int, default=10, help="autoregressive forecast length")
ap.add_argument("--ar_only", type=int, default=0, help="skip single-shot and steady families")
ap.add_argument("--skip_ss", type=int, default=0, help="skip families evaluated in single-shot mode")
ap.add_argument("--k_relax", type=int, default=4, help="REVISION-R1 10-27: relaxation steps of steady families (MOTION-IVP FT: 1)")
a = ap.parse_args()
if a.bf16:
    tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")

from core.pretrain import load_ckpt
from data import loader
from data.registry import FAMILIES, NUM_SLOTS, SEMANTIC_SLOTS, MOTION_FAMILIES
from v3.data_ar import make_example_ar, stack_batch
from v3.model import V3Model
from v3.train import BOX_TABLE_2D, BOX_TABLE_3D, _rel, _grid

model = V3Model(S=NUM_SLOTS, d=a.d, depth=a.depth, d_w=a.d_w, n_p=a.n_p, d_cond=128,
                dims2=(64, 64), dims3=(32, 32, 32), t_in=10, transport=True,
                n_fams=len(FAMILIES), n_semantic=SEMANTIC_SLOTS,
                gate_cond=1, dict_decode=0, fam_head=1)
model.warm_build()
V = model.trainable_variables
load_ckpt(V, a.ckpt, d_w=a.d_w, names=[v.name for v in V])
ALL_HEADS = list(model.banks.names) + list(getattr(model.banks, "_gen", [])) + ["bernoulli", "kinematic", "vortex", "reaction_nl",
                                        "spectral", "coefficient", "compressible"]
print(f"[stats] {a.ckpt}: {sum(int(np.prod(v.shape)) for v in V)/1e6:.1f}M params | env "
      f"{ {k: v for k, v in os.environ.items() if k.startswith('MOTION_')} }", flush=True)
KEYS = ("coords_node", "x_win", "fmask", "geom_node", "y_node", "ic_q", "coords_q",
        "geom_q", "y_q", "tstep_mask", "cmask", "fbmask", "op_multihot", "cond", "fam_id", "sdf_box")


def bdims(fam):
    if int(os.environ.get("MOTION_BOX_UNIFORM", "0")):          # REVISION-R1: models trained with --box_table 0
        return (int(os.environ["MOTION_BOX_UNIFORM"]),) * FAMILIES[fam].K
    spec = FAMILIES[fam]
    r = (BOX_TABLE_2D if spec.K == 2 else BOX_TABLE_3D).get(fam, 64 if spec.K == 2 else 32)
    return (int(r),) * spec.K


def tensors(b):
    return tuple(tf.constant(b[k]) for k in KEYS)


def statics(b):
    gd = tuple(b.get("dims", ())) or None
    return (int(b["K"]), bool(b["steady"]), bool(b["dense2d"]),
            tuple(int(r) for r in b["roles"]), int(b["k_seg"]), gd,
            tuple(int(v) for v in b["box_dims"]), bool(b["surf"]))


def per_sample_ar(ts, sc, K, steady, dense2d, roles, k_seg, grid_dims, box_dims, surf):
    """rollout_loss, but returns per-sample per-segment rel arrays (B, k_seg) x 3 + weights."""
    cn, xw, fm, gn, yn, icq, cq, gq, yq, tsm, cm, fbm, op, cond, fid, sdb = ts
    Nn = tf.shape(cn)[1]
    qall = tf.concat([cn, cq], 1); gall = tf.concat([gn, gq], 1)
    u_prev = tf.concat([xw[:, -1], icq], 1)
    win, fmask = xw, fm
    prev_grid = prev2_grid = None
    if dense2d:
        prev_grid = _grid(xw[:, -1], grid_dims, model.S)
        if int(xw.shape[1]) >= 2:
            prev2_grid = _grid(xw[:, -2], grid_dims, model.S)
    rn, rp, rf, wt = [], [], [], []
    for k in range(k_seg):
        pred = model.step(cn, win, fmask, gn, K, roles, cond, op, qall, gall, u_prev,
                          steady=steady, prev_grid=prev_grid, grid_dims=grid_dims,
                          fam_id=fid, sdf_box=sdb, box_dims=box_dims, fbmask=fbm, surf=surf,
                          prev2_grid=prev2_grid)
        pred = tf.cast(pred, tf.float32)
        kt = 0 if steady else k
        y_all = tf.concat([yn[:, kt], yq[:, kt]], 1)
        rn.append(_rel(pred, y_all, cm, False))
        rp.append(_rel(pred, y_all, cm, False, sc))
        rf.append(_rel(pred, y_all, cm, True, fluct=True))
        wt.append(tf.fill(tf.shape(rn[-1]), float(k + 1)) if steady else tsm[:, k])
        u_prev = pred
        pred_n = u_prev[:, :Nn]
        win = tf.concat([win[:, 1:], pred_n[:, None]], 1)
        fmask = tf.concat([fmask[:, 1:], tf.ones_like(fmask[:, :1])], 1)
        if dense2d:
            prev2_grid, prev_grid = prev_grid, _grid(pred_n, grid_dims, model.S)
    st = lambda x: tf.stack(x, 1)
    return st(rn), st(rp), st(rf), st(wt)


def per_sample_ss(ts, sc, K, steady, dense2d, roles, k_seg, grid_dims, box_dims, surf):
    cn, xw, fm, gn, yn, icq, cq, gq, yq, tsm, cm, fbm, op, cond, fid, sdb = ts
    qall = tf.concat([cn, cq], 1); gall = tf.concat([gn, gq], 1)
    u_prev = tf.concat([xw[:, -1], icq], 1)
    preds = tf.cast(model.step_ss(cn, xw, fm, gn, K, roles, cond, op, qall, gall, u_prev,
                                  kf=k_seg, fam_id=fid, sdf_box=sdb, box_dims=box_dims), tf.float32)
    rn, rp, rf, wt = [], [], [], []
    for k in range(k_seg):
        y_all = tf.concat([yn[:, k], yq[:, k]], 1)
        rn.append(_rel(preds[:, k], y_all, cm, False))
        rp.append(_rel(preds[:, k], y_all, cm, False, sc))
        rf.append(_rel(preds[:, k], y_all, cm, True, fluct=True))
        wt.append(tsm[:, k])
    st = lambda x: tf.stack(x, 1)
    return st(rn), st(rp), st(rf), st(wt)



# REVISION-R1 10-29: receding-horizon ('rh') families (MOTION-IVP: MOTION_FORCE_MODE=ss + MOTION_RH_FAMS) are rolled out
# with their own hop-1 operator, model.step_ss(kf=RH_KF, leads=[1]), each prediction appended to the window, as at
# inference (score_ivp_l1.py HOP=1). They used to fall through to per_sample_ar (model.step = the per-frame operator,
# never trained by these models; its one-call readout equals the single-shot readout at the END of the RH_KF horizon).
RH_KF = int(os.environ.get("RH_KF", "20"))


def per_sample_rh(ts, sc, K, steady, dense2d, roles, k_seg, grid_dims, box_dims, surf):
    cn, xw, fm, gn, yn, icq, cq, gq, yq, tsm, cm, fbm, op, cond, fid, sdb = ts
    Nn = tf.shape(cn)[1]
    qall = tf.concat([cn, cq], 1); gall = tf.concat([gn, gq], 1)
    u_prev = tf.concat([xw[:, -1], icq], 1)
    win, fmask = xw, fm
    rn, rp, rf, wt = [], [], [], []
    for k in range(k_seg):
        pg = _grid(win[:, -1], grid_dims, model.S) if dense2d else None
        pred = tf.cast(model.step_ss(cn, win, fmask, gn, K, roles, cond, op, qall, gall, u_prev, kf=RH_KF,
                                     fam_id=fid, sdf_box=sdb, box_dims=box_dims, prev_grid=pg,
                                     grid_dims=grid_dims if dense2d else None, leads=[1])[:, 0], tf.float32)
        y_all = tf.concat([yn[:, k], yq[:, k]], 1)
        rn.append(_rel(pred, y_all, cm, False))
        rp.append(_rel(pred, y_all, cm, False, sc))
        rf.append(_rel(pred, y_all, cm, True, fluct=True))
        wt.append(tsm[:, k])
        u_prev = pred
        win = tf.concat([win[:, 1:], pred[:, :Nn][:, None]], 1)
        fmask = tf.concat([fmask[:, 1:], tf.ones_like(fmask[:, :1])], 1)
    st = lambda x: tf.stack(x, 1)
    return st(rn), st(rp), st(rf), st(wt)

model.banks.ko_index = {h: i for i, h in enumerate(ALL_HEADS)}
model.banks.ko_mask = tf.Variable(tf.ones([len(ALL_HEADS)]), trainable=False, name="ko_mask")
model.warp_scale = tf.Variable(1.0, trainable=False, name="warp_scale")
model.z_scale = tf.Variable(1.0, trainable=False, name="z_scale")    # "z0": readout input -> 0
f_ar = tf.function(per_sample_ar, reduce_retracing=True, jit_compile=bool(a.xla))
f_ss = tf.function(per_sample_ss, reduce_retracing=True, jit_compile=bool(a.xla))
f_rh = tf.function(per_sample_rh, reduce_retracing=True, jit_compile=bool(a.xla))


def apply_config(cfg):
    """Assign the runtime switches (no retrace): ko:<head> zeroes that head, warp_off zeroes the warp."""
    m = np.ones(len(ALL_HEADS), np.float32)
    for hname in [h for h in os.environ.get("MOTION_TRAIN_KO", "").split(",") if h]:   # REVISION-R1: heads removed
        m[model.banks.ko_index[hname]] = 0.0                                          # during training stay off
    if cfg.startswith("ko:"):
        for hname in cfg[3:].split("+"):                  # "ko:a+b" = simultaneous knockout
            assert hname in model.banks.ko_index, f"unknown head {hname}"
            m[model.banks.ko_index[hname]] = 0.0
    model.banks.ko_mask.assign(m)
    model.warp_scale.assign(0.0 if cfg == "warp_off" else 1.0)
    model.z_scale.assign(0.0 if cfg == "z0" else 1.0)     # B-R: exact anchor; V3: anchor + D(0)


def seed_for(fam, idx, draw):
    return zlib.crc32(f"{fam}:{idx}:{draw}".encode())


configs = [c for c in a.configs.split(",") if c]
fams = list(MOTION_FAMILIES) if a.fams == "all" else a.fams.split(",")
# REVISION-R1 10-27: --configs ko_all = base + one knockout per head (identically-zero heads excluded)
if configs == ["ko_all"]:
    configs = ["base"] + [f"ko:{h}" for h in dict.fromkeys(ALL_HEADS) if h not in ("vortex", "boundary_layer")]
    print(f"[stats] ko_all: {len(configs) - 1} heads {configs[1:]}", flush=True)
# REVISION-R1 10-27: MOTION_GLOBAL_NORM models (MOTION-IVP) need the per-family statistics before examples are built
import v3.data_ar as _DA
if _DA.GLOBAL_NORM_ON:
    for _f in fams:
        _DA.GLOBAL_NORM[_f] = _DA.compute_global_norm(loader.units_for(_f, "train"))[:2]
        print(f"[stats] global norm {_f}", flush=True)
if a.rng_mode == "eval19":
    a.batch = a.batch3d = 1
    a.draws = 1
    assert configs == ["base"] or len(configs) == 1, "eval19 mode reproduces a single configuration"
g19 = np.random.default_rng(1234)                         # eval19's single global rng


def build(fam):
    """All draws' examples for one family (deterministic per-trajectory seeds; eval19: global rng)."""
    try:
        units = list(itertools.islice(loader.units_for(fam, a.split), a.n))
    except Exception as e:
        return fam, None, f"{type(e).__name__} {e}"
    if not units:
        return fam, None, "no units"
    if a.ar_only and (FAMILIES[fam].time == "steady" or fam in ("ACE",)):
        return fam, None, "not autoregressive (horizon test)"
    draws = []
    for draw in range(a.draws):
        exs = []
        for idx, s in enumerate(units):
            rng = g19 if a.rng_mode == "eval19" else np.random.default_rng(seed_for(fam, idx, draw))
            exs.append(make_example_ar(s, 10, a.k_fut, a.k_relax, 2048, 8192, rng, box_dims=bdims(fam), ti_fixed=a.t_obs))
        draws.append(exs)
    if a.skip_ss and draws and draws[0] and draws[0][0]["mode"] == "ss":
        return fam, None, "single-shot mode skipped (--skip_ss)"
    return fam, draws, len(units)


from concurrent.futures import ThreadPoolExecutor
out = {}
t0 = time.time()
pool = ThreadPoolExecutor(1)                              # one worker: keeps eval19's rng order
futs = [pool.submit(build, f) for f in fams[:1]]
for fi, fam in enumerate(fams):
    if fi + 1 < len(fams):
        futs.append(pool.submit(build, fams[fi + 1]))     # prefetch the next family
    _, draws, info = futs[fi].result()
    if draws is None:
        print(f"  [skip] {fam}: {info}", flush=True)
        continue
    spec = FAMILIES[fam]
    B = a.batch3d if spec.K == 3 else a.batch
    batches = [[stack_batch(exs[i:i + B]) for i in range(0, len(exs), B)] for exs in draws]
    for cfg in configs:
        apply_config(cfg)
        res = {k: [] for k in ("norm", "phys", "fluct", "w")}
        for bl in batches:
            per = {k: [] for k in res}
            for b in bl:
                fn = f_ss if b["mode"] == "ss" else (f_rh if b["mode"] == "rh" else f_ar)   # REVISION-R1 10-29: rh = MOTION-IVP
                _sc = (np.concatenate([b["scale"], b["shift"]], -1)            # REVISION-R1 z-score arm: add the mean back
                       if os.environ.get("MOTION_INPUT_ZSCORE", "0") == "1" else b["scale"])   # (same as v3/train.py eval)
                rn, rp, rf, wt = fn(tensors(b), tf.constant(_sc), *statics(b))
                for k, v in zip(("norm", "phys", "fluct", "w"), (rn, rp, rf, wt)):
                    per[k].append(v.numpy())
            for k in res:
                res[k].append(np.concatenate(per[k], 0))            # (n, k_seg)
        for k in res:
            out[f"{cfg}|{fam}|{k}"] = np.stack(res[k], 1).astype(np.float32)   # (n, draws, k_seg)
        w = out[f"{cfg}|{fam}|w"]
        traj = lambda m: (out[f"{cfg}|{fam}|{m}"] * w).sum(-1) / np.maximum(w.sum(-1), 1e-6)
        print(f"[{cfg}] {fam:44s} n={info:3d} phys={100*traj('phys').mean():7.3f} "
              f"norm={100*traj('norm').mean():7.3f}  ({time.time()-t0:.0f}s)", flush=True)
    del batches, draws
apply_config("base")
meta = {"ckpt": a.ckpt, "split": a.split, "n": a.n, "draws": a.draws, "configs": configs,
        "rng_mode": a.rng_mode, "env": {k: v for k, v in os.environ.items() if k.startswith("MOTION_")},
        "arch": [a.d, a.depth, a.d_w, a.n_p], "bf16": a.bf16}
np.savez_compressed(a.out, **out, __meta__=json.dumps(meta))
print(f"STATS_OK {a.out} ({time.time()-t0:.0f}s)", flush=True)
