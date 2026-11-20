"""Score a v3 checkpoint with the PREVIOUS paper's Poseidon IVP metric so the two runs sit on one
axis: MEDIAN rel-L1 in PHYSICAL space, per family, reported both @final frame and full-trajectory
(mean over leads 1..KF). The old runner (code/train/train_ivp.py) restricted this to SOLUTION_SLOTS
because its pipeline fed coefficient fields as scored slots; v3 routes coefficients through `cond`,
so every active slot IS a solution slot and the family c_mask is the exact analogue.

Why rel-L1-physical rather than our joint rel-L2: it weights a channel by its physical magnitude,
so a channel that is constant at t=0 (CE-KH's transverse velocity) contributes to neither numerator
nor denominator instead of dominating one of them. That is why the old runs read CE-KH at ~3%.
"""
import os
import traceback

import numpy as np
import tensorflow as tf
try:                                                  # Keras 2.15 eager leak: Softmax keeps its logits on the output
    from keras.src.layers.activation import softmax as _smod   # tensor -> memory grows per call until OOM (n=58 of 128)
    _sm_call = _smod.Softmax.call
    def _sm_nolog(self, inputs, mask=None):
        out = _sm_call(self, inputs, mask)
        try: del out._keras_logits
        except AttributeError: pass
        return out
    _smod.Softmax.call = _sm_nolog
except Exception:
    pass

from core.pretrain import load_ckpt
from data import loader
from data.registry import FAMILIES, NUM_SLOTS, SEMANTIC_SLOTS
from v3.data_ar import make_example_ar, stack_batch
from v3 import data_ar as _DA
from v3.model import V3Model
from v3.train import BOX_TABLE_2D, BOX_TABLE_3D, lead_cond
# REVISION-R1 10-24: MOTION_LC=1 = lead-conditioned checkpoint (stage 3 option B; cond carries [1, L/KF]): the "direct"
# forecast is one call per lead L with cond L, and every hop-AR call is conditioned on L = HOP. Batched path only.
LC = os.environ.get("MOTION_LC", "0") == "1"

CKPT = os.environ["CKPT"]
FAMS = os.environ.get("FAMS", "CE-RP,CE-CRP,CE-KH,CE-Gauss,NS-Sines,NS-Gauss").split(",")
KF = int(os.environ.get("KF", "20"))
NEV = int(os.environ.get("NEV", "32"))
T_IN = int(os.environ.get("T_IN", "1"))
# REVISION-R1 10-20: HOP > 0 (with DECODE=ss): hop-AR inference of a single-shot model, as the published MOTION-IVP NS-PwC
# number - the lead-HOP output of a full KF-frame single-shot call becomes the next initial condition (re-normalized
# exactly as the loader does: per-channel std, and mean when MOTION_INPUT_ZSCORE), KF/HOP calls; scored on leads
# HOP, 2*HOP, ..., KF. The direct single-shot score on the same leads is printed alongside.
HOP = int(os.environ.get("HOP", "0"))
_ZS = os.environ.get("MOTION_INPUT_ZSCORE", "0") == "1"
SPLIT = os.environ.get("SPLIT", "val")
# REVISION-R1 10-24: Poseidon-paper metric (Herde et al. 2024, SM D.1 / Table 6) next to ours: relative L1 at ONE lead
# (PAPER_LEAD; steady tasks: the solution) per function of interest (PAPER_FUNCS = slot groups "0,1;3;4" - a vector field
# is one function), mean over the functions, median over trajectories. Reported for the direct decode and, with HOP,
# for the hop-AR decode.
PAPER_LEAD = int(os.environ.get("PAPER_LEAD", "0"))
PAPER_FUNCS = [[int(c) for c in g.split(",")] for g in os.environ.get("PAPER_FUNCS", "").split(";") if g]


def _paper_err(P, Y, ptm):
    """P, Y: (N+Q, S) physical; ptm: (N+Q,) point weights. Mean over functions of sum|P-Y| / sum|Y|."""
    es = []
    for g in PAPER_FUNCS:
        es.append(float(np.sum(np.abs(P[:, g] - Y[:, g]) * ptm[:, None]) / (np.sum(np.abs(Y[:, g]) * ptm[:, None]) + 1e-12)))
    return float(np.mean(es))
DECODE = os.environ.get("DECODE", "auto")

model = V3Model(S=NUM_SLOTS, d=384, depth=8, d_w=16, n_p=int(os.environ.get("N_P", "16")), d_cond=128,   # MOTION-IVP-S ss: 32
                dims2=(64, 64), dims3=(32, 32, 32), t_in=T_IN, transport=1,
                n_fams=len(FAMILIES), n_semantic=SEMANTIC_SLOTS, gate_cond=1, fam_head=1,
                dict_decode=0)
model.warm_build()
V = model.trainable_variables
load_ckpt(V, CKPT, d_w=16, names=[v.name for v in V])
# FINGERPRINT the loaded weights: two different checkpoints scoring identically to four
# significant figures is not something weak backbone dependence can produce, so verify that the
# load actually landed rather than trusting load_ckpt's own count.
_fp = {v.name: float(tf.reduce_sum(tf.cast(v, tf.float64)).numpy())
       for v in V if v.name in ("b0_attn/b0_attn_mha/query/kernel:0", "dec_o/kernel:0",
                                "dec_famW:0")}
print(f"[score] loaded {CKPT} | {sum(int(np.prod(v.shape)) for v in V)/1e6:.1f}M params",
      flush=True)
print(f"[score] weight fingerprint: "
      + "  ".join(f"{k.split('/')[-1]}={x:+.6e}" for k, x in _fp.items()), flush=True)


def box_of(K, fam):
    u = int(os.environ.get("BOX_UNIFORM", "0"))           # REVISION-R1: models trained with --box_table 0 (uniform)
    if u:
        return (u,) * K
    r = (BOX_TABLE_2D if K == 2 else BOX_TABLE_3D).get(fam)
    return (int(r),) * K if r else ((32,) * 2 if K == 2 else (16,) * 3)


if _DA.GLOBAL_NORM_ON:                                       # REVISION-R1 Poseidon-style normalization: the same
    assert _ZS, "MOTION_GLOBAL_NORM=1 needs MOTION_INPUT_ZSCORE=1"   # per-family statistics as in training (train split,
    for fam in FAMS:                                        # first 64 trajectories; for a FT task = its FT trajectories)
        m_, s_, n_ = _DA.compute_global_norm(loader.units_for(fam, "train"))
        _DA.GLOBAL_NORM[fam] = (m_, s_)
        print(f"[score] global norm {fam} (n={n_}): mean {np.round(m_, 4).tolist()} std {np.round(s_, 4).tolist()}", flush=True)

SCORE_BATCH = int(os.environ.get("SCORE_BATCH", "1"))


def _batched_predict(exs, i0, fam):
    """Single-shot prediction (+ hop-HOP AR in physical units) for a list of same-family examples.
    Returns {i0 + i: (pred (1,KF,N+Q,S) normalized, {lead: (N+Q,S) physical})}; mirrors the per-sample code."""
    b = stack_batch(exs); B = len(exs)
    cn, xw, fm, gn = (tf.constant(b[k]) for k in ("coords_node", "x_win", "fmask", "geom_node"))
    icq, cq, gq = (tf.constant(b[k]) for k in ("ic_q", "coords_q", "geom_q"))
    qall = tf.concat([cn, cq], 1); gall = tf.concat([gn, gq], 1)
    u_prev = tf.concat([xw[:, -1], icq], 1)
    K, roles = int(b["K"]), tuple(int(r) for r in b["roles"])
    cond, op = tf.constant(b["cond"]), tf.constant(b["op_multihot"])
    fid, sdb = tf.constant(b["fam_id"]), tf.constant(b["sdf_box"])
    bx, dims = box_of(K, fam), tuple(b.get("dims", ()))
    _d2 = bool(b["dense2d"]) and K == 2
    _pg0 = tf.reshape(xw[:, -1], [B, *dims, NUM_SLOTS]) if _d2 else None
    if LC:
        pred = np.stack([model.step_ss(cn, xw, fm, gn, K, roles, lead_cond(cond, k, KF), op, qall, gall, u_prev, kf=KF,
                                       fam_id=fid, sdf_box=sdb, box_dims=bx, prev_grid=_pg0,
                                       grid_dims=dims if _d2 else None, leads=[k]).numpy()[:, 0]
                         for k in range(1, KF + 1)], 1)                                 # (B,KF,N+Q,S) direct per lead
    else:
        pred = model.step_ss(cn, xw, fm, gn, K, roles, cond, op, qall, gall, u_prev, kf=KF, fam_id=fid, sdf_box=sdb,
                             box_dims=bx, prev_grid=_pg0, grid_dims=dims if _d2 else None).numpy()   # (B,KF,N+Q,S)
    Ps = [dict() for _ in range(B)]
    if HOP > 0:
        scv = b["scale"].reshape(B, 1, -1); shv = (b["shift"].reshape(B, 1, -1) if "shift" in b else np.zeros_like(scv))
        cmv = b["cmask"].reshape(B, 1, -1) > 0; Nn = int(cn.shape[1])
        X = pred[:, HOP - 1] * scv + shv                                                # (B,N+Q,S) physical
        for i in range(B):
            Ps[i][HOP] = X[i]
        _hl = None if os.environ.get("HOP_FULL") == "1" else [HOP]
        # REVISION-R1 10-25: the fed-back frame is APPENDED to the observed window (fmask 1), as in training
        # (singleshot_rh_loss); for a t_in-1 model this is the previous behaviour (window = the new frame).
        win_, fmw_ = xw, fm
        for L_ in range(2 * HOP, KF + 1, HOP):
            if _DA.GLOBAL_NORM_ON:
                sc_, sh_ = scv, shv
            else:
                sd_ = X[:, :Nn].std(1, keepdims=True); mu_ = X[:, :Nn].mean(1, keepdims=True)
                sc_ = np.where((sd_ > 1e-6) & cmv, sd_, 1.0)
                sh_ = np.where(cmv, mu_, 0.0) if _ZS else np.zeros_like(sc_)
            Xn = ((X - sh_) / sc_).astype(np.float32)
            win_ = tf.concat([win_[:, 1:], tf.constant(Xn[:, None, :Nn])], 1); up_ = tf.constant(Xn)
            fmw_ = tf.concat([fmw_[:, 1:], tf.ones_like(fmw_[:, :1])], 1)
            pg_ = tf.reshape(win_[:, -1], [B, *dims, NUM_SLOTS]) if _d2 else None
            pr_ = model.step_ss(cn, win_, fmw_, gn, K, roles, lead_cond(cond, HOP, KF) if LC else cond, op, qall, gall,
                                up_, kf=KF, fam_id=fid, sdf_box=sdb, box_dims=bx, prev_grid=pg_,
                                grid_dims=dims if _d2 else None, leads=_hl).numpy()
            X = pr_[:, (HOP - 1) if _hl is None else 0] * sc_ + sh_
            for i in range(B):
                Ps[i][L_] = X[i]
    return {i0 + i: (pred[i:i + 1], Ps[i]) for i in range(B)}



# REVISION-R1 10-29 (user): batched + compiled AR decode. The AR branch (model.step, the per-frame operator of the
# MOTION-S / MOTION-NS / MOTION-NSP models) used to run one trajectory per call in eager mode. _batched_predict_ar runs
# SCORE_BATCH same-family trajectories per call, and (SCORE_TFF=1, default) one AR step is a tf.function traced once per
# (family, batch size); the rollout itself mirrors the per-sample AR branch below exactly (prediction appended to the
# window with fmask 1, prev_grid = the new frame). Validate with SCORE_BATCH=1 vs >1 on the same NEV.
SCORE_TFF = os.environ.get("SCORE_TFF", "1") == "1"
_STEP_FN = {}


def _ar_step_fn(key, K, roles, dims, dense2d, bx):
    if key in _STEP_FN:
        return _STEP_FN[key]
    _g = (K == 2 and dense2d)

    def _f(cn, win, fmask, gn, cond, op, qall, gall, up, prev_grid, fid, sdb, fb):
        return model.step(cn, win, fmask, gn, K, roles, cond, op, qall, gall, up, steady=False,
                          prev_grid=prev_grid if _g else None, grid_dims=dims if dense2d else None, fam_id=fid,
                          sdf_box=sdb, box_dims=bx, fbmask=fb)
    _STEP_FN[key] = tf.function(_f, reduce_retracing=True) if SCORE_TFF else _f
    return _STEP_FN[key]


def _batched_predict_ar(exs, i0, fam):
    """AR rollout (model.step, KF frames) for a list of same-family transient examples.
    Returns {i0 + i: (pred (1,KF,N+Q,S) normalized, {})}; mirrors the per-sample AR branch."""
    b = stack_batch(exs); B = len(exs)
    cn, xw, fm, gn = (tf.constant(b[k]) for k in ("coords_node", "x_win", "fmask", "geom_node"))
    icq, cq, gq = (tf.constant(b[k]) for k in ("ic_q", "coords_q", "geom_q"))
    qall = tf.concat([cn, cq], 1); gall = tf.concat([gn, gq], 1)
    up = tf.concat([xw[:, -1], icq], 1)
    K, roles = int(b["K"]), tuple(int(r) for r in b["roles"])
    cond, op = tf.constant(b["cond"]), tf.constant(b["op_multihot"])
    fid, sdb, fb = tf.constant(b["fam_id"]), tf.constant(b["sdf_box"]), tf.constant(b["fbmask"])
    bx, dims = box_of(K, fam), tuple(b.get("dims", ()))
    dense2d = bool(b["dense2d"]); _g = (K == 2 and dense2d); Nn = int(cn.shape[1])
    step = _ar_step_fn((fam, B), K, roles, dims, dense2d, bx)
    win, fmask = xw, fm
    prev_grid = tf.reshape(xw[:, -1], [B, *dims, NUM_SLOTS]) if _g else tf.zeros([1])
    steps = []
    for _ in range(KF):
        try:
            pa = step(cn, win, fmask, gn, cond, op, qall, gall, up, prev_grid, fid, sdb, fb)
        except Exception:                                    # trace failure -> eager fallback for this family
            if not SCORE_TFF or not steps:
                print("    [ar-batch] tf.function failed, eager fallback:", traceback.format_exc().splitlines()[-1],
                      flush=True)
            _STEP_FN[(fam, B)] = step = _STEP_FN[(fam, B)].python_function if hasattr(_STEP_FN[(fam, B)],
                                                                                    "python_function") else step
            pa = step(cn, win, fmask, gn, cond, op, qall, gall, up, prev_grid, fid, sdb, fb)
        up = pa; pn = pa[:, :Nn]
        win = tf.concat([win[:, 1:], pn[:, None]], 1)
        fmask = tf.concat([fmask[:, 1:], tf.ones_like(fmask[:, :1])], 1)
        if _g:
            prev_grid = tf.reshape(pn, [B, *dims, NUM_SLOTS])
        steps.append(pa)
    pred = tf.stack(steps, 1).numpy()                                                   # (B,KF,N+Q,S)
    return {i0 + i: (pred[i:i + 1], {}) for i in range(B)}


rows = {}
pooled_f, pooled_t = [], []
# REVISION-R1 10-27 (downstream prediction panels): DUMP=<prefix> -> <prefix>_<family>.npz with the physical ground truth and predictions on
# the node set (= every grid cell for 2D grids up to 128^2, C-order of dims) of the first DUMP_N trajectories: direct
# (single-shot / steady) at every lead and, with HOP, the hop-AR rollout at its leads; per-trajectory errors as scored
DUMP = os.environ.get("DUMP", ""); DUMP_N = int(os.environ.get("DUMP_N", "16"))
for fam in FAMS:
    _dump = []
    rng = np.random.default_rng(1234)
    fin, tra = [], []
    fin_h, tra_h, fin_e, tra_e = [], [], [], []
    pap_d, pap_h = [], []
    _gt = []                                                     # REVISION-R1 10-30: lead-1 |y| norm per trajectory (pairing check)
    try:
        it = iter(loader.units_for(fam, SPLIT))
        # REVISION-R1 10-24: SCORE_BATCH > 1 (single-shot decode, transient families): the examples are drawn first (same
        # rng order as the per-sample path), and the single-shot call and the whole hop-AR rollout run SCORE_BATCH
        # trajectories per model call; the per-sample metric code below then reads the cached predictions.
        _EX, _CACHE = None, {}
        if SCORE_BATCH > 1 and HOP == 0 or SCORE_BATCH > 1 and DECODE == "ss":   # REVISION-R1 10-29: + AR batch
            _EX = []
            for _ in range(NEV):
                try:
                    _s = next(it)
                except StopIteration:
                    break
                _EX.append(make_example_ar(_s, T_IN, KF, 4, 2048, 8192, rng, box_dims=box_of(int(_s["K"]), fam)))
            if _EX and bool(_EX[0]["steady"]):
                pass                                              # steady: per-sample relaxation path below
            else:
                _ss_mode = DECODE == "ss" or (DECODE == "auto" and _EX[0].get("mode", "ar") == "ss")
                _bp = _batched_predict if _ss_mode else _batched_predict_ar
                for i0 in range(0, len(_EX), SCORE_BATCH):
                    _CACHE.update(_bp(_EX[i0:i0 + SCORE_BATCH], i0, fam))
        for n in range(NEV):
            if _EX is not None:
                if n >= len(_EX):
                    break
                ex = _EX[n]
            else:
                try:
                    s = next(it)
                except StopIteration:
                    break
                ex = make_example_ar(s, T_IN, KF, 4, 2048, 8192, rng,
                                     box_dims=box_of(int(s["K"]), fam))
            b = stack_batch([ex])
            cn, xw, fm, gn = (tf.constant(b[k]) for k in
                              ("coords_node", "x_win", "fmask", "geom_node"))
            icq, cq, gq = (tf.constant(b[k]) for k in ("ic_q", "coords_q", "geom_q"))
            qall = tf.concat([cn, cq], 1)
            gall = tf.concat([gn, gq], 1)
            u_prev = tf.concat([xw[:, -1], icq], 1)
            K, roles = int(b["K"]), tuple(int(r) for r in b["roles"])
            cond, op = tf.constant(b["cond"]), tf.constant(b["op_multihot"])
            fid, sdb = tf.constant(b["fam_id"]), tf.constant(b["sdf_box"])
            bx, dims = box_of(K, fam), tuple(b.get("dims", ()))
            # DECODE IN THE MODE THE FAMILY WAS TRAINED IN. AR_MODE defaults to "ar", so every CE
            # and NS family rolls out autoregressively; decoding those single-shot is an inference
            # mode the weights never saw and reads ~3x worse than the training-time eval.
            if bool(b["steady"]):                                 # REVISION-R1 10-23: steady task = MOTION relaxation
                Nn = cn.shape[1]; dense2d = bool(b["dense2d"]); up = u_prev       # (k_relax steps, as in training;
                _g = (K == 2 and dense2d)                                          #  the last step is the solution)
                # REVISION-R1 10-24: the relaxation feeds back exactly as v3/train.py rollout_loss does: each step's
                # prediction is appended to the window (fmask 1) and becomes prev_grid (the previous one prev2_grid).
                # The first version kept the zero window / zero grid for every step (Poisson-Gauss test 218% vs
                # val track 8.4%).
                win_, fm_ = xw, fm
                prev_grid = tf.reshape(xw[:, -1], [1, *dims, NUM_SLOTS]) if _g else None
                prev2_grid = tf.reshape(xw[:, -2], [1, *dims, NUM_SLOTS]) if (_g and int(xw.shape[1]) >= 2) else None
                for _ in range(int(os.environ.get("K_RELAX", "4"))):
                    up = model.step(cn, win_, fm_, gn, K, roles, cond, op, qall, gall, up, steady=True,
                                    prev_grid=prev_grid, grid_dims=dims if dense2d else None, fam_id=fid,
                                    sdf_box=sdb, box_dims=bx, fbmask=tf.constant(b["fbmask"]), prev2_grid=prev2_grid)
                    pn_ = up[:, :Nn]
                    win_ = tf.concat([win_[:, 1:], pn_[:, None]], 1)
                    fm_ = tf.concat([fm_[:, 1:], tf.ones_like(fm_[:, :1])], 1)
                    if _g:
                        prev2_grid, prev_grid = prev_grid, tf.reshape(pn_, [1, *dims, NUM_SLOTS])
                pred = up.numpy()[:, None]                                       # (B,1,N+Q,S)
            elif n in _CACHE:
                pred = _CACHE[n][0]
            elif DECODE == "ss" or (DECODE == "auto" and b.get("mode", "ar") == "ss"):
                _d2 = bool(b["dense2d"]) and K == 2                  # MOTION_SS_WARP needs the IC grid
                pred = model.step_ss(cn, xw, fm, gn, K, roles, cond, op, qall, gall, u_prev,
                                     kf=KF, fam_id=fid, sdf_box=sdb, box_dims=bx,
                                     prev_grid=tf.reshape(xw[:, -1], [1, *dims, NUM_SLOTS]) if _d2 else None,
                                     grid_dims=dims if _d2 else None).numpy()
            else:
                win, fmask, up = xw, fm, u_prev
                Nn = cn.shape[1]
                dense2d = bool(b["dense2d"])
                prev_grid = (tf.reshape(xw[:, -1], [1, *dims, NUM_SLOTS])
                             if (K == 2 and dense2d) else None)
                steps = []
                for _ in range(KF):
                    pa = model.step(cn, win, fmask, gn, K, roles, cond, op, qall, gall, up,
                                    steady=False, prev_grid=prev_grid,
                                    grid_dims=dims if dense2d else None, fam_id=fid,
                                    sdf_box=sdb, box_dims=bx,
                                    fbmask=tf.constant(b["fbmask"]))
                    up = pa
                    pn = pa[:, :Nn]
                    win = tf.concat([win[:, 1:], pn[:, None]], 1)
                    fmask = tf.concat([fmask[:, 1:], tf.ones_like(fmask[:, :1])], 1)
                    if K == 2 and dense2d:
                        prev_grid = tf.reshape(pn, [1, *dims, NUM_SLOTS])
                    steps.append(pa)
                pred = tf.stack(steps, 1).numpy()               # (B,KF,N+Q,S)
            y = np.concatenate([b["y_node"], b["y_q"]], 2)        # (1,KF,N+Q,S)
            if bool(b["steady"]):
                y = y[:, :1]
            if n == 0:
                anc = np.concatenate([b["x_win"][:, -1], b["ic_q"]], 1)   # (1,N+Q,S)
                act = np.nonzero(b["cmask"][0])[0]
                print(f"    [diag] mode={b.get('mode')} dense2d={b['dense2d']} "
                      f"active slots={list(act)} pred{pred.shape} y{y.shape}", flush=True)
                print(f"    [diag] fam_id={int(b['fam_id'][0])} "
                      f"op_multihot={np.nonzero(b['op_multihot'][0])[0].tolist()} "
                      f"fbmask(active)={[float(b['fbmask'][0][c]) for c in act]} "
                      f"cond_rms={float(np.sqrt((b['cond'][0]**2).mean())):.4e} "
                      f"roles={list(b['roles'])[:4]}", flush=True)
                for c in act:
                    pr, gt, an = pred[0, :, :, c], y[0, :, :, c], anc[0, :, c]
                    print(f"       slot{c}: rms(pred)={np.sqrt((pr**2).mean()):.4e} "
                          f"rms(y)={np.sqrt((gt**2).mean()):.4e} "
                          f"rms(anchor)={np.sqrt((an**2).mean()):.4e} "
                          f"rms(pred-anchor)={np.sqrt(((pr-an[None])**2).mean()):.4e}",
                          flush=True)
            if n == 0:
                print(f"    [shapes] pred={pred.shape} y={y.shape} scale={b['scale'].shape} "
                      f"cmask={b['cmask'].shape} tsm={b['tstep_mask'].shape}", flush=True)
            sc = b["scale"][:, None, None, :]                     # (B,1,1,S) physical units
            cm = b["cmask"][:, None, None, :]                     # active slots only
            if os.environ.get("SCORE_SLOTS"):                     # REVISION-R1 10-22: score a slot subset (Poseidon comparison:
                _ss = np.zeros(NUM_SLOTS, np.float32)             #  CE rho,u,v,p = slots 3,0,1,4; NS u,v = 0,1)
                _ss[[int(x) for x in os.environ["SCORE_SLOTS"].split(",")]] = 1.0
                cm = cm * _ss[None, None, None, :]
            tsm = b["tstep_mask"][0] if not bool(b["steady"]) else np.ones(1, np.float32)   # (KF,) valid leads
            if os.environ.get("SCORE_EXCLUDE_GEOM") == "1":           # REVISION-R1 10-23 (SE-AF): drop the airfoil interior
                gpt = np.concatenate([b["geom_node"], b["geom_q"]], 1)[..., 0]   # (1,N+Q) mask/std > 0 inside the body
                cm = cm * (gpt <= 0)[:, None, :, None]
            num = np.sum(np.abs(pred - y) * sc * cm, axis=(2, 3))[0]     # (KF,)
            sh = (b["shift"][:, None, None, :] if "shift" in b else 0.0)   # MOTION_INPUT_ZSCORE: add the mean back
            den = np.sum(np.abs(y * sc + sh) * cm, axis=(2, 3))[0] + 1e-8
            r = num / den
            val = np.where(tsm > 0)[0]
            if not len(val):
                continue
            fin.append(r[val[-1]])
            tra.append(float(np.mean(r[val])))
            _gt.append(float(den[val[0]]))
            _ptm = np.ones(pred.shape[2])
            if os.environ.get("SCORE_EXCLUDE_GEOM") == "1":
                _ptm = (np.concatenate([b["geom_node"], b["geom_q"]], 1)[0, :, 0] <= 0).astype(float)
            if PAPER_FUNCS:
                _L = 1 if bool(b["steady"]) else PAPER_LEAD
                _scv = b["scale"].reshape(-1); _shv = b["shift"].reshape(-1) if "shift" in b else np.zeros_like(_scv)
                if _L <= pred.shape[1] and (bool(b["steady"]) or tsm[_L - 1] > 0):
                    pap_d.append(_paper_err(pred[0, _L - 1] * _scv + _shv, y[0, _L - 1] * _scv + _shv, _ptm))
            if HOP > 0 and pred.ndim == 4:                       # hop-AR from the same single-shot model
                scv = b["scale"].reshape(-1); shv = (b["shift"].reshape(-1) if "shift" in b else np.zeros_like(scv))
                cmv = b["cmask"].reshape(-1) > 0; Nn = int(cn.shape[1])
                leads = list(range(HOP, KF + 1, HOP)); P = {}
                cur = pred[0] * scv + shv                                 # (KF,N+Q,S) physical
                X = cur[HOP - 1]; P[HOP] = X
                _d2 = bool(b["dense2d"]) and K == 2
                if n in _CACHE:
                    P = _CACHE[n][1]
                for L_ in (leads[1:] if n not in _CACHE else []):
                    sd_ = X[:Nn].std(0); mu_ = X[:Nn].mean(0)
                    sc_ = np.where((sd_ > 1e-6) & cmv, sd_, 1.0); sh_ = np.where(cmv, mu_, 0.0) if _ZS else np.zeros_like(sc_)
                    if _DA.GLOBAL_NORM_ON:                       # fixed statistics: the next IC keeps the same affine map
                        sc_, sh_ = scv, shv
                    Xn = ((X - sh_) / sc_).astype(np.float32)
                    xw_ = tf.constant(Xn[None, None, :Nn]); up_ = tf.constant(Xn[None])
                    pg_ = tf.reshape(xw_[:, -1], [1, *dims, NUM_SLOTS]) if _d2 else None
                    # REVISION-R1 10-24: decode only lead HOP (step_ss leads=, same time map; was all KF leads, of
                    # which HOP-1 alone was used: ~20x the decoder work per hop). HOP_FULL=1 restores the old call.
                    _hl = None if os.environ.get("HOP_FULL") == "1" else [HOP]
                    pr_ = model.step_ss(cn, xw_, fm, gn, K, roles, cond, op, qall, gall, up_, kf=KF, fam_id=fid,
                                        sdf_box=sdb, box_dims=bx, prev_grid=pg_, grid_dims=dims if _d2 else None,
                                        leads=_hl).numpy()
                    X = pr_[0, (HOP - 1) if _hl is None else 0] * sc_ + sh_; P[L_] = X
                yph = y[0] * scv + shv; cmb = cmv[None, :]
                if PAPER_FUNCS and PAPER_LEAD in P and tsm[PAPER_LEAD - 1] > 0:
                    pap_h.append(_paper_err(P[PAPER_LEAD], yph[PAPER_LEAD - 1], _ptm))
                rh = [np.sum(np.abs(P[L_] - yph[L_ - 1]) * cmb) / (np.sum(np.abs(yph[L_ - 1]) * cmb) + 1e-8) for L_ in leads]
                ok = [i for i, L_ in enumerate(leads) if tsm[L_ - 1] > 0]
                if ok:
                    fin_h.append(rh[ok[-1]]); tra_h.append(float(np.mean([rh[i] for i in ok])))
                    re_ = [r[L_ - 1] for L_ in leads]
                    fin_e.append(re_[ok[-1]]); tra_e.append(float(np.mean([re_[i] for i in ok])))
            if DUMP and n < DUMP_N:
                _sc = b["scale"].reshape(-1); _sh = b["shift"].reshape(-1) if "shift" in b else np.zeros_like(_sc)
                _Nn = int(cn.shape[1])
                _rec = dict(y=(y[0] * _sc + _sh)[:, :_Nn].astype(np.float32),
                            direct=(pred[0] * _sc + _sh)[:, :_Nn].astype(np.float32), err_direct=float(tra[-1]),
                            dims=np.array(dims), coords=np.asarray(b["coords_node"][0], np.float32))
                if HOP > 0 and pred.ndim == 4 and 'P' in dir() and P:
                    _rec["ar"] = np.stack([P[L_][:_Nn] for L_ in leads]).astype(np.float32)
                    _rec["ar_leads"] = np.array(leads)
                    _rec["err_ar"] = float(tra_h[-1]) if tra_h else float("nan")
                _dump.append(_rec)
    except Exception:
        traceback.print_exc()
    if DUMP and _dump:
        _out = {f"{k}_{i}": v for i, r_ in enumerate(_dump) for k, v in r_.items()}
        np.savez_compressed(f"{DUMP}_{fam}.npz", n=len(_dump), **_out)
        print(f"[dump] {DUMP}_{fam}.npz: {len(_dump)} trajectories", flush=True)
    if os.environ.get("PER_TRAJ_OUT") and fin:                   # REVISION-R1 10-30: per-trajectory errors (paired tests)
        import json
        json.dump({"family": fam, "split": SPLIT, "n": len(tra), "hop": HOP, "ckpt": os.environ.get("CKPT", ""),
                   "direct_fulltraj": tra, "ar_fulltraj": tra_h, "gt_norm_lead1": _gt},
                  open(os.environ["PER_TRAJ_OUT"], "w"))
        print(f"[per-traj] {os.environ['PER_TRAJ_OUT']}: {len(tra)} trajectories", flush=True)
    if fin:
        rows[fam] = (float(np.median(fin)), float(np.median(tra)), len(fin))
        pooled_f += fin
        pooled_t += tra
        print(f"  {fam}: @final={rows[fam][0]*100:.2f}%  full-traj={rows[fam][1]*100:.2f}%  "
              f"(n={len(fin)})", flush=True)
        if pap_d:
            print(f"  {fam} [PAPER metric, lead {PAPER_LEAD}, funcs {PAPER_FUNCS}] direct: median {np.median(pap_d)*100:.3f}% (n={len(pap_d)})"
                  + (f" | hop-{HOP} AR: median {np.median(pap_h)*100:.3f}% (n={len(pap_h)})" if pap_h else ""), flush=True)
        if fin_h:
            print(f"  {fam} [hop-{HOP} AR, leads {HOP}..{KF}]: @final={np.median(fin_h)*100:.2f}%  full-traj={np.median(tra_h)*100:.2f}%"
                  f"  | direct on the same leads: @final={np.median(fin_e)*100:.2f}%  full-traj={np.median(tra_e)*100:.2f}%  (n={len(fin_h)})", flush=True)

if pooled_f:
    print(f"[EVAL-L1] MEDIAN rel-L1 @final (physical) overall={np.median(pooled_f)*100:.3f}%  | "
          + "  ".join(f"{k}={v[0]*100:.2f}%" for k, v in rows.items()), flush=True)
    print(f"[EVAL-L1] MEDIAN rel-L1 FULL-TRAJ (physical) overall={np.median(pooled_t)*100:.3f}%  | "
          + "  ".join(f"{k}={v[1]*100:.2f}%" for k, v in rows.items()), flush=True)
print("L1_SCORE_DONE", flush=True)
