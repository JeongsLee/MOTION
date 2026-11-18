"""Equivariance diagnostics for the foundationv2 models (revision R1; Fig. 4b protocol, A01-corrected).

For every dense 128^2 periodic 2-D family (all grid nodes are predicted, so whole fields can be
transformed and compared), each trajectory is rolled out from its initial window under a group
element g applied to the INPUT (and target), and the prediction is mapped back with g^-1:
  defect_g = || g^-1 f(g x) - f(x) || / || f(x) ||        (consistency, relative L2 over space+channels)
  err_g    = || g^-1 f(g x) - y || / || y ||              (accuracy of the transformed replica)
Group elements: D4 (8: id, fr, fc, r180, d1(transpose), r90, r270, d2) x periodic shifts (0, 1, 8 cells).
Ensembles (mean of the back-mapped replicas): K4 = {id, fr, fc, r180}, K8 = D4, T3 = {0, 1, 8} shifts,
K8xT3 = all 24. Reported per frame, physical units, scored channels only (cmask).

Vector slots 0/1 (Vx, Vy): a flip negates the component along the flipped axis, a transpose swaps
them. Which slot is aligned with array axis 1 is CHECKED at start-up on incom_ns ground truth
(divergence of the two pairings; the wrong pairing has O(1) divergence) and the run aborts if the
check is ambiguous (A01 lesson: padded/vector channel conventions must be verified, not assumed).

  python eval_equiv.py --ckpt C --out OUT.npz [--split test] [--n 32] [--fams default]
         [--members d4xshift|d4|shift] [--batch 8] [--xla 1] [--bf16 0]
"""
import argparse, json, os, time, zlib
import itertools
import numpy as np
import tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--fams", default="default")
ap.add_argument("--members", default="d4xshift")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--d", type=int, default=384)
ap.add_argument("--depth", type=int, default=8)
ap.add_argument("--d_w", type=int, default=16)
ap.add_argument("--n_p", type=int, default=16)
ap.add_argument("--xla", type=int, default=1)
ap.add_argument("--bf16", type=int, default=0)
ap.add_argument("--dump", type=int, default=0, help="REVISION-R1: store the identity predictions/targets (physical) of the first N trajectories")
a = ap.parse_args()
if a.bf16:
    tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")

from core.pretrain import load_ckpt
from data import loader
from data.registry import FAMILIES, NUM_SLOTS, SEMANTIC_SLOTS
from v3.data_ar import make_example_ar, stack_batch
from v3.model import V3Model
from v3.train import BOX_TABLE_2D, _grid

# periodic, no gravity / external direction, dense 128^2, transient
DEFAULT_FAMS = ["shallow_water", "diff_react", "com_ns", "incom_ns", "CE-RP", "CE-CRP", "CE-KH",
                "CE-Gauss", "CE-RM", "NS-Sines", "NS-Gauss", "ACE"]
GD = {"id": (0, 0, 0), "fr": (0, 1, 0), "fc": (0, 0, 1), "r180": (0, 1, 1),
      "d1": (1, 0, 0), "r90": (1, 1, 0), "r270": (1, 0, 1), "d2": (1, 1, 1)}
K4 = ("id", "fr", "fc", "r180")
# REVISION-R1 (10-21): anisotropic joint-benchmark families get only their admitted mirror (published MOTION-NS
# convention, MOTION_NCS_share fig4 jobscripts): CFDBench = spanwise ROW mirror with NO sign change (both prebuilt
# flow slots are u-like scalars, cfd_cond_sym2.py); PDEArena = row mirror with the vector sign (eps_base_final "fr").
# Ensemble "K2" = mean of identity and that mirror.
GD_D4 = tuple(GD)                     # the dihedral group proper (8); fr_ns is NOT a D4 element
GD["fr_ns"] = (0, 1, 0)
MEMBERS_OVERRIDE = {"cfdbench": [("id", 0), ("fr_ns", 0)], "pdearena_ns": [("id", 0), ("fr", 0)],
                    "pdearena_uncond": [("id", 0), ("fr", 0)]}
# Translation classification already fixed in the paper (SI Note 5 / Table S6, seam criterion applied
# to the stored trajectories); used verbatim for these families so the ablation numbers are on the
# same footing. Other families fall back to the seam-ratio rule (<= 3).
PAPER_SHIFT_CLASS = {"shallow_water": True, "com_ns": True, "incom_ns": True, "ACE": False,
                     "pdearena_ns": False, "Poisson-Gauss": False}
SHIFTS = (0, 1, 8)

# REVISION-R1 (10-22): MOTION-IVP-S (single-shot from one IC frame, 20 leads): EQUIV_T_IN=1 EQUIV_KF=20; defaults = the
# 10-in / 10-out NS-S protocol. MOTION_GLOBAL_NORM=1: the family statistics are computed from the train split as in
# train.py / score_ivp_l1.py. Single-shot decode passes the IC grid so MOTION_SS_WARP models use their warp.
# EQUIV_SCORE_SLOTS="3,0,1,4": score only these slots (Poseidon channels rho, u, v, p; drops CE energy). Ensembles and
# defects use the same mask; the model still predicts every slot.
SCORE_SLOTS = [int(v) for v in os.environ.get("EQUIV_SCORE_SLOTS", "").split(",") if v]
EQ_T_IN = int(os.environ.get("EQUIV_T_IN", "10")); EQ_KF = int(os.environ.get("EQUIV_KF", "10"))
model = V3Model(S=NUM_SLOTS, d=a.d, depth=a.depth, d_w=a.d_w, n_p=a.n_p, d_cond=128,
                dims2=(64, 64), dims3=(32, 32, 32), t_in=EQ_T_IN, transport=True,
                n_fams=len(FAMILIES), n_semantic=SEMANTIC_SLOTS,
                gate_cond=1, dict_decode=0, fam_head=1)
model.warm_build()
V = model.trainable_variables
load_ckpt(V, a.ckpt, d_w=a.d_w, names=[v.name for v in V])
print(f"[equiv] {a.ckpt}: {sum(int(np.prod(v.shape)) for v in V)/1e6:.1f}M params | env "
      f"{ {k: v for k, v in os.environ.items() if k.startswith('MOTION_')} }", flush=True)
KEYS = ("coords_node", "x_win", "fmask", "geom_node", "y_node", "ic_q", "coords_q",
        "geom_q", "y_q", "tstep_mask", "cmask", "fbmask", "op_multihot", "cond", "fam_id", "sdf_box")


# ---------------------------------------------------------------- group action on (T, H, W, S)
VA1 = None          # slot aligned with array axis 1 for the CURRENT family (set per family)
VA1_BY_FMT = {}     # adapter format -> pairing, from a divergence-free family of that format
DIVFREE_REF = {"grid_npy": "incom_ns", "grid_nchw": "NS-Sines"}
# Families whose own continuity test is ambiguous AND whose same-layout fallback proved wrong.
# CE-RM (m1f, 2026-10-16): with VA1=1 (CE-Gauss layout) every single-axis reflection gave replica
# error 23.8 vs 7.6 (identity) while r180/transposes stayed at 7.6 -- the wrong-pairing signature
# (pairing signature 6.16). So CE-RM stores the NS convention despite its CE channel map.
PAIRING_OVERRIDE = {"CE-RM": 0}


def _flip(x, ax):
    x = np.flip(x, axis=ax).copy()
    s = VA1 if ax == 1 else 1 - VA1
    x[..., s] = -x[..., s]
    return x


def _transpose(x):
    x = np.swapaxes(x, 1, 2).copy()
    x[..., [0, 1]] = x[..., [1, 0]]
    return x


def g_fwd(x, g, s):
    if g == "fr_ns":                      # scalar row mirror (self-inverse)
        return np.flip(x, axis=1).copy()
    t, fr, fc = GD[g]
    if s: x = np.roll(x, (s, s), axis=(1, 2))
    if fr: x = _flip(x, 1)
    if fc: x = _flip(x, 2)
    if t: x = _transpose(x)
    return x


def g_inv(x, g, s):                      # exact inverse of g_fwd (reverse order, each op self-inverse)
    if g == "fr_ns":
        return np.flip(x, axis=1).copy()
    t, fr, fc = GD[g]
    if t: x = _transpose(x)
    if fc: x = _flip(x, 2)
    if fr: x = _flip(x, 1)
    if s: x = np.roll(x, (-s, -s), axis=(1, 2))
    return x


def check_vector_axes():
    """Divergence test (div u = 0) per adapter format: the array layout can differ between
    adapters (e.g. PDEBench npy vs Poseidon nchw), so each format gets its own pairing."""
    global VA1
    res = {}
    d = lambda u, ax: np.roll(u, -1, ax) - np.roll(u, 1, ax)
    for fmt, ref in DIVFREE_REF.items():
        # REVISION-R1 (10-21): test up to 8 units and keep the first DECISIVE one (the first MTF-split incom_ns test
        # unit is forced / nearly static and gave 0.90 vs 1.33); if none is decisive, MOTION_EQUIV_VA1="fmt:VA1,..."
        # (from earlier decisive runs: grid_npy 0.0048 vs 1.5252 -> 0) is used, otherwise abort as before.
        _ov = dict(kv.split(":") for kv in os.environ.get("MOTION_EQUIV_VA1", "").split(",") if ":" in kv)
        divA = divB = None
        # REVISION-R1 11-01: the reference family may lack the requested split (heldout = rows held out by both
        # models exists only for the six Poseidon families); this format-level check then reads its test split
        # (the pairing of each evaluated family is determined separately from its own data).
        try:
            _units = list(itertools.islice(loader.units_for(ref, a.split), 8))
        except FileNotFoundError:
            print(f"[equiv] vector-axis check {fmt}: {ref} has no '{a.split}' split; using test", flush=True)
            _units = list(itertools.islice(loader.units_for(ref, "test"), 8))
        for s in _units:
            f = np.asarray(s["fields"], np.float64)
            f = f.reshape(f.shape[0], *[int(v) for v in s["dims"]], f.shape[-1])[:4]
            u0, u1 = f[..., 0], f[..., 1]
            nrm = np.sqrt((d(u0, 1) ** 2).mean() + (d(u1, 2) ** 2).mean())
            divA = np.sqrt(((d(u0, 1) + d(u1, 2)) ** 2).mean()) / nrm      # slot0 <-> axis1
            divB = np.sqrt(((d(u1, 1) + d(u0, 2)) ** 2).mean()) / nrm      # slot1 <-> axis1
            print(f"[equiv] vector-axis check {fmt} ({ref}): rel div slot0~axis1 {divA:.4f} | slot1~axis1 {divB:.4f}",
                  flush=True)
            if min(divA, divB) < 0.3 * max(divA, divB):
                break
        if not min(divA, divB) < 0.3 * max(divA, divB):
            assert fmt in _ov, f"ambiguous vector-axis pairing for {fmt}; abort"
            print(f"[equiv] {fmt}: no decisive unit; using MOTION_EQUIV_VA1 override {_ov[fmt]}", flush=True)
            divA, divB = (0.0, 1.0) if int(_ov[fmt]) == 0 else (1.0, 0.0)
        VA1_BY_FMT[fmt] = 0 if divA < divB else 1
        VA1 = VA1_BY_FMT[fmt]
        for g in GD:                                   # self-test: g_inv(g_fwd(x)) == x
            for sh in SHIFTS:
                assert np.array_equal(g_inv(g_fwd(f, g, sh), g, sh), f), (g, sh)
        res[fmt] = {"ref": ref, "divA": divA, "divB": divB, "VA1": VA1_BY_FMT[fmt]}
    return res


def family_pairing(sample, dims):
    """Per-FAMILY slot<->axis pairing from the family's own ground truth (the layout can differ between
    families even within one adapter: CE-RP showed single-flip-only defects of 41% under the NS-Sines
    pairing). Compressible (density slot 3 scored): continuity residual dt(rho) + div(rho u), with the
    unknown dt scale fitted by least squares. Incompressible (velocity only): div u.
    Returns (VA1 or None if no vector slots, diagnostics)."""
    cm = np.asarray(sample["cmask"]).astype(bool)
    if not (cm[0] and cm[1]):
        return None, {"vector": False}
    f = np.asarray(sample["fields"], np.float64)
    f = f.reshape(f.shape[0], *dims, f.shape[-1])
    d = lambda u, ax: np.roll(u, -1, ax) - np.roll(u, 1, ax)
    out = {}
    for va, lab in ((0, "slot0~axis1"), (1, "slot1~axis1")):
        ua, ub = (0, 1) if va == 0 else (1, 0)              # ua along axis 1, ub along axis 2
        if cm[3]:                                           # compressible: continuity
            r0, r1 = f[0, ..., 3], f[1, ..., 3]
            rb = 0.5 * (f[0] + f[1])
            D = d(rb[..., 3] * rb[..., ua], 0) + d(rb[..., 3] * rb[..., ub], 1)
            dr = (r1 - r0)
            al = -(dr * D).sum() / ((D * D).sum() + 1e-30)
            out[lab] = float(np.sqrt(((dr + al * D) ** 2).sum() / ((dr ** 2).sum() + 1e-30)))
        else:                                               # incompressible: divergence
            ff = f[:4]
            nrm = np.sqrt((d(ff[..., 0], 1) ** 2).mean() + (d(ff[..., 1], 2) ** 2).mean()) + 1e-30
            out[lab] = float(np.sqrt(((d(ff[..., ua], 1) + d(ff[..., ub], 2)) ** 2).mean()) / nrm)
    out["test"] = "continuity" if cm[3] else "divergence"
    a_, b_ = out["slot0~axis1"], out["slot1~axis1"]
    if min(a_, b_) > 0.7 * max(a_, b_):
        out["ambiguous"] = True
        return None, out
    return (0 if a_ < b_ else 1), out


# ---------------------------------------------------------------- model rollouts returning node predictions
def tensors(b):
    return tuple(tf.constant(b[k]) for k in KEYS)


def statics(b):
    gd = tuple(b.get("dims", ())) or None
    return (int(b["K"]), bool(b["steady"]), bool(b["dense2d"]),
            tuple(int(r) for r in b["roles"]), int(b["k_seg"]), gd,
            tuple(int(v) for v in b["box_dims"]), bool(b["surf"]))


def pred_ar(ts, K, steady, dense2d, roles, k_seg, grid_dims, box_dims, surf):
    cn, xw, fm, gn, yn, icq, cq, gq, yq, tsm, cm, fbm, op, cond, fid, sdb = ts
    Nn = tf.shape(cn)[1]
    qall = tf.concat([cn, cq], 1); gall = tf.concat([gn, gq], 1)
    u_prev = tf.concat([xw[:, -1], icq], 1)
    win, fmask = xw, fm
    prev_grid = _grid(xw[:, -1], grid_dims, model.S)
    prev2_grid = _grid(xw[:, -2], grid_dims, model.S) if int(xw.shape[1]) >= 2 else None
    outs = []
    for k in range(k_seg):
        pred = tf.cast(model.step(cn, win, fmask, gn, K, roles, cond, op, qall, gall, u_prev,
                                  steady=steady, prev_grid=prev_grid, grid_dims=grid_dims,
                                  fam_id=fid, sdf_box=sdb, box_dims=box_dims, fbmask=fbm, surf=surf,
                                  prev2_grid=prev2_grid), tf.float32)
        outs.append(pred[:, :Nn])
        u_prev = pred
        pred_n = u_prev[:, :Nn]
        win = tf.concat([win[:, 1:], pred_n[:, None]], 1)
        fmask = tf.concat([fmask[:, 1:], tf.ones_like(fmask[:, :1])], 1)
        prev2_grid, prev_grid = prev_grid, _grid(pred_n, grid_dims, model.S)
    return tf.stack(outs, 1)                                       # (B, k, N, S)


def pred_ss(ts, K, steady, dense2d, roles, k_seg, grid_dims, box_dims, surf):
    cn, xw, fm, gn, yn, icq, cq, gq, yq, tsm, cm, fbm, op, cond, fid, sdb = ts
    Nn = tf.shape(cn)[1]
    qall = tf.concat([cn, cq], 1); gall = tf.concat([gn, gq], 1)
    u_prev = tf.concat([xw[:, -1], icq], 1)
    _w = bool(getattr(model, "ss_warp", False)) and dense2d and K == 2
    p = model.step_ss(cn, xw, fm, gn, K, roles, cond, op, qall, gall, u_prev,
                      kf=k_seg, fam_id=fid, sdf_box=sdb, box_dims=box_dims,
                      prev_grid=_grid(xw[:, -1], grid_dims, model.S) if _w else None,
                      grid_dims=grid_dims if _w else None)
    return tf.cast(p, tf.float32)[:, :, :Nn]


def pred_rh(ts, K, steady, dense2d, roles, k_seg, grid_dims, box_dims, surf):
    """REVISION-R1 10-27: receding-horizon hop-1 rollout of MOTION-IVP option C (= score_ivp_l1.py HOP=1 under
    MOTION_GLOBAL_NORM): each call decodes lead 1 of a k_seg-lead single-shot step, the prediction is appended to the
    observed window (fmask 1) and becomes the next initial condition."""
    cn, xw, fm, gn, yn, icq, cq, gq, yq, tsm, cm, fbm, op, cond, fid, sdb = ts
    Nn = tf.shape(cn)[1]
    qall = tf.concat([cn, cq], 1); gall = tf.concat([gn, gq], 1)
    u_prev = tf.concat([xw[:, -1], icq], 1)
    _w = bool(getattr(model, "ss_warp", False)) and dense2d and K == 2
    win, fmask, outs = xw, fm, []
    for k in range(k_seg):
        p = tf.cast(model.step_ss(cn, win, fmask, gn, K, roles, cond, op, qall, gall, u_prev,
                                  kf=k_seg, fam_id=fid, sdf_box=sdb, box_dims=box_dims,
                                  prev_grid=_grid(win[:, -1], grid_dims, model.S) if _w else None,
                                  grid_dims=grid_dims if _w else None, leads=[1]), tf.float32)[:, 0]
        outs.append(p[:, :Nn])
        u_prev = p
        win = tf.concat([win[:, 1:], p[:, None, :Nn]], 1)
        fmask = tf.concat([fmask[:, 1:], tf.ones_like(fmask[:, :1])], 1)
    return tf.stack(outs, 1)


# REVISION-R1 (10-22; R2-3 / R3-3 cost table): EQUIV_TIME_ONLY=1 times the identity rollout of every family twice
# (the first pass includes tracing / XLA compilation) and reports seconds per trajectory of the model forward
# (example building excluded) and the peak GPU memory, then skips the equivalence members.
T_FWD = [0.0]
TIME_ONLY = os.environ.get("EQUIV_TIME_ONLY", "0") == "1"
DUMP_MEMBERS = {m for m in os.environ.get("EQUIV_DUMP_MEMBERS", "").split(",") if m}
DUMP_LEAD = int(os.environ.get("EQUIV_DUMP_LEAD", "-1"))   # REVISION-R1 11-06: frame index of the dumps (default last)   # e.g. fr,d1,id_s8 (default: none)
from concurrent.futures import ThreadPoolExecutor
_POOL = ThreadPoolExecutor(8)                              # CPU-side example building (numpy releases the GIL)
f_ar = tf.function(pred_ar, reduce_retracing=True, jit_compile=bool(a.xla))
f_ss = tf.function(pred_ss, reduce_retracing=True, jit_compile=bool(a.xla))
f_rh = tf.function(pred_rh, reduce_retracing=True, jit_compile=bool(a.xla))


def seed_for(fam, idx):
    return zlib.crc32(f"{fam}:{idx}:0".encode())


def rollout(units, fam, g, sh, dims):
    """Predictions for every trajectory under member (g, sh), mapped back to the original frame.
    Returns physical-unit back-mapped predictions P (n, k, H, W, S), targets Y (same), mask tstep, cmask."""
    def _one(idx_s):
        idx, s = idx_s
        s2 = dict(s)
        f = np.asarray(s["fields"])
        f = f.reshape(f.shape[0], *dims, f.shape[-1])
        s2["fields"] = g_fwd(f, g, sh) if (g != "id" or sh) else f
        if g in ("fr", "fr_ns") and not sh:                 # REVISION-R1: row-mirror the other spatial arrays
            for k_, v_ in s.items():                        # (geometry / mask) with the state (no sign change)
                if k_ != "fields" and isinstance(v_, np.ndarray) and v_.ndim >= 3 and tuple(v_.shape[-3:-1]) == tuple(dims):
                    s2[k_] = np.flip(v_, axis=v_.ndim - 3).copy()
        return make_example_ar(s2, EQ_T_IN, EQ_KF, 4, 2048, 8192, np.random.default_rng(seed_for(fam, idx)),
                               box_dims=(int(os.environ.get("MOTION_BOX_UNIFORM", "0")) or int(BOX_TABLE_2D.get(fam, 64)),) * 2)   # REVISION-R1: --box_table 0 models (NS-S)
    exs = list(_POOL.map(_one, enumerate(units)))         # per-trajectory seeds -> order-independent
    P, Y, W, CM = [], [], [], []
    for i in range(0, len(exs), a.batch):
        b = stack_batch(exs[i:i + a.batch])
        assert b["dense2d"], f"{fam} is not dense2d"
        fn = f_ss if b["mode"] == "ss" else (f_rh if b["mode"] == "rh" else f_ar)   # REVISION-R1 10-27: rh = MOTION-IVP
        _t = time.time()
        p = fn(tensors(b), *statics(b)).numpy()                    # (B,k,N,S) normalized
        T_FWD[0] += time.time() - _t
        sc = np.asarray(b["scale"], np.float32).reshape(p.shape[0], 1, 1, -1)
        sh_ = (np.asarray(b["shift"], np.float32).reshape(p.shape[0], 1, 1, -1)    # REVISION-R1 z-score arm: mean back
               if os.environ.get("MOTION_INPUT_ZSCORE", "0") == "1" else 0.0)
        p = p * sc + sh_; y = np.asarray(b["y_node"], np.float32)[:, :p.shape[1]] * sc + sh_
        B, k = p.shape[:2]
        p = p.reshape(B, k, *dims, -1); y = y.reshape(B, k, *dims, -1)
        for j in range(B):
            P.append(g_inv(p[j], g, sh)); Y.append(g_inv(y[j], g, sh))
        W.append(np.asarray(b["tstep_mask"], np.float32)[:, :k]); CM.append(np.asarray(b["cmask"], np.float32))
    CMa = np.concatenate(CM, 0)
    if SCORE_SLOTS:                                       # REVISION-R1 (10-22): score a slot subset only
        _keep = np.zeros(CMa.shape[-1], np.float32); _keep[SCORE_SLOTS] = 1.0
        CMa = CMa * _keep[None]
    return np.stack(P), np.stack(Y), np.concatenate(W, 0), CMa


def rel(a_, b_, cm):                      # (n,k,H,W,S) -> (n,k), scored channels only
    m = cm[:, None, None, None, :]
    num = np.sqrt((((a_ - b_) * m) ** 2).sum(axis=(2, 3, 4)))
    den = np.sqrt(((b_ * m) ** 2).sum(axis=(2, 3, 4))) + 1e-12
    return num / den


meta = {"ckpt": a.ckpt, "split": a.split, "n": a.n, "members": a.members,
        "env": {k: v for k, v in os.environ.items() if k.startswith("MOTION_")}}
meta["vector_check"] = check_vector_axes()
fams = DEFAULT_FAMS if a.fams == "default" else a.fams.split(",")
if a.members == "d4":
    members = [(g, 0) for g in GD_D4]
elif a.members == "shift":
    members = [("id", s) for s in SHIFTS]
else:
    members = [(g, s) for s in SHIFTS for g in GD_D4]
# REVISION-R1 11-06: EQUIV_MEMBERS="id,fr,r90,id_s8" restricts the members (name = D4 element, optional _s<shift>);
# the identity must be listed first (defects and gains are relative to it)
if os.environ.get("EQUIV_MEMBERS"):
    members = [(m.split("_s")[0], int(m.split("_s")[1]) if "_s" in m else 0) for m in os.environ["EQUIV_MEMBERS"].split(",")]
    assert members[0] == ("id", 0), "EQUIV_MEMBERS must start with id"
out, summary = {}, {}
t0 = time.time()
from v3 import data_ar as _DA
for fam in fams:
    spec = FAMILIES[fam]
    if _DA.GLOBAL_NORM_ON and fam not in _DA.GLOBAL_NORM:   # REVISION-R1: same family statistics as training
        _m, _s, _n = _DA.compute_global_norm(loader.units_for(fam, "train"))
        _DA.GLOBAL_NORM[fam] = (_m, _s)
        print(f"  [{fam}] global norm (n={_n}): mean {np.round(_m, 4).tolist()} std {np.round(_s, 4).tolist()}", flush=True)
    units = list(itertools.islice(loader.units_for(fam, a.split), a.n))
    if not units:
        print(f"  [skip] {fam}: no units", flush=True); continue
    units = [dict(u, fields=np.asarray(u["fields"])) for u in units]   # read once, reuse for 24 members
    dims = tuple(int(v) for v in units[0]["dims"])
    if fam in PAIRING_OVERRIDE:                           # resolved by the single-flip signature (see below)
        VA1, pdiag = PAIRING_OVERRIDE[fam], {"override": True}
    else:
        VA1, pdiag = family_pairing(units[0], dims)
    if VA1 is None and pdiag.get("vector", True):
        # ambiguous own-data test (e.g. CE-KH: weak density change between frames): try more
        # trajectories, then fall back to a DECISIVE family with the same adapter AND channel map
        # (same storage layout; e.g. CE-KH -> CE-RP), and only then to the format default.
        for _u in units[1:4]:
            VA1, pdiag = family_pairing(_u, dims)
            if VA1 is not None:
                break
        if VA1 is None:
            key = (spec.fmt, tuple(spec.channels))
            for ref, rspec in FAMILIES.items():
                if ref == fam or (rspec.fmt, tuple(rspec.channels)) != key:
                    continue
                try:
                    ru = list(itertools.islice(loader.units_for(ref, a.split), 1))
                except Exception:
                    continue
                if not ru:
                    continue
                VA1, rd = family_pairing(ru[0], tuple(int(v) for v in ru[0]["dims"]))
                if VA1 is not None:
                    pdiag = dict(pdiag, group_ref=ref, group_ref_diag=rd)
                    print(f"  [{fam}] pairing ambiguous; using same-layout family {ref} -> VA1={VA1}", flush=True)
                    break
        if VA1 is None:
            VA1 = VA1_BY_FMT.get(spec.fmt, 0)
            print(f"  [warn] {fam}: pairing ambiguous {pdiag}; using {spec.fmt} default {VA1}", flush=True)
    if VA1 is None:
        VA1 = 0                                             # no vector slots: pairing irrelevant
    print(f"  [{fam}] vector pairing {pdiag} -> VA1={VA1}", flush=True)
    if len(dims) != 2 or dims[0] != dims[1]:
        print(f"  [skip] {fam}: dims {dims}", flush=True); continue
    extra = [k for k, v in units[0].items() if isinstance(v, np.ndarray) and k != "fields"
             and v.ndim >= 2 and tuple(v.shape[-3:-1]) == dims]
    if extra:
        print(f"  [warn] {fam}: other spatial arrays not transformed: {extra}", flush=True)
    # PERIODICITY CHECK: a shift is a symmetry of the dynamics only for periodic BCs. Seam jump
    # |f[0]-f[-1]| relative to the typical neighbour difference (~1 periodic, >>1 otherwise).
    _f = np.asarray(units[0]["fields"], np.float64)
    _f = _f.reshape(_f.shape[0], *dims, _f.shape[-1])[..., np.asarray(units[0]["cmask"]).astype(bool)]
    _nb = np.mean([np.abs(np.diff(_f, axis=ax)).mean() for ax in (1, 2)]) + 1e-12
    _seam = np.mean([np.abs(np.take(_f, 0, ax) - np.take(_f, -1, ax)).mean() for ax in (1, 2)])
    seam_ratio = float(_seam / _nb)
    # same admissibility rule as the paper (Methods "Physical-equivalence tests", SI Note 5/Table S6):
    # seam comparable to interior differences (or ~0 when the disturbance never reaches the boundary,
    # e.g. SWE) admits translation; ACE (8.8-10x) and other non-periodic fields are excluded.
    shift_ok = seam_ratio <= 3.0
    if fam in PAPER_SHIFT_CLASS:                         # paper's pre-registered classification wins
        shift_ok = PAPER_SHIFT_CLASS[fam]
    print(f"  [{fam}] seam ratio {seam_ratio:.2f} -> translation {'admitted' if shift_ok else 'NOT admissible (D4 only)'}",
          flush=True)
    if TIME_ONLY:
        T_FWD[0] = 0.0; rollout(units, fam, "id", 0, dims); t_first = T_FWD[0]
        try: tf.config.experimental.reset_memory_stats("GPU:0")
        except Exception: pass
        T_FWD[0] = 0.0; rollout(units, fam, "id", 0, dims); t_run = T_FWD[0]
        try: peak = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2**30
        except Exception: peak = float("nan")
        summary[fam] = {"n": len(units), "batch": a.batch, "s_per_traj": t_run / len(units), "first_pass_s": t_first,
                        "peak_GiB": peak}
        print(f"[time {fam:14s}] n={len(units)} batch {a.batch}: {t_run / len(units):.4f} s/traj (forward, warm) | "
              f"first pass {t_first:.1f} s | peak {peak:.2f} GiB", flush=True)
        continue
    P0, Y0, W0, CM = rollout(units, fam, "id", 0, dims)
    if a.dump:                                            # REVISION-R1: identity predictions for figure panels
        out[f"{fam}|dump_pred"] = P0[:a.dump, DUMP_LEAD].astype(np.float16); out[f"{fam}|dump_gt"] = Y0[:a.dump, DUMP_LEAD].astype(np.float16)   # last frame (EQUIV_DUMP_LEAD: frame index, 11-06)
        out[f"{fam}|dump_cmask"] = CM[:a.dump]
    fam_members = MEMBERS_OVERRIDE.get(fam, members)
    acc = {"K4": np.zeros_like(P0), "K8": np.zeros_like(P0), "T3": np.zeros_like(P0), "K8xT3": np.zeros_like(P0),
           "K2": np.zeros_like(P0)}
    cnt = {k: 0 for k in acc}
    fs = {}
    for g, sh in fam_members:
        Pg = P0 if (g == "id" and sh == 0) else rollout(units, fam, g, sh, dims)[0]
        name = g if sh == 0 else f"{g}_s{sh}"
        out[f"{fam}|{name}|defect"] = rel(Pg, P0, CM).astype(np.float32)
        out[f"{fam}|{name}|err"] = rel(Pg, Y0, CM).astype(np.float32)
        if name in DUMP_MEMBERS and a.dump:           # REVISION-R1 11-03: inverse-transformed predictions (last frame)
            out[f"{fam}|dumpm_{name}"] = Pg[:a.dump, DUMP_LEAD].astype(np.float16)
        wm = lambda e: float((e * W0).sum() / W0.sum())
        fs[name] = (100 * wm(out[f"{fam}|{name}|err"]), 100 * wm(out[f"{fam}|{name}|defect"]))
        if sh == 0 and g in K4: acc["K4"] += Pg; cnt["K4"] += 1
        if sh == 0: acc["K8"] += Pg; cnt["K8"] += 1
        if g == "id": acc["T3"] += Pg; cnt["T3"] += 1
        acc["K8xT3"] += Pg; cnt["K8xT3"] += 1
        if fam in MEMBERS_OVERRIDE and sh == 0: acc["K2"] += Pg; cnt["K2"] += 1
        del Pg
    out[f"{fam}|w"] = W0
    ens = {}
    for k in acc:
        if cnt[k] == (4 if k == "K4" else 8 if k == "K8" else 3 if k == "T3" else 2 if k == "K2" else 24):
            e = rel(acc[k] / cnt[k], Y0, CM)
            out[f"{fam}|ens_{k}|err"] = e.astype(np.float32)
            ens[k] = 100 * float((e * W0).sum() / W0.sum())
    base = fs["id"][0]
    summary[fam] = {"per_g": fs, "ens": ens, "seam_ratio": seam_ratio, "shift_admissible": shift_ok,
                    "pairing": pdiag, "VA1": VA1}
    d4d = [fs[g][1] for g in GD_D4 if g != "id" and g in fs]
    shd = [fs[f"id_s{s}"][1] for s in (1, 8) if f"id_s{s}" in fs] if shift_ok else []
    print(f"[{fam:14s}] n={len(units)} err {base:.3f} | D4 defect mean {np.mean(d4d) if d4d else float('nan'):.3f} "
          f"max {max(d4d) if d4d else float('nan'):.3f} | shift defect {['%.3f' % v for v in shd]} | "
          + " ".join(f"{k} {v:.3f} ({100*(v-base)/base:+.1f}%)" for k, v in ens.items()
                     if shift_ok or k in ("K4", "K8", "K2"))
          + f"  ({time.time()-t0:.0f}s)", flush=True)
    # PAIRING SIGNATURE: a wrong vector pairing corrupts exactly the single-axis reflections
    # {fr, fc, r90, r270} while {r180, d1, d2} are pairing-independent. Ratio >> 1 -> suspect.
    if all(g in fs for g in ("fr", "fc", "r90", "r270", "r180", "d1", "d2")):
        sig = (np.mean([fs[g][1] for g in ("fr", "fc", "r90", "r270")])
               / max(np.mean([fs[g][1] for g in ("r180", "d1", "d2")]), 1e-9))
        summary[fam]["pairing_signature"] = float(sig)
        if sig > 2.0:
            print(f"  [warn] {fam}: pairing signature {sig:.2f} (single-flip defects >> others) -> vector pairing suspect",
                  flush=True)
        # replica accuracy ratio (paper SI Table S6 convention): transformed-copy error / identity error
        summary[fam]["replica_ratio"] = {g: fs[g][0] / max(fs["id"][0], 1e-9) for g in fs}
    print("     per-g (err, defect): " + " ".join(f"{k}=({v[0]:.2f},{v[1]:.2f})" for k, v in fs.items()
                                               if not k.endswith("_s1") and not k.endswith("_s8")), flush=True)
    del P0, Y0, acc
meta["summary"] = summary
np.savez_compressed(a.out, **out, __meta__=json.dumps(meta))
print(f"EQUIV_OK {a.out} ({time.time()-t0:.0f}s)", flush=True)
