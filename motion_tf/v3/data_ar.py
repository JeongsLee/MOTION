"""AR training examples over the unified corpus (DESIGN_V3 §2.3).

Contract per example (mesh-agnostic; grid = its cell-centers):
  coords_node (N,K)      fixed node set: encoder input AND the AR feedback set
  x_win       (T_in,N,S) observed window at the nodes (right-aligned; steady -> zeros)
  y_node      (KF,N,S)   future frames at the SAME nodes (feedback supervision)
  coords_q / geom_q / y_q  extra collocation targets per frame (near-wall biased for sdf)
  tstep_mask  (KF,)      frame validity (short horizons zero-padded)
  cond        (COND_DIM,) symbolic equation + params + BC + steady + K   (v3.cond)

2D grids keep the FULL native cell set (dense2d=True; 128^2=16384 nodes) so per-frame AR
feedback is dense and the transport warp has a dense previous frame. 3D grids and meshes
subsample enc_n nodes (bounded encoder; collocation keeps native-res supervision).

Steady families: x_win = zeros, the target is the equilibrium field; the trainer runs
K_relax AR steps and supervises EVERY step against the target (pseudo-transient
continuation; later steps weighted higher).
"""
from __future__ import annotations

from collections import defaultdict

import os

import numpy as np

from data.loader import SLOT_REMAP
from data.registry import FAMILIES, SEMANTIC_SLOTS


from .cond import build_cond
from .policy import ar_mode

T_IN_MAX = 10
GEOM_W = 8
DENSE2D_MAX = 128 * 128            # keep 2D grids fully dense up to this cell count


def _cell_coords(dims):
    axes = [(np.arange(n) + 0.5) / n for n in dims]
    return np.stack([m.ravel() for m in np.meshgrid(*axes, indexing="ij")], -1).astype(np.float32)


def _geom_levers(sample, N):
    """v2's geometry levers: sdf families -> [norm SDF, BL mask] + near-wall weights."""
    wall_w = None
    geom = None
    gk = FAMILIES[sample["family"]].geom_kind if sample["family"] in FAMILIES else "none"
    if sample.get("geom") is not None:
        g = np.asarray(sample["geom"], np.float32).reshape(N, -1)
        if g.shape[1] < GEOM_W:
            g = np.concatenate([g, np.zeros((N, GEOM_W - g.shape[1]), np.float32)], -1)
        g = g[:, :GEOM_W].copy()
        if gk == "sdf":
            sdf = g[:, 0].copy()
            L = float(np.percentile(np.abs(sdf), 90)) + 1e-6
            dist = np.abs(sdf) / L
            tau = 0.1
            g[:, 0] = np.clip(sdf / L, -4.0, 4.0)
            g[:, 1] = np.clip(1.0 - dist / tau, 0.0, 1.0) ** 2
            wall_w = np.exp(-dist / tau).astype(np.float64)
        geom = g
    return geom, wall_w


def resample_sdf(vol, dims):
    """(R,)*K stored SDF volume -> (*dims,1) by nearest-cell resampling (K-agnostic).
    Nearest (not linear) keeps it cheap and the field is smooth at 48^3 vs box 32^3."""
    K = vol.ndim
    idx = [np.minimum((np.arange(d) + 0.5) / d * vol.shape[a], vol.shape[a] - 1).astype(int)
           for a, d in enumerate(dims[:K])]
    out = vol
    for a in range(K):
        out = np.take(out, idx[a], axis=a)
    return out[..., None].astype(np.float32)


def _pick(rng, N, n, w=None):
    """n indices from N; half near-wall-biased when weights given (v2 lever)."""
    if w is not None and w.sum() > 0:
        p = w / w.sum()
        nh = n // 2
        return np.concatenate([rng.choice(N, size=nh, p=p),
                               rng.integers(0, N, size=n - nh)])
    return rng.integers(0, N, size=n)


SCALE_OVERRIDE = None    # (S,) per-channel scale; None = per-window std (default, training/eval)
# REVISION-R1 MOTION-NS-S z-score arm (10-19): MOTION_INPUT_ZSCORE=1 also removes the per-window channel mean
# (instance z-score over the observed window, as BCAT / PROSE-FD / the published MOTION-NS), applied to inputs,
# targets and the anchor alike; the model works in z-space and the physical eval adds the mean back (example key
# "shift"). Default off = the DC-preserving std-only normalization.
INPUT_ZSCORE = os.environ.get("MOTION_INPUT_ZSCORE", "0") == "1"
# REVISION-R1 MOTION-IVP-S Poseidon-style normalization (10-21, user): MOTION_GLOBAL_NORM=1 (requires MOTION_INPUT_ZSCORE=1,
# whose "shift" path carries the mean back for the physical eval) replaces the per-window statistics by FIXED per-family,
# per-channel mean/std, computed as the published MOTION-IVP / Poseidon loader does (motion_tf data/poseidon.py
# _global_stats): the first 64 training trajectories x all frames x all cells, std + 1e-6. Each sample keeps its physical
# amplitude relative to its family. GLOBAL_NORM is filled by the caller (train.py / score_ivp_l1.py) before examples are built.
GLOBAL_NORM_ON = os.environ.get("MOTION_GLOBAL_NORM", "0") == "1"
GLOBAL_NORM = {}                                            # family -> (mean (S,), std (S,)) float32


def compute_global_norm(units, n_stat=64):
    """Per-channel mean/std over the first n_stat samples of `units` (all valid frames, all cells/nodes)."""
    ssum = ssq = None; cnt = None; n = 0
    for s in units:
        if n >= n_stat:
            break
        n += 1
        f = np.asarray(s["fields"], np.float64)
        f = f.reshape(f.shape[0], -1, f.shape[-1])          # (T,N,S)
        tm = s.get("tmask")
        tm = None if tm is None else np.asarray(tm).reshape(-1) > 0
        if tm is not None and tm.shape[0] == f.shape[0] and tm.any():
            f = f[tm]                                       # padded frames excluded
        if ssum is None:
            ssum = np.zeros(f.shape[-1]); ssq = np.zeros(f.shape[-1]); cnt = 0
        ssum += f.sum(axis=(0, 1)); ssq += (f ** 2).sum(axis=(0, 1)); cnt += f.shape[0] * f.shape[1]
    m = ssum / cnt
    sd = np.sqrt(np.maximum(ssq / cnt - m ** 2, 0.0)) + 1e-6
    return m.astype(np.float32), sd.astype(np.float32), n


# REVISION-R1 metadata-free transfer (MOTION_FT_NOMETA=1): no equation information is given -- every operator
# token on (all mechanisms available, as for a model that is not told the equation), parameters and parameter
# names removed from cond. BC / steady / K stay (they describe the grid and the task format, not the physics).
FT_NOMETA = os.environ.get("MOTION_FT_NOMETA", "0") == "1"
# REVISION-R1 MOTION-IVP-S pretraining: metadata dropout (probability p per training example; same transform as
# FT_NOMETA) so that one model serves the metadata-free transfer setting. Train-time only (ic_rand > 0).
META_DROP = float(os.environ.get("MOTION_META_DROP", "0"))
# REVISION-R1 MOTION-IVP-S: all2all as in the motion_tf IVP pretraining (ic_rand=2): the IC frame is drawn from
# [0, IC_MAXFRAC*(T-1)] and the future window may run past the trajectory end (those frames are masked).
IC_MAXFRAC = float(os.environ.get("MOTION_IC_MAXFRAC", "0.5"))
# REVISION-R1 R1-2 (evaluation only): wrong or missing metadata. MOTION_META_MODE =
#   all_on : every operator token on, parameters masked          none  : no operator token, parameters masked
#   swap   : the operator tokens of the least-overlapping other family (parameters masked)
#   drop1  : the family's first listed operator token removed     add1  : one spurious token added (buoyancy or wave)
META_MODE = os.environ.get("MOTION_META_MODE", "")


def _meta_override(sample):
    """-> (op_multihot, param overrides dict) for META_MODE, or None when off."""
    if not META_MODE:
        return None
    from data.registry import FAMILIES as _F, MOTION_FAMILIES as _MF
    from data.symbolic import op_ids as _op_ids, OPERATOR_VOCAB as _V
    fam = sample["family"]; ops = list(_F[fam].operators)
    mh0 = np.asarray(sample["op_multihot"], np.float32)
    if META_MODE == "all_on":
        mh = np.ones_like(mh0)
    elif META_MODE == "none":
        mh = np.zeros_like(mh0)
    elif META_MODE == "swap":
        cand = [f for f in _MF if f != fam]
        other = min(cand, key=lambda f: (len(set(_F[f].operators) & set(ops)), _MF.index(f)))
        mh = _op_ids(list(_F[other].operators))[1].astype(np.float32)
    elif META_MODE == "drop1":
        mh = _op_ids(ops[1:])[1].astype(np.float32)
    elif META_MODE == "add1":
        extra = "buoyancy" if "buoyancy" not in ops else "wave"
        mh = _op_ids(ops + [extra])[1].astype(np.float32)
    else:
        raise ValueError(META_MODE)
    mh = mh[:len(mh0)] if len(mh) >= len(mh0) else np.concatenate([mh, np.zeros(len(mh0) - len(mh), np.float32)])
    return mh, {"param_mask": np.zeros_like(np.asarray(sample["param_mask"], np.float32)),
                "param_ids": np.zeros_like(np.asarray(sample["param_ids"]))}
# REVISION-R1 strict IVP at the pretrained architecture: the model keeps t_in=10 (the encoder input projection
# is sized by t_in, so --t_in 1 would re-initialise it), and only the first MOTION_TI_FIXED frames are observed
# (the rest masked by fmask). Applies to every make_example_ar call (train stream, in-training eval, scorers).
TI_FIXED = int(os.environ.get("MOTION_TI_FIXED", "0"))
FRAME_STRIDE = int(os.environ.get("MOTION_FRAME_STRIDE", "1"))   # REVISION-R1 10-29: see make_example_ar
# REVISION-R1 10-27 (user): MOTION_INIT_REPEAT=n -> a window that starts at the initial condition (t0 = 0) with fewer
# than n observed frames is filled up to n frames with copies of the initial frame (fmask 1). For the second-order
# wave tasks this encodes d_t u(0) = 0 (given by construction) as a zero frame-to-frame change, the form in which the
# pretrained window reads temporal change; one frame alone carries no time-derivative information. Train and score.
INIT_REPEAT = int(os.environ.get("MOTION_INIT_REPEAT", "0"))
# REVISION-R1 10-27 (user): MOTION_WAVE_V0="src:dst" -> the observed time derivative of slot src at the anchor frame,
# v0 = u(anchor) - u(anchor - 1) (0 at anchor 0, the zero-velocity initial condition), is written into the unused
# input slot dst of the anchor frame (closed wave IVP of the submitted version, motion_tf train_ivp.py v0-slot).
_V0 = os.environ.get("MOTION_WAVE_V0", "")
WAVE_V0 = tuple(int(x) for x in _V0.split(":")) if _V0 else None
_V0_LOG = 0


def make_example_ar(sample, t_in, k_fut, k_relax, n_colloc, enc_n, rng, box_dims=None,
                    t_in_min=0, phys_grid=0, ic_rand=0, ti_fixed=0):
    K, mode = int(sample["K"]), sample["mode"]
    fields = sample["fields"]
    S = fields.shape[-1]
    if mode == "grid":
        dims = tuple(int(x) for x in sample["dims"])
        node_all = _cell_coords(dims)
        flat = fields.reshape(fields.shape[0], -1, S)
    else:
        dims = None
        node_all = np.asarray(sample["coords"], np.float32)
        flat = fields
    # REVISION-R1 10-29 (user): MOTION_FRAME_STRIDE=s keeps every s-th stored frame (0, s, 2s, ...) before the window and
    # targets are drawn, so one AR step spans s stored frames (s = 2: Poseidon's time step, snapshots 0, 2, ..., 20; a
    # k_fut-10 model then covers the 20-frame horizon). Train, val and test share this code path. Default 1 (unchanged).
    if FRAME_STRIDE > 1 and flat.shape[0] > 1:
        flat = flat[::FRAME_STRIDE]
    T = flat.shape[0]
    N = node_all.shape[0]
    steady = (T == 1)

    # std-only per-channel normalization (v2 lesson: keep the DC component).
    # GATHER-THEN-NORMALIZE (MOTION-NCS 08-06): the old order normalized the FULL field array
    # before slicing, which for a 3D trajectory is a ~2.7 GB divide plus a full-volume std per
    # example — the builder cost that kept the GPU idle even once the read was amortized. The
    # std is now estimated on a 16k-cell subsample when the grid is large (estimator noise
    # ~1/sqrt(160k), irrelevant), and only the gathered slices are divided. `scale` remains the
    # exact factor applied, so the physical-space eval (which multiplies it back) is unaffected.
    cm = sample["cmask"].astype(bool)
    # WINDOW POSITION is drawn BEFORE the statistics, because the statistics must describe the
    # window that is actually fed. Sampling t0 afterwards (the m1ivp bug, 08-10) normalized a
    # mid-trajectory window by the initial condition's scale.
    ti = 1 if steady else min(t_in, T - 1)
    ti_fixed = ti_fixed or TI_FIXED                 # env MOTION_TI_FIXED: strict-IVP observed history at a t_in-10 model
    if not steady and ti_fixed:                     # REVISION-R1 eval: fixed observed history (<= t_in)
        ti = min(int(ti_fixed), ti)
    elif not steady and t_in_min and ti > t_in_min:
        # VARIABLE HISTORY. Train the same weights across the whole spectrum from a single
        # initial condition (a true IVP, what a solver is actually given) up to the long
        # window the forecasting literature uses. Two reasons: one checkpoint then serves
        # both regimes, and — more importantly — with a short history the model CANNOT infer
        # coefficients from the observed decay, so the pressure to actually use the symbolic
        # equation / BC / coefficient-field conditioning comes from the objective rather than
        # from hope. The steady families already train at zero history, so the masked-window
        # path is exercised; this extends it to the transient ones.
        ti = int(rng.integers(t_in_min, ti + 1))
    # IC-POSITION RANDOMIZATION (B-II all2all analog): the INPUT stays a single snapshot
    # (or fixed-length window) but its temporal position is sampled, so arbitrary mid-
    # trajectory states become initial conditions. Eval keeps t0=0 (strict from-IC).
    if not steady and ic_rand == 2:                 # motion_tf IVP all2all (partial windows masked via tstep)
        t0 = int(rng.integers(0, max(int(IC_MAXFRAC * (T - 1)) - ti + 1, 0) + 1))
    else:
        t0 = (int(rng.integers(0, T - k_fut - ti + 1))
              if (not steady and ic_rand and T - k_fut - ti >= 1) else 0)
    nometa = FT_NOMETA or (bool(ic_rand) and META_DROP > 0 and rng.random() < META_DROP)
    _mo = _meta_override(sample)
    si = rng.integers(0, N, size=16384) if N > 65536 else slice(None)
    ref = flat[t0:t0 + ti][:, si]
    sd = ref.std(axis=(0, 1), keepdims=True)
    # NO within-frame floor on sd (tried 08-10, reverted): a 1-frame window can leave a channel
    # exactly constant (CE-KH's transverse velocity, CE-Gauss's density at t=0, measured), and
    # flooring at 5% of the channel RMS then divides a constant ~1.0 field by 0.05 against a true
    # trajectory variation of 0.008 -- the DC-preserving convention parks that channel at ~20x
    # its neighbours, which is an input distribution the pretrained weights have never seen. It
    # made CE worse, not better (188% -> 739%). The pretraining fallback below is what m1 was
    # trained under, so the warm start stays on-distribution; degenerate channels simply keep
    # their physical units.
    scale = np.where((sd > 1e-6) & cm[None, None], sd, 1.0)   # (1,1,S)
    if SCALE_OVERRIDE is not None:                          # REVISION-R1 latent composition: one shared normalization
        scale = np.asarray(SCALE_OVERRIDE, scale.dtype).reshape(scale.shape)
    shift = (np.where(cm[None, None], ref.mean(axis=(0, 1), keepdims=True), 0.0).astype(np.float32)
             if INPUT_ZSCORE else np.zeros_like(scale, dtype=np.float32))   # (1,1,S)
    if GLOBAL_NORM_ON:                                       # REVISION-R1 Poseidon-style fixed per-family statistics
        gm, gs = GLOBAL_NORM[sample["family"]]
        scale = np.where(cm & (gs > 1e-5), gs, 1.0).reshape(1, 1, S).astype(scale.dtype)
        shift = np.where(cm, gm, 0.0).reshape(1, 1, S).astype(np.float32)
    sc2 = scale[0]                                            # (1,S) for 2-D slices
    _mu = np.where(cm, ref.mean(axis=(0, 1)), 0.0); _sd = np.where(cm, sd.reshape(-1), 0.0)   # MOTION_COND_STATS
    zstat = np.concatenate([np.sign(_mu) * np.log1p(np.abs(_mu)), np.log1p(_sd)]).astype(np.float32)

    geom_all, wall_w = _geom_levers(sample, N)
    _gk = FAMILIES[sample["family"]].geom_kind if sample["family"] in FAMILIES else "none"
    if geom_all is None and _gk == "interface":
        # EVOLVING-INTERFACE GEOMETRY (PB arm C, 08-12): the family's own SDF channel at the
        # WINDOW ANCHOR fills the geometry pathway -- same normalization as the static-sdf
        # families so the pretrained geom weights transfer, quasi-static over the segment.
        # Also biases node/colloc sampling toward the interface (the wall_w lever).
        _sdf = flat[t0 + ti - 1][:, 6].astype(np.float32)      # dfun slot
        _L = float(np.percentile(np.abs(_sdf), 90)) + 1e-6
        _dist = np.abs(_sdf) / _L
        _tau = 0.1
        geom_all = np.zeros((N, GEOM_W), np.float32)
        geom_all[:, 0] = np.clip(_sdf / _L, -4.0, 4.0)
        geom_all[:, 1] = np.clip(1.0 - _dist / _tau, 0.0, 1.0) ** 2
        if int(os.environ.get("IFACE_BIAS", "0")):
            # interface-biased sampling starved the bulk flow (armC1 lagged A/B 2x at
            # matched steps, 08-12) -- OFF by default, geom features alone are the arm.
            wall_w = np.exp(-_dist / _tau).astype(np.float64)

    dense2d = (mode == "grid" and K == 2 and N <= DENSE2D_MAX)
    if dense2d:
        ni = np.arange(N)
    else:
        ni = np.unique(_pick(rng, N, enc_n, wall_w)) if N > enc_n else np.arange(N)
        if len(ni) < enc_n:                                   # pad to fixed count (replace)
            ni = np.concatenate([ni, rng.integers(0, N, size=enc_n - len(ni))])
    qi = _pick(rng, N, n_colloc, wall_w)
    # PHYS-GRID (Fig2 arm-2): prepend a regular stride lattice (phys_grid^K cells, C-order) to
    # the collocation queries so the physics-residual loss can take finite differences on a
    # coarse grid — anchor values (ic_q) and targets (y_q) then come along for free. 3D
    # transient grid families only; every consumer below indexes with qi generically.
    if phys_grid and mode == "grid" and K == 3 and not steady and dims is not None:
        g = int(phys_grid)
        ax = [np.arange(0, d, max(d // g, 1))[:g] for d in dims]
        gi = (ax[0][:, None, None] * dims[1] * dims[2]
              + ax[1][None, :, None] * dims[2] + ax[2][None, None, :]).ravel()
        qi = np.concatenate([gi.astype(qi.dtype), qi])
    nq = len(qi)

    coords_node = node_all[ni]
    geom_node = geom_all[ni] if geom_all is not None else np.zeros((len(ni), GEOM_W), np.float32)
    geom_q = geom_all[qi] if geom_all is not None else np.zeros((nq, GEOM_W), np.float32)

    x_win = np.zeros((t_in, len(ni), S), np.float32)          # fixed width, right-aligned
    fmask = np.zeros(t_in, np.float32)
    if steady:
        y_node = (flat[0:1, ni] - shift) / scale              # (1,N,S) target equilibrium
        y_q = (flat[0:1, qi] - shift) / scale
        ic_q = np.zeros((nq, S), np.float32)            # relaxation starts from zero
        tstep = np.ones(1, np.float32)
        kf = k_relax
    else:
        anchor = t0 + ti - 1
        x_win[t_in - ti:] = (flat[t0:t0 + ti][:, ni] - shift) / scale
        fmask[t_in - ti:] = 1.0
        if INIT_REPEAT > ti and t0 == 0:                     # copies of u(0) before the observed frames
            nr = min(INIT_REPEAT, t_in)
            x_win[t_in - nr:t_in - ti] = x_win[t_in - ti]
            fmask[t_in - nr:t_in - ti] = 1.0
        if WAVE_V0 is not None:                               # every observed frame carries its backward difference
            _vs, _vd = WAVE_V0
            _sc = float(np.asarray(scale).reshape(-1)[_vs])
            for _k in range(ti):
                _t = t0 + _k
                _prev = flat[_t - 1][ni, _vs] if _t >= 1 else flat[_t][ni, _vs]
                x_win[t_in - ti + _k, :, _vd] = (flat[_t][ni, _vs] - _prev) / _sc
            global _V0_LOG
            if _V0_LOG < 6:                                  # visible check in the job log (first examples)
                _V0_LOG += 1
                print(f"[v0] anchor {anchor} ti {ti} |v0| {float(np.abs(x_win[-1, :, _vd]).mean()):.4f} "
                      f"|u| {float(np.abs(x_win[-1, :, _vs]).mean()):.4f}", flush=True)
        fut = np.arange(anchor + 1, min(anchor + 1 + k_fut, T))
        y_node = np.zeros((k_fut, len(ni), S), np.float32)
        y_q = np.zeros((k_fut, nq, S), np.float32)
        tstep = np.zeros(k_fut, np.float32)
        y_node[:len(fut)] = (flat[fut][:, ni] - shift) / scale
        y_q[:len(fut)] = (flat[fut][:, qi] - shift) / scale
        tstep[:len(fut)] = 1.0
        # TEMPORAL VALIDITY. Families whose native horizon is shorter than the cache width are
        # clip-repeat padded (pdearena_uncond: 14 real frames in a 20-frame cache, so frames
        # 14-19 are copies of frame 13). Supervising or scoring those is measuring a constant:
        # of the 10 predicted frames only 4 are real, which is exactly the T=4 horizon PROSE/BCAT
        # restrict that family to. Leaving them in flatters the number rather than hurting it.
        tv = sample.get("tmask")
        if tv is not None:
            tv = np.asarray(tv, np.float32)
            if FRAME_STRIDE > 1 and len(tv) > 1:
                tv = tv[::FRAME_STRIDE]
            tstep[:len(fut)] *= np.array([tv[j] if j < len(tv) else 0.0 for j in fut], np.float32)
        ic_q = (flat[anchor, qi] - shift[0]) / sc2            # hard anchor at the colloc points
        kf = k_fut

    # GINO lever: dense volumetric shape context for co-dimension-1 (surface) families, whose
    # points occupy only ~3% of the latent box. Zeros where a family has none (inert).
    bd = tuple(box_dims) if box_dims else ((32,) * K)
    sv = sample.get("sdf_vol")
    if sv is None and _gk == "interface" and mode == "grid" and not steady and dims is not None:
        # arm C: the banks' geom_box carries the anchor-frame interface SDF as well.
        sv = flat[anchor][:, 6].reshape(tuple(int(d) for d in dims[:K]))
    sdf_box = (resample_sdf(np.asarray(sv, np.float32), bd) if sv is not None
               else np.zeros((*bd, 1), np.float32))

    # FEEDBACK MASK — which slots may carry state back into this family's own window.
    # Every SEMANTIC channel is now strictly private: a family may only feed back what it owns, so
    # no family can write into another's physical quantity and no shared hidden convention can
    # form across families. That is the independence the slot split is for. It does cost the
    # model its only cross-segment memory (the latent box is rebuilt every segment), which is why
    # slots >= SEMANTIC_SLOTS exist and stay open for everyone: unscored, unowned, collision-free.
    cmv = np.asarray(sample["cmask"], np.float32)
    fb = np.ones(cmv.shape[-1], np.float32)
    fb[:SEMANTIC_SLOTS] = cmv[:SEMANTIC_SLOTS]      # semantic channels: strictly this family's

    fam_names = list(FAMILIES.keys())
    return {
        "sdf_box": sdf_box,
        "box_dims": tuple(int(v) for v in bd),   # per-family (see diag_resolution.py floors)
        # TEMPORAL MODE (MOTION-NCS): "ar" = per-frame full AR, "ss" = one whole-horizon
        # single-shot call, "steady" = K_relax relaxation. Set by v3/policy.py per family.
        "mode": ar_mode(sample["family"], steady),
        "K": K, "family": sample["family"], "steady": steady, "dense2d": dense2d,
        # A 2-manifold through a 32^3 box touches ~1000 of 32768 cells: 97% of the volumetric
        # path carries nothing for these families, and what it does carry costs the model an
        # embedding AND a projection that differ for every car. The slice tokens are the
        # manifold-intrinsic alternative and they already exist — see V3Model.step.
        "surf": FAMILIES[sample["family"]].geom_kind == "surface"
        if sample["family"] in FAMILIES else False,
        "fam_id": np.int32(fam_names.index(sample["family"])
                           if sample["family"] in fam_names else 0),
        "dims": np.array(dims) if dims else None, "k_seg": kf,
        "coords_node": coords_node.astype(np.float32),
        "x_win": x_win.astype(np.float32), "fmask": fmask,
        "geom_node": geom_node.astype(np.float32),
        "y_node": y_node.astype(np.float32), "ic_q": ic_q.astype(np.float32),
        "coords_q": node_all[qi].astype(np.float32),
        "geom_q": geom_q.astype(np.float32),
        "y_q": y_q.astype(np.float32),
        "tstep_mask": tstep,
        "cmask": sample["cmask"].astype(np.float32), "fbmask": fb,
        "roles": [int(r) for r in sample["roles"]],
        "op_multihot": (_mo[0] if _mo is not None else
                        np.ones_like(np.asarray(sample["op_multihot"], np.float32)) if nometa
                        else np.asarray(sample["op_multihot"], np.float32)),
        "cond": build_cond({**sample, "K": K, "steady": steady, "zstat": zstat,
                            **({"op_multihot": np.ones_like(np.asarray(sample["op_multihot"], np.float32)),
                                "param_mask": np.zeros_like(np.asarray(sample["param_mask"], np.float32)),
                                "param_ids": np.zeros_like(np.asarray(sample["param_ids"]))} if nometa else {}),
                            **({"op_multihot": _mo[0], **_mo[1]} if _mo is not None else {})}),
        "scale": scale.reshape(-1).astype(np.float32),
        "shift": shift.reshape(-1).astype(np.float32),
    }


def stack_batch(exs):
    """Same-(K, steady, dense2d, N) examples -> batch dict of stacked arrays."""
    b = {"K": int(exs[0]["K"]), "steady": bool(exs[0]["steady"]),
         "dense2d": bool(exs[0]["dense2d"]), "roles": exs[0]["roles"],
         "box_dims": tuple(exs[0]["box_dims"]), "surf": bool(exs[0]["surf"]),
         "mode": exs[0]["mode"],
         "k_seg": int(exs[0]["k_seg"]), "families": [e["family"] for e in exs]}
    if exs[0]["dims"] is not None:
        b["dims"] = tuple(int(x) for x in exs[0]["dims"])
    for k in ("coords_node", "x_win", "fmask", "geom_node", "y_node", "ic_q",
              "coords_q", "geom_q", "y_q", "tstep_mask", "cmask", "fbmask", "op_multihot",
              "cond", "scale", "shift", "fam_id", "sdf_box"):
        b[k] = np.stack([e[k] for e in exs])
    return b


def bucketed(example_iter, batch_for):
    """Bucket by (K, steady, dense2d, N_node, box_dims); yield stacked batches.
    batch_for(key) -> per-bucket batch size (3D buckets may need smaller).

    box_dims is part of the key because it is now per-family: sdf_box carries that shape, so
    mixing resolutions inside a batch would not stack, and the traced graph is specialised on it.
    """
    buckets = defaultdict(list)
    for ex in example_iter:
        key = (int(ex["K"]), bool(ex["steady"]), bool(ex["dense2d"]),
               int(ex["coords_node"].shape[0]), tuple(ex["box_dims"]), bool(ex["surf"]),
               ex["mode"])
        buckets[key].append(ex)
        if len(buckets[key]) >= batch_for(key):
            yield stack_batch(buckets[key])
            buckets[key] = []
