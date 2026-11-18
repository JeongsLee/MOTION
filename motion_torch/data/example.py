"""Example building for the PyTorch port (numpy only, no TensorFlow).

Ports, function by function, of the reference code paths:
  build_cond        <- motion_tf/v3/cond.py           (cond vector; MOTION_COND_STATS)
  ar_mode           <- motion_tf/v3/policy.py         (per-family temporal mode; MOTION_FORCE_MODE / MOTION_RH_FAMS)
  make_example_ar   <- motion_tf/v3/data_ar.py        (normalization incl. MOTION_INPUT_ZSCORE / MOTION_GLOBAL_NORM,
                                                     point sampling, anchors, masks)
  stack_batch, compute_global_norm, resample_sdf    <- motion_tf/v3/data_ar.py
  BOX_TABLE_2D/3D, box_dims_for                     <- motion_tf/v3/train.py + eval_stats.py bdims()
  load_local_units                                  <- latent_control_k.py LOCAL_UNITS restore
The environment switches are read from a MotionConfig instead of os.environ. Given the same sample and the same
numpy Generator state, make_example_ar returns arrays identical to the reference (verified in tests/parity_*.py).
"""
from __future__ import annotations

import pickle
from collections import defaultdict

import numpy as np

from .registry import FAMILIES, MOTION_FAMILIES, NUM_SLOTS, SEMANTIC_SLOTS
from .symbolic import PARAM_VOCAB, fit_multihot

T_IN_MAX = 10
GEOM_W = 8
DENSE2D_MAX = 128 * 128
BC_TYPES = ("periodic", "wall", "open")

# ------------------------------------------------------------------------------------------- policy (v3/policy.py)
AR_MODE = {
    "shallow_water": "ar", "com_ns": "ar", "incom_ns": "ar",
    "pdearena_ns": "ar", "pdearena_uncond": "ar",
    "CE-RP": "ar", "CE-CRP": "ar", "CE-KH": "ar", "CE-Gauss": "ar", "CE-RM": "ar",
    "NS-Sines": "ar", "NS-Gauss": "ar",
    "pdebench3d_cns_Rand_M0.1_Eta1e-08_Zeta1e-08": "ar",
    "pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08": "ar",
    "pdebench3d_cns_Turb_M1.0_Eta1e-08_Zeta1e-08": "ar",
    "ACE": "ss", "diff_react": "ss", "Wave-Layer": "ss",
    "Poisson-Gauss": "steady",
}


def ar_mode(family, steady, cfg):
    if steady:
        return "steady"
    if family in set(cfg.rh_fams):
        return "rh"
    if cfg.force_mode:
        return cfg.force_mode
    return AR_MODE.get(family, "ar")


# ------------------------------------------------------------------------------- latent box per family (v3/train.py)
BOX_TABLE_2D = {
    "shallow_water": 32, "com_ns": 32, "incom_ns": 32, "ACE": 32, "Wave-Layer": 32,
    "CE-KH": 32, "CE-Gauss": 32, "NS-Gauss": 32, "Poisson-Gauss": 32, "diff_react": 32,
    "CE-RP": 64, "CE-CRP": 64, "CE-RM": 64, "NS-Sines": 64,
    "pdearena_ns": 128, "pdearena_uncond": 128,
    "NS-PwC": 64, "GCE-RT": 64, "PoolBoil-Sub": 64, "PoolBoil-SubX": 64,
}
BOX_TABLE_3D = {
    "pdebench3d_cns_Rand_M0.1_Eta1e-08_Zeta1e-08": 16,
    "pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08": 32,
    "pdebench3d_cns_Turb_M1.0_Eta1e-08_Zeta1e-08": 32,
}


def box_dims_for(family, cfg):
    """Latent box of a family at evaluation, as eval_stats.py bdims(): MOTION_BOX_UNIFORM or the box table."""
    K = FAMILIES[family].K
    if cfg.box_uniform:
        return (int(cfg.box_uniform),) * K
    r = (BOX_TABLE_2D if K == 2 else BOX_TABLE_3D).get(family, 64 if K == 2 else 32)
    return (int(r),) * K


# ------------------------------------------------------------------------------------------- cond (v3/cond.py)
def build_cond(ex, cfg):
    spec = FAMILIES[ex["family"]]
    op_mh = np.asarray(ex["op_multihot"], np.float32)
    pf = (np.asarray(ex["param_feats"], np.float32)
          * np.asarray(ex["param_mask"], np.float32)[:, None]).reshape(-1)
    pname = np.zeros(len(PARAM_VOCAB), np.float32)
    for pid in np.asarray(ex["param_ids"]).reshape(-1):
        pname[int(pid)] = 1.0
    pname[0] = 0.0
    bc = np.zeros((3, len(BC_TYPES)), np.float32)
    for a in range(3):
        t = spec.bc[a] if a < len(spec.bc) else "periodic"
        bc[a, BC_TYPES.index(t) if t in BC_TYPES else 0] = 1.0
    steady = np.array([1.0 if ex["steady"] else 0.0], np.float32)
    kk = np.zeros(2, np.float32)
    kk[int(ex["K"]) - 2] = 1.0
    v = np.concatenate([op_mh, pf, pname, bc.reshape(-1), steady, kk])
    if cfg.cond_stats:
        zs = ex.get("zstat")
        v = np.concatenate([v, np.zeros(2 * NUM_SLOTS, np.float32) if zs is None else np.asarray(zs, np.float32)])
    assert v.shape[0] == cfg.cond_dim, (v.shape, cfg.cond_dim)
    return v.astype(np.float32)


# ------------------------------------------------------------------------------------------- data_ar helpers
def _cell_coords(dims):
    axes = [(np.arange(n) + 0.5) / n for n in dims]
    return np.stack([m.ravel() for m in np.meshgrid(*axes, indexing="ij")], -1).astype(np.float32)


def _geom_levers(sample, N):
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
    K = vol.ndim
    idx = [np.minimum((np.arange(d) + 0.5) / d * vol.shape[a], vol.shape[a] - 1).astype(int)
           for a, d in enumerate(dims[:K])]
    out = vol
    for a in range(K):
        out = np.take(out, idx[a], axis=a)
    return out[..., None].astype(np.float32)


def _pick(rng, N, n, w=None):
    if w is not None and w.sum() > 0:
        p = w / w.sum()
        nh = n // 2
        return np.concatenate([rng.choice(N, size=nh, p=p), rng.integers(0, N, size=n - nh)])
    return rng.integers(0, N, size=n)


def compute_global_norm(units, n_stat=64):
    """MOTION_GLOBAL_NORM statistics: per-channel mean/std over the first n_stat samples (all valid frames/cells)."""
    ssum = ssq = None
    cnt = None
    n = 0
    for s in units:
        if n >= n_stat:
            break
        n += 1
        f = np.asarray(s["fields"], np.float64)
        f = f.reshape(f.shape[0], -1, f.shape[-1])
        tm = s.get("tmask")
        tm = None if tm is None else np.asarray(tm).reshape(-1) > 0
        if tm is not None and tm.shape[0] == f.shape[0] and tm.any():
            f = f[tm]
        if ssum is None:
            ssum = np.zeros(f.shape[-1]); ssq = np.zeros(f.shape[-1]); cnt = 0
        ssum += f.sum(axis=(0, 1)); ssq += (f ** 2).sum(axis=(0, 1)); cnt += f.shape[0] * f.shape[1]
    m = ssum / cnt
    sd = np.sqrt(np.maximum(ssq / cnt - m ** 2, 0.0)) + 1e-6
    return m.astype(np.float32), sd.astype(np.float32), n


def make_example_ar(sample, t_in, k_fut, k_relax, n_colloc, enc_n, rng, cfg, box_dims=None, t_in_min=0,
                    phys_grid=0, ic_rand=0, ti_fixed=0, global_norm=None, ic_maxfrac=0.5, meta_drop=0.0):
    """Port of v3/data_ar.make_example_ar. `global_norm` = {family: (mean (S,), std (S,))} when cfg.global_norm.
    MOTION_META_MODE (eval-only wrong-metadata study) is not ported."""
    sample = dict(sample)
    sample["op_multihot"] = fit_multihot(sample["op_multihot"], cfg.legacy_vocab)
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
    T = flat.shape[0]
    N = node_all.shape[0]
    steady = (T == 1)
    cm = sample["cmask"].astype(bool)
    ti = 1 if steady else min(t_in, T - 1)
    ti_fixed = ti_fixed or cfg.ti_fixed
    if not steady and ti_fixed:
        ti = min(int(ti_fixed), ti)
    elif not steady and t_in_min and ti > t_in_min:
        ti = int(rng.integers(t_in_min, ti + 1))
    if not steady and ic_rand == 2:
        t0 = int(rng.integers(0, max(int(ic_maxfrac * (T - 1)) - ti + 1, 0) + 1))
    else:
        t0 = (int(rng.integers(0, T - k_fut - ti + 1))
              if (not steady and ic_rand and T - k_fut - ti >= 1) else 0)
    nometa = cfg.ft_nometa or (bool(ic_rand) and meta_drop > 0 and rng.random() < meta_drop)
    si = rng.integers(0, N, size=16384) if N > 65536 else slice(None)
    ref = flat[t0:t0 + ti][:, si]
    sd = ref.std(axis=(0, 1), keepdims=True)
    scale = np.where((sd > 1e-6) & cm[None, None], sd, 1.0)
    shift = (np.where(cm[None, None], ref.mean(axis=(0, 1), keepdims=True), 0.0).astype(np.float32)
             if cfg.input_zscore else np.zeros_like(scale, dtype=np.float32))
    if cfg.global_norm:
        assert global_norm is not None and sample["family"] in global_norm, \
            "cfg.global_norm needs global_norm={family: (mean, std)} (compute_global_norm on the train split)"
        gm, gs = global_norm[sample["family"]]
        scale = np.where(cm & (gs > 1e-5), gs, 1.0).reshape(1, 1, S).astype(scale.dtype)
        shift = np.where(cm, gm, 0.0).reshape(1, 1, S).astype(np.float32)
    sc2 = scale[0]
    _mu = np.where(cm, ref.mean(axis=(0, 1)), 0.0); _sd = np.where(cm, sd.reshape(-1), 0.0)
    zstat = np.concatenate([np.sign(_mu) * np.log1p(np.abs(_mu)), np.log1p(_sd)]).astype(np.float32)

    geom_all, wall_w = _geom_levers(sample, N)
    _gk = FAMILIES[sample["family"]].geom_kind if sample["family"] in FAMILIES else "none"
    if geom_all is None and _gk == "interface":
        _sdf = flat[t0 + ti - 1][:, 6].astype(np.float32)
        _L = float(np.percentile(np.abs(_sdf), 90)) + 1e-6
        _dist = np.abs(_sdf) / _L
        _tau = 0.1
        geom_all = np.zeros((N, GEOM_W), np.float32)
        geom_all[:, 0] = np.clip(_sdf / _L, -4.0, 4.0)
        geom_all[:, 1] = np.clip(1.0 - _dist / _tau, 0.0, 1.0) ** 2

    dense2d = (mode == "grid" and K == 2 and N <= DENSE2D_MAX)
    if dense2d:
        ni = np.arange(N)
    else:
        ni = np.unique(_pick(rng, N, enc_n, wall_w)) if N > enc_n else np.arange(N)
        if len(ni) < enc_n:
            ni = np.concatenate([ni, rng.integers(0, N, size=enc_n - len(ni))])
    qi = _pick(rng, N, n_colloc, wall_w)
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

    x_win = np.zeros((t_in, len(ni), S), np.float32)
    fmask = np.zeros(t_in, np.float32)
    if steady:
        y_node = (flat[0:1, ni] - shift) / scale
        y_q = (flat[0:1, qi] - shift) / scale
        ic_q = np.zeros((nq, S), np.float32)
        tstep = np.ones(1, np.float32)
        kf = k_relax
    else:
        anchor = t0 + ti - 1
        x_win[t_in - ti:] = (flat[t0:t0 + ti][:, ni] - shift) / scale
        fmask[t_in - ti:] = 1.0
        fut = np.arange(anchor + 1, min(anchor + 1 + k_fut, T))
        y_node = np.zeros((k_fut, len(ni), S), np.float32)
        y_q = np.zeros((k_fut, nq, S), np.float32)
        tstep = np.zeros(k_fut, np.float32)
        y_node[:len(fut)] = (flat[fut][:, ni] - shift) / scale
        y_q[:len(fut)] = (flat[fut][:, qi] - shift) / scale
        tstep[:len(fut)] = 1.0
        tv = sample.get("tmask")
        if tv is not None:
            tv = np.asarray(tv, np.float32)
            tstep[:len(fut)] *= np.array([tv[j] if j < len(tv) else 0.0 for j in fut], np.float32)
        ic_q = (flat[anchor, qi] - shift[0]) / sc2
        kf = k_fut

    bd = tuple(box_dims) if box_dims else ((32,) * K)
    sv = sample.get("sdf_vol")
    if sv is None and _gk == "interface" and mode == "grid" and not steady and dims is not None:
        sv = flat[anchor][:, 6].reshape(tuple(int(d) for d in dims[:K]))
    sdf_box = (resample_sdf(np.asarray(sv, np.float32), bd) if sv is not None
               else np.zeros((*bd, 1), np.float32))

    cmv = np.asarray(sample["cmask"], np.float32)
    fb = np.ones(cmv.shape[-1], np.float32)
    fb[:SEMANTIC_SLOTS] = cmv[:SEMANTIC_SLOTS]

    fam_names = list(FAMILIES.keys())
    nm = ({"op_multihot": np.ones_like(np.asarray(sample["op_multihot"], np.float32)),
           "param_mask": np.zeros_like(np.asarray(sample["param_mask"], np.float32)),
           "param_ids": np.zeros_like(np.asarray(sample["param_ids"]))} if nometa else {})
    return {
        "sdf_box": sdf_box,
        "box_dims": tuple(int(v) for v in bd),
        "mode": ar_mode(sample["family"], steady, cfg),
        "K": K, "family": sample["family"], "steady": steady, "dense2d": dense2d,
        "surf": FAMILIES[sample["family"]].geom_kind == "surface" if sample["family"] in FAMILIES else False,
        "fam_id": np.int32(fam_names.index(sample["family"]) if sample["family"] in fam_names else 0),
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
        "op_multihot": (np.ones_like(np.asarray(sample["op_multihot"], np.float32)) if nometa
                        else np.asarray(sample["op_multihot"], np.float32)),
        "cond": build_cond({**sample, "K": K, "steady": steady, "zstat": zstat, **nm}, cfg),
        "scale": scale.reshape(-1).astype(np.float32),
        "shift": shift.reshape(-1).astype(np.float32),
    }


BATCH_KEYS = ("coords_node", "x_win", "fmask", "geom_node", "y_node", "ic_q", "coords_q", "geom_q", "y_q",
              "tstep_mask", "cmask", "fbmask", "op_multihot", "cond", "scale", "shift", "fam_id", "sdf_box")


def stack_batch(exs):
    b = {"K": int(exs[0]["K"]), "steady": bool(exs[0]["steady"]),
         "dense2d": bool(exs[0]["dense2d"]), "roles": exs[0]["roles"],
         "box_dims": tuple(exs[0]["box_dims"]), "surf": bool(exs[0]["surf"]),
         "mode": exs[0]["mode"],
         "k_seg": int(exs[0]["k_seg"]), "families": [e["family"] for e in exs]}
    if exs[0]["dims"] is not None:
        b["dims"] = tuple(int(x) for x in exs[0]["dims"])
    for k in BATCH_KEYS:
        b[k] = np.stack([e[k] for e in exs])
    return b


def bucketed(example_iter, batch_for):
    buckets = defaultdict(list)
    for ex in example_iter:
        key = (int(ex["K"]), bool(ex["steady"]), bool(ex["dense2d"]), int(ex["coords_node"].shape[0]),
               tuple(ex["box_dims"]), bool(ex["surf"]), ex["mode"])
        buckets[key].append(ex)
        if len(buckets[key]) >= batch_for(key):
            yield stack_batch(buckets[key])
            buckets[key] = []


# ------------------------------------------------------------------------------------------- local units
def _restore_unit(u):
    """export_units.py stores active slots only; widen back to NUM_SLOTS (latent_control_k.py restore)."""
    u = dict(u)
    f = u.pop("fields")
    full = np.zeros(f.shape[:-1] + (int(u.pop("_S")),), f.dtype)
    full[..., u.pop("_act")] = f
    u["fields"] = full
    return u


def load_local_units(path):
    """{family: [unit dict, ...]} from a pickle written by motion_tf/export_units.py (LOCAL_UNITS)."""
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    return {fam: [_restore_unit(u) for u in us] for fam, us in d.items()}


def synthetic_sample(family, rng, dims=None, T=20, legacy=False):
    """Random grid sample of a registry family (tests; mirrors tests/test_v3_smoke.py)."""
    from .symbolic import encode
    spec = FAMILIES[family]
    K = spec.K
    dims = tuple(dims or ((32, 32) if K == 2 else (16, 16, 16)))
    cm = np.zeros(NUM_SLOTS, np.float32)
    from .registry import FAMILIES as _F
    # active slots: the family's channel map (raw -> universal slot), with the semantic remap of data/loader.py
    remap = {"airfrans": {4: 9}, "shapenet_car": {4: 9}, "drivaernet_pressure": {4: 9}, "Wave-Layer": {6: 10},
             "geofno_airfoil": {4: 11}, "geofno_elasticity": {4: 11}}.get(family, {})
    for c in spec.channels:
        cm[remap.get(c, c)] = 1.0
    Tn = 1 if spec.time == "steady" else T
    fields = rng.normal(size=(Tn, *dims, NUM_SLOTS)).astype(np.float32) * cm + 0.3 * cm
    s = {"family": family, "K": K, "mode": "grid", "dims": np.array(dims), "fields": fields.astype(np.float32),
         "cmask": cm, "roles": list(spec.roles[:K]), **encode(spec, legacy=False)}
    return s
