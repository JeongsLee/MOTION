"""Streaming data pipeline (tf.data) — reads each batch on-the-fly from the on-disk cache, in the
background, overlapped with GPU compute via prefetch. NO upfront 40GB copy, NO 106GB in-RAM load,
instant startup, and scales to datasets far larger than RAM (mandatory for the full comp/incomp-NS
6-family set). Per training step consumes only ~one batch (~10MB), well within the object-storage
volume's bandwidth, so shards stream directly from the volume.

Split: per family, hold out the LAST `test_frac` (by pdearena shard, or by index for single-file
SWE/DR) as a held-out test set; everything else is train. Train is streamed; the (smaller) test set
is materialized in RAM once for evaluation.
"""
from __future__ import annotations
import glob
import os

import numpy as np
import tensorflow as tf

from .prose import SPEC, _ensure_cache, _npy_shape, C_MAX


def _resolve_cache(ds, n_per, t_num, cache_dir):
    """Find an existing cache for `ds` and REUSE it — never rebuild a smaller subset from raw when a
    bigger processed cache already exists. Prefer the exact n_per; else the LARGEST available cache
    (build_plan then caps to n_per). Prefer shards over a single .npy at equal n. Build only if none."""
    import re
    exact_npy = os.path.join(cache_dir, f"{ds}_n{n_per}_t{t_num}.npy")
    exact_sh = os.path.join(cache_dir, f"{ds}_n{n_per}_t{t_num}_shards")
    if os.path.exists(os.path.join(exact_sh, "_DONE")):
        return exact_sh
    if os.path.exists(exact_npy):
        return exact_npy
    cands = [c for c in glob.glob(os.path.join(cache_dir, f"{ds}_n*_t{t_num}_shards"))
             if os.path.exists(os.path.join(c, "_DONE"))]
    cands += glob.glob(os.path.join(cache_dir, f"{ds}_n*_t{t_num}.npy"))
    if cands:
        def _key(c):                                          # (n_samples, shards>npy)
            m = re.search(rf"{re.escape(ds)}_n(\d+)_t", os.path.basename(c))
            return (int(m.group(1)) if m else 0, 1 if c.endswith("_shards") else 0)
        return max(cands, key=_key)
    return _ensure_cache(ds, n_per, t_num, cache_dir)         # nothing exists → build (only then)


def _family_files(ds, n_per, t_num, cache_dir):
    """Cache file(s) for a family, CAPPED to the first n_per samples (cumulative across shards) — so a
    900-sample run just reads the head of the existing full cache instead of rebuilding a 900 cache."""
    path = _resolve_cache(ds, n_per, t_num, cache_dir)
    files = (sorted(glob.glob(os.path.join(path, "shard_*.npy"))) if os.path.isdir(path) else [path])
    out, run = [], 0
    for f in files:
        if run >= n_per:
            break
        take = min(int(_npy_shape(f)[0]), n_per - run)        # cap the (last) file to hit n_per exactly
        out.append((f, take)); run += take
    return out


def _padcm(cm):                                              # 6-wide SPEC c_mask -> C_MAX-wide (zero-pad)
    a = np.asarray(cm, np.float32)
    return np.concatenate([a, np.zeros(max(0, C_MAX - a.shape[0]), np.float32)])[:C_MAX]


def build_plan(datasets, n_per, t_num, cache_dir, test_frac=0.1, seed=0):
    """Return (train_plan, test_plan, descs, cmasks). Each plan entry = (file, fam_id, slots, idxs)."""
    train, test = [], []
    descs = {fi: np.array(SPEC[ds]["desc"], np.float32) for fi, ds in enumerate(datasets)}
    cmasks = {fi: _padcm(SPEC[ds]["c_mask"]) for fi, ds in enumerate(datasets)}
    for fi, ds in enumerate(datasets):
        slots = SPEC[ds]["slots"]
        files = _family_files(ds, n_per, t_num, cache_dir)
        if len(files) > 1:                                   # pdearena: split by SHARD
            n_te = max(1, int(round(test_frac * len(files))))
            for f, c in files[:-n_te]:
                train.append((f, fi, slots, np.arange(c)))
            for f, c in files[-n_te:]:
                test.append((f, fi, slots, np.arange(c)))
        else:                                                # SWE/DR: split by index in the one file
            f, c = files[0]
            n_te = max(1, int(round(test_frac * c)))
            train.append((f, fi, slots, np.arange(0, c - n_te)))
            test.append((f, fi, slots, np.arange(c - n_te, c)))
    print(f"  [stream] train units={len(train)} test units={len(test)}", flush=True)
    return train, test, descs, cmasks


def _to_6slot_one(sample, slots):                            # (t,128,128,d) → (t,128,128,6)
    out = np.zeros(sample.shape[:3] + (C_MAX,), np.float32)
    for j, s in enumerate(slots):
        out[..., s] = sample[..., j]
    return out


def _instance_norm(u_in, u_tg, eps=1e-6):
    """PROSE-style per-trajectory instance norm: per-channel mean/std over the INPUT window (T_in,H,W)
    normalize BOTH input and target. Equalizes the wildly different per-family scales (density~1 vs
    velocity vs height) so multi-family training is well-conditioned. Unused (all-zero) slots have
    mean=std=0 → (0-0)/eps = 0, stay zero. Model trains/predicts in this normalized space."""
    m = u_in.mean(axis=(0, 1, 2), keepdims=True)             # (1,1,1,C) per-channel
    s = u_in.std(axis=(0, 1, 2), keepdims=True)
    return (u_in - m) / (s + eps), (u_tg - m) / (s + eps)


def _global_stats(plan, Ti, n_stat=64):
    """Per-family GLOBAL per-channel input-window mean/std (Poseidon-style FIXED normalization), computed
    once over up to n_stat trajectories/family. Unlike per-sample instance norm, a fixed per-family std
    PRESERVES each sample's physical energy ratio → high-energy (turbulent) samples keep their weight in
    the (MSE) loss, while channels stay balanced (each ÷ its own family std). Returns {fam_id:(m(C,),s(C,))}."""
    stats = {}
    for fid in sorted({u[1] for u in plan}):
        acc, cnt = [], 0
        for f, fam, slots, idxs in [u for u in plan if u[1] == fid]:
            arr = np.load(f, mmap_mode="r")
            for i in idxs:
                acc.append(_to_6slot_one(np.asarray(arr[i]), slots)[:Ti].astype(np.float32))
                cnt += 1
                if cnt >= n_stat:
                    break
            if cnt >= n_stat:
                break
        a = np.stack(acc, 0)                                 # (n,Ti,H,W,C)
        stats[fid] = (a.mean(axis=(0, 1, 2, 3)).astype(np.float32),
                      a.std(axis=(0, 1, 2, 3)).astype(np.float32))
    return stats


def _make_gen(plan, Ti, Nt, descs, cmasks, tmasks, inorm=False, tgt0=False, gstats=None, fills=None,
              order_seed=None, geom_chs=None, ntd=None):
    ntd = ntd or Nt                                          # data frames (Nt may be padded past the data)
    # order_seed: if set, use a FIXED-seed RNG for the shuffle so the trajectory order is DETERMINISTIC and
    # reproducible across runs (independent of the training seed). Required for the data-efficacy scaling
    # curve: milestone-N then = the same first-N trajectories every run. None = legacy per-run random shuffle.
    def gen():
        rng = np.random.default_rng(order_seed) if order_seed is not None else np.random
        order = rng.permutation(len(plan))                   # reshuffle file order each epoch (det if order_seed)
        for k in order:
            f, fam, slots, idxs = plan[k]
            arr = np.load(f)                                 # load one file (shard ~130MB / SWE-DR)
            d, cm, tm = descs[fam], cmasks[fam], tmasks[fam]
            gch = None if geom_chs is None else geom_chs.get(fam)   # native channel → geom_mask (cfdbench boundary)
            fv = fills[fam] if fills is not None else None   # (C,) normalized fill for unused slots (nan = leave)
            for i in rng.permutation(idxs):
                u = _to_6slot_one(arr[i], slots)             # (t,128,128,6)
                xin = u[:Ti].astype(np.float32)
                # tgt0 (ic_first / full-trajectory supervision): target = the WHOLE window u[0:Nt]
                # (input interval + future). else legacy: target = u[Ti-1 : Ti-1+Nt] (last-input IC + future).
                xtg = (u[:ntd] if tgt0 else u[Ti - 1: Ti - 1 + ntd]).astype(np.float32)
                if Nt > ntd:                                 # temporal padding: trailing model frames carry
                    xtg = np.concatenate([xtg, np.zeros((Nt - ntd,) + xtg.shape[1:], np.float32)])  # zeros,
                    # masked out of loss/eval by the tmask (same mechanism as pdearena_uncond valid_t)
                if gstats is not None:                       # GLOBAL per-family norm (fixed std → keeps energy)
                    gm, gs = gstats[fam]
                    xin = (xin - gm) / (gs + 1e-6); xtg = (xtg - gm) / (gs + 1e-6)
                elif inorm:                                  # per-sample instance norm (legacy)
                    xin, xtg = _instance_norm(xin, xtg)
                if fv is not None:                           # DUMMY SUPERVISION: unused slots -> const (NORMALIZED
                    for s in range(C_MAX):                   # space, both in+out) so the shared backbone gets a
                        if not np.isnan(fv[s]):              # CONSISTENT cross-family target (vs masking = garbage)
                            xin[..., s] = fv[s]; xtg[..., s] = fv[s]
                gm = (arr[i][0, :, :, gch:gch + 1].astype(np.float32) if gch is not None   # static boundary @t0
                      else np.ones((arr[i].shape[1], arr[i].shape[2], 1), np.float32))      # no geometry → Π=identity
                yield (xin, xtg, d, np.zeros(4, np.float32), cm, tm, gm)
    return gen


def _build_tmasks(datasets, Ti, Nt, tgt0, ntd=None):
    """Per-family TEMPORAL validity mask (Nt,) for the OUTPUT window — PROSE's data_mask_len. Families
    with a `valid_t` in SPEC (only pdearena_uncond=14, padded to t20) mask the padded output frames so
    they don't contribute to loss; all others are fully valid. Output frame j maps to global frame
    `base+j` (base = Ti-1 for legacy future-only target, 0 for ic_first full-trajectory target);
    frame j valid iff base+j < valid_t."""
    base = 0 if tgt0 else (Ti - 1)
    tmasks = {}
    for fi, ds in enumerate(datasets):
        vt = SPEC[ds].get("valid_t", None)
        m = np.ones(Nt, np.float32)
        if ntd is not None and ntd < Nt:                     # temporal padding: frames >= ntd are pad
            m[int(ntd):] = 0.0
        if vt is not None:
            for j in range(Nt):
                if base + j >= int(vt):
                    m[j] = 0.0
        tmasks[fi] = m
    return tmasks


def make_train_stream(cfg, global_batch, cache_dir, test_frac=0.1):
    """Streaming train dataset (infinite) + the materialized test arrays for eval."""
    Ti, Nt, t_num = cfg.T_in, cfg.Nt, cfg.t_num
    # data_nt DECOUPLES the loss/data horizon (DNT frames the stream yields) from the MODEL's ADA horizon
    # (cfg.Nt). AR-t5: model outputs cfg.Nt=6 frames per call (t=5) but the stream must still deliver the full
    # DNT=11 (10 futures) so the 2-segment AR training/eval can supervise the whole benchmark horizon. Unset →
    # DNT=Nt (single-shot, unchanged). Everything data-facing (tmasks/signature/_make_gen) uses DNT.
    DNT = int(getattr(cfg, "data_nt", 0)) or Nt
    NTD = int(getattr(cfg, "nt_data", 0)) or DNT            # data frames; DNT>NTD = temporally padded grid
    inorm = bool(getattr(cfg, "instance_norm", False))       # PROSE-style per-trajectory normalization
    gnorm = bool(getattr(cfg, "global_norm", False))         # Poseidon-style FIXED per-family normalization
    tgt0 = bool(getattr(cfg, "ic_first", False))             # full-trajectory supervision (target = u[0:Nt])
    train_plan, test_plan, descs, cmasks = build_plan(
        cfg.datasets, cfg.n_per, t_num, cache_dir, test_frac, getattr(cfg, "seed", 0))
    tmasks = _build_tmasks(cfg.datasets, Ti, DNT, tgt0, ntd=NTD)        # PROSE-style temporal validity mask (Nt,)
    # cfdbench_geom: move CFDBench's STATIC boundary/obstacle mask OUT of the shared scalar slot 2 (where it
    # COLLIDES with pdearena_ns/incom_ns smoke/particles — and CFDBench, being a fast-converging strong-gradient
    # family, can dominate that channel's shared representation) INTO the model's geometry mask (Π freeze-solid
    # + encoder geom input). The field then uses slots [0,1] (Vx,Vy) only; the boundary (native ch 2, static)
    # becomes the per-sample geom_mask. Loss/metric UNCHANGED (cfdbench c_mask slot2 was already 0). Off → every
    # family gets geom_mask=ones (identical to the no-geom default). gated; default OFF.
    _cfgeom = bool(getattr(cfg, "cfdbench_geom", False))
    geom_chs = {fi: (2 if (ds == "cfdbench" and _cfgeom) else None) for fi, ds in enumerate(cfg.datasets)}
    if _cfgeom:
        _ovr = lambda pl: [(f, fi, ([0, 1] if cfg.datasets[fi] == "cfdbench" else sl), ix) for (f, fi, sl, ix) in pl]
        train_plan, test_plan = _ovr(train_plan), _ovr(test_plan)
        print("  [stream] cfdbench_geom: CFDBench boundary (native ch2) -> geom_mask; field slots -> [0,1]", flush=True)
    # DUMMY SUPERVISION (Poseidon-style, opt-in): instead of MASKING a family's unused slots out of the loss
    # (c_mask=0 -> garbage, fragments the shared backbone across families), SUPERVISE them to a CONSISTENT
    # constant so every family shares the same target structure. density slot -> rho_const (incompressible
    # rho=const, ALIVE), other unused slots -> pad_const (0 = pad_zero / "dead", non-zero = ALIVE). Applied to
    # the TRAIN stream ONLY; the test/eval keeps the ORIGINAL c_mask (real channels) so the metric stays
    # directly comparable to the masking baseline. Fills are in NORMALIZED space (avoids the 0/0 blow-up of
    # normalizing a const with an absent family's zero stats).
    # restricted to the CROSS-FAMILY PHYSICS slots only (density=3, pressure=4) — NOT every unused slot.
    # Blanket "all unused -> const" wrongly forced FOREIGN slots (scalar-h on NS, SW's real velocity) to a
    # constant = many dead/wrong channels -> instability + polluted shared velocity. Here only `dummy_slots`
    # get a CONSISTENT target when a family LACKS them: density -> rho_const (incompressible rho=const, ALIVE),
    # pressure -> p_const (Poseidon incompressible convention p=0). Families that HAVE the slot (com_ns: real
    # rho+p) are untouched; all other unused slots stay MASKED (original c_mask=0). TRAIN cm only; eval cm orig.
    # SEPARATED-DENSITY (incomp_rho_slot): put the incompressible const-density marker in a DEDICATED slot
    # (needs C_MAX=7) so com_ns keeps the REAL density slot 3 to ITSELF — the shared decoder no longer
    # collapses to a "ρ=const majority" (4 families) over com_ns's varying ρ → com_ns should recover, while
    # incompressible families still get the clean const-rho auxiliary in the separate slot.
    _incomp_slot = int(getattr(cfg, "incomp_rho_slot", -1))
    _dummy_slots = [int(s) for s in getattr(cfg, "dummy_slots", [])]
    _fillval = {3: float(getattr(cfg, "rho_const", 0.0)), 4: float(getattr(cfg, "p_const", 0.0))}
    train_cmasks, fills = cmasks, None
    if _incomp_slot >= 0:
        train_cmasks, fills = {}, {}
        _rhoc = float(getattr(cfg, "rho_const", 1.0))
        for fi, ds in enumerate(cfg.datasets):
            used = set(SPEC[ds]["slots"]); cm = _padcm(SPEC[ds]["c_mask"]); fv = np.full(C_MAX, np.nan, np.float32)
            if 3 not in used:                       # incompressible (no real density) -> const-rho in SEPARATE slot
                cm[_incomp_slot] = 1.0; fv[_incomp_slot] = _rhoc
            train_cmasks[fi], fills[fi] = cm, fv
        print(f"  [stream] SEPARATED-DENSITY: incompressible const-rho={_rhoc} -> slot {_incomp_slot} "
              f"(com_ns keeps real density slot 3 alone) — train cm supervised, eval cm original", flush=True)
    elif _dummy_slots:
        train_cmasks, fills = {}, {}
        for fi, ds in enumerate(cfg.datasets):
            used = set(SPEC[ds]["slots"]); cm = _padcm(SPEC[ds]["c_mask"])
            fv = np.full(C_MAX, np.nan, np.float32)
            for s in _dummy_slots:
                if s not in used:
                    cm[s] = 1.0
                    fv[s] = _fillval.get(s, 0.0)
            train_cmasks[fi], fills[fi] = cm, fv
        print(f"  [stream] DUMMY-SUPERVISE slots {_dummy_slots} (density→{_fillval[3]} pressure→{_fillval[4]}) "
              f"only when family lacks them — train cm supervised, eval cm original", flush=True)
    gstats = None
    if gnorm:                                                # take precedence over inorm; per-sample OFF
        gstats = _global_stats(train_plan, Ti, int(getattr(cfg, "stat_samples", 64)))
        print(f"  [stream] GLOBAL per-family norm (fixed std) over {len(gstats)} families", flush=True)
    sig = (tf.TensorSpec((Ti, cfg.Nx, cfg.Nx, C_MAX), tf.float32),
           tf.TensorSpec((DNT, cfg.Nx, cfg.Nx, C_MAX), tf.float32),
           tf.TensorSpec((cfg.desc_dim,), tf.float32),
           tf.TensorSpec((4,), tf.float32),
           tf.TensorSpec((C_MAX,), tf.float32),
           tf.TensorSpec((DNT,), tf.float32),
           tf.TensorSpec((cfg.Nx, cfg.Nx, 1), tf.float32))   # per-sample geom_mask (fluid=1/solid=0; ones if none)
    # balanced_sampling: build ONE infinite stream PER FAMILY and interleave them with equal weight, so
    # EVERY batch is family-balanced. The single concatenated generator (else-branch) streams unit-by-unit
    # with only a shuffle_buf window — fine for small runs (units ~810), but for FULL data a family is one
    # giant unit (SWE = 1 .npy ~90k samples), so the model sees long mono-family stretches and a family
    # whose unit comes late is starved (observed: SWE 61% @1000 = barely trained). sample_from_datasets
    # fixes this — each family is drawn equally regardless of its #shards/#samples.
    buf = int(getattr(cfg, "shuffle_buf", 1024))
    if bool(getattr(cfg, "balanced_sampling", True)):
        fam_ids = sorted({u[1] for u in train_plan})
        per_fam = []
        for fid in fam_ids:
            sub = [u for u in train_plan if u[1] == fid]
            per_fam.append(
                tf.data.Dataset.from_generator(_make_gen(sub, Ti, DNT, descs, train_cmasks, tmasks, inorm, tgt0, gstats, fills, geom_chs=geom_chs, ntd=NTD),
                                               output_signature=sig)
                .shuffle(buf).repeat())
        w = [1.0 / len(per_fam)] * len(per_fam)
        ds = (tf.data.Dataset.sample_from_datasets(per_fam, weights=w, stop_on_empty_dataset=False)
              .batch(global_batch, drop_remainder=True).prefetch(tf.data.AUTOTUNE))
        print(f"  [stream] balanced sampling over {len(per_fam)} families (equal weight)", flush=True)
    else:
        ds = (tf.data.Dataset.from_generator(_make_gen(train_plan, Ti, DNT, descs, train_cmasks, tmasks, inorm, tgt0, gstats, fills, geom_chs=geom_chs, ntd=NTD), output_signature=sig)
              .shuffle(buf)
              .repeat()
              .batch(global_batch, drop_remainder=True)
              .prefetch(tf.data.AUTOTUNE))

    # test set → RAM. CAP per family (test_max_per_fam) so FULL-data runs don't try to materialize ~10%
    # of 100k trajectories (tens of GB → minutes of silent IO / OOM). mmap_mode='r' avoids loading a
    # whole big shard/.npy (e.g. pdearena 36GB) — only the needed rows are read. The cap is a no-op for
    # small runs (n900 → ~90 test/family < 120). 120/family ≈ 600 total is plenty for monitoring eval.
    test_cap = int(getattr(cfg, "test_max_per_fam", 120))
    xin, xtg, dsc, cf, cm, tmk, fam = [], [], [], [], [], [], []
    mean, std, geomte = [], [], []                               # per-sample per-channel input-window stats + geom_mask
    fam_count = {}
    for f, fi, slots, idxs in test_plan:
        if fam_count.get(fi, 0) >= test_cap:
            continue                                             # family budget hit → skip loading shard
        arr = np.load(f, mmap_mode="r")                          # lazy: read only the rows we take
        gch = geom_chs.get(fi)
        for i in idxs:
            if fam_count.get(fi, 0) >= test_cap:
                break
            geomte.append(np.asarray(arr[i][0, :, :, gch:gch + 1], np.float32) if gch is not None
                          else np.ones((arr.shape[2], arr.shape[3], 1), np.float32))   # static boundary @t0 / ones
            u = _to_6slot_one(np.asarray(arr[i]), slots)
            a = u[:Ti].astype(np.float32)
            b = (u[:NTD] if tgt0 else u[Ti - 1: Ti - 1 + NTD]).astype(np.float32)
            if DNT > NTD:
                b = np.concatenate([b, np.zeros((DNT - NTD,) + b.shape[1:], np.float32)])
            # capture the PHYSICAL per-channel mean/std (from input window) so eval can DENORMALIZE the
            # normalized pred/ref back to physical space → PROSE-style physical rel-L2. For non-inorm runs
            # the data is already physical → mean=0,std=1 makes denorm a no-op.
            if gstats is not None:                               # GLOBAL norm: same fixed family stats as train
                m_c, s_c = gstats[fi]
                a = (a - m_c) / (s_c + 1e-6); b = (b - m_c) / (s_c + 1e-6)
            elif inorm:
                m_c = a.mean(axis=(0, 1, 2)); s_c = a.std(axis=(0, 1, 2))
                a = (a - m_c) / (s_c + 1e-6); b = (b - m_c) / (s_c + 1e-6)
            else:
                m_c = np.zeros(a.shape[-1], np.float32); s_c = np.ones(a.shape[-1], np.float32)
            xin.append(a); xtg.append(b)
            mean.append(m_c.astype(np.float32)); std.append(s_c.astype(np.float32))
            dsc.append(descs[fi]); cf.append(np.zeros(4, np.float32)); cm.append(cmasks[fi])
            tmk.append(tmasks[fi]); fam.append(fi)
            fam_count[fi] = fam_count.get(fi, 0) + 1
    test = dict(xin=np.asarray(xin, np.float32), xtg=np.asarray(xtg, np.float32),
                desc=np.asarray(dsc, np.float32), coef=np.asarray(cf, np.float32),
                cm=np.asarray(cm, np.float32), tm=np.asarray(tmk, np.float32),
                fam=np.asarray(fam, np.int32),
                mean=np.asarray(mean, np.float32), std=np.asarray(std, np.float32),
                geom=np.asarray(geomte, np.float32),
                fam_names=list(cfg.datasets))
    print(f"  [stream] test materialized: {test['xin'].shape[0]} samples", flush=True)
    return ds, test
