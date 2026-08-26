"""PHASE-2 (Poseidon transfer) data loader — SEPARATE from Phase-1 stream.py (both pipelines run).
Reads PDEgym data per family from EITHER a monolithic NetCDF (_assembled/<ds>.nc) OR a directory of native
.npy shards (_assembled/<ds>_shards/) — the shard path is FUSE-safe and is used when the monolithic .nc
corrupts on large FUSE writes (NS-Gauss: 82GB .nc had valid metadata but every data read threw 'NetCDF:
HDF error'; converted to shards via _nc_to_shards.py). Maps each family's physical channels into a unified
8-slot layout, per-dataset GLOBAL channel norm, streams full 21-frame trajectories. all2all IC-sampling +
masked supervision is in train_ivp.py.

8-slot layout: 0:vx 1:vy 2:tracer 3:density 4:pressure 5:energy 6:geom-mask* 7:forcing/source* (*input-only).
  CE  'data'     5ch = [density,vx,vy,pressure,energy] -> slots [3,0,1,4,5]
  NS  'velocity' 3ch = [vx,vy,tracer]                  -> slots [0,1,2]
"""
from __future__ import annotations
import os, glob
import numpy as np
import tensorflow as tf
from netCDF4 import Dataset

NCH = 8
ASSEMBLED = os.environ.get("POSE_ASSEMBLED", "/code-vol/data/poseidon/_assembled")
N_TEST = 240

POSEIDON_SPEC = {
    "CE-RP":    dict(var="data",     ch2slot=[3, 0, 1, 4, 5], c_mask=[1, 1, 0, 1, 1, 1, 0, 0]),
    "CE-KH":    dict(var="data",     ch2slot=[3, 0, 1, 4, 5], c_mask=[1, 1, 0, 1, 1, 1, 0, 0]),
    "CE-CRP":   dict(var="data",     ch2slot=[3, 0, 1, 4, 5], c_mask=[1, 1, 0, 1, 1, 1, 0, 0]),
    "CE-Gauss": dict(var="data",     ch2slot=[3, 0, 1, 4, 5], c_mask=[1, 1, 0, 1, 1, 1, 0, 0]),
    "NS-Sines": dict(var="velocity", ch2slot=[0, 1, 2],       c_mask=[1, 1, 1, 0, 0, 0, 0, 0]),
    "NS-Gauss": dict(var="velocity", ch2slot=[0, 1, 2],       c_mask=[1, 1, 1, 0, 0, 0, 0, 0]),
    # --- DOWNSTREAM transfer tasks (Poseidon's 15-task suite) ---
    "NS-PwC":   dict(var="velocity", ch2slot=[0, 1, 2],       c_mask=[1, 1, 1, 0, 0, 0, 0, 0]),  # incompressible NS, in-family
    "CE-RM":    dict(var="solution", ch2slot=[3, 0, 1, 4, 5], c_mask=[1, 1, 0, 1, 1, 1, 0, 0]),  # compressible Euler Richtmyer-Meshkov (shock)
    "ACE":      dict(var="solution", ch2slot=[2],             c_mask=[0, 0, 1, 0, 0, 0, 0, 0], no_channel=True),  # Allen-Cahn scalar -> tracer slot
    # Wave-Layer: 2-channel assembled = [solution(t) -> slot2 (the QoI), c(x) broadcast -> slot6 (static
    # wave-speed, FED + supervised-constant so it survives the rollout & informs the bank's ∇/Δ). Hyperbolic.
    "Wave-Layer": dict(var="solution", ch2slot=[2, 6],        c_mask=[0, 0, 1, 0, 0, 0, 1, 0], feed_only=[6]),  # slot6=c is FED (input) but NOT scored/trained
    # Poisson-Gauss: STEADY elliptic (−Δu=f, no time axis; vars source/solution). Framed as a 2-frame
    # trajectory: ch0=QoI (0 at frame0, solution at frame1) -> slot2; ch1=source broadcast (static input,
    # like Wave's c) -> slot7 (forcing, fed-not-scored). Source/solution stay in SEPARATE slots — their
    # physical scales differ ~200x and each gets its own norm stats. Use FT_IC_FRAC=0 (IC=frame0 only).
    "Poisson-Gauss": dict(steady=("source", "solution"), ch2slot=[2, 7],
                          c_mask=[0, 0, 1, 0, 0, 0, 0, 1], feed_only=[7]),
}


class _NCReader:
    """single-sample reader for a monolithic .nc (big slices throw HDF errors over FUSE; read one at a time)."""
    def __init__(self, ds):
        self.nc = Dataset(os.path.join(ASSEMBLED, f"{ds}.nc"))
        self.v = self.nc.variables[POSEIDON_SPEC[ds]["var"]]
        self.no_ch = bool(POSEIDON_SPEC[ds].get("no_channel"))       # scalar field: (sample,time,x,y), no channel dim
        self.n = self.v.shape[0]; self.C = 1 if self.no_ch else self.v.shape[2]
    def __getitem__(self, idx):
        a = np.asarray(self.v[idx]).astype(np.float32)               # (T,C,H,W) or (T,H,W) if no channel
        return a[:, None] if self.no_ch else a                       # -> (T,C,H,W)


class _ShardReader:
    """reader over native .npy shards (n_i,21,C,128,128); FUSE-safe. Caches the last-touched shard."""
    def __init__(self, ds):
        self.dir = os.path.join(ASSEMBLED, f"{ds}_shards")
        self.files = sorted(glob.glob(os.path.join(self.dir, "shard_*.npy")))
        self.index = []                                              # global idx -> (file_i, local_i)
        for fi, f in enumerate(self.files):
            ni = int(np.load(f, mmap_mode="r").shape[0])
            self.index += [(fi, j) for j in range(ni)]
        self.n = len(self.index)
        self.C = int(np.load(self.files[0], mmap_mode="r").shape[2])
        self._cf, self._ca = -1, None
    def __getitem__(self, idx):
        fi, j = self.index[idx]
        if fi != self._cf:
            self._ca = np.load(self.files[fi], mmap_mode="r"); self._cf = fi
        return np.asarray(self._ca[j]).astype(np.float32)            # (21,C,H,W)


class _SteadyNCReader:
    """time-independent (elliptic) task: two vars (source, solution), no time dim. Builds a 2-frame
    native trajectory (2,2,H,W): frame0 = [QoI=0, source], frame1 = [QoI=solution, source] — the source
    is broadcast static input (Wave-c pattern), the QoI appears at the 'final time'."""
    def __init__(self, ds):
        sp = POSEIDON_SPEC[ds]
        self.nc = Dataset(os.path.join(ASSEMBLED, f"{ds}.nc"))
        self.vsrc = self.nc.variables[sp["steady"][0]]
        self.vsol = self.nc.variables[sp["steady"][1]]
        self.n = self.vsrc.shape[0]; self.C = 2
    def __getitem__(self, idx):
        src = np.asarray(self.vsrc[idx]).astype(np.float32)          # (H,W)
        sol = np.asarray(self.vsol[idx]).astype(np.float32)
        z = np.zeros_like(sol)
        return np.stack([np.stack([z, src]), np.stack([sol, src])])  # (T=2,C=2,H,W)


def _open(ds):
    """prefer shards (FUSE-safe) over the monolithic .nc."""
    if "steady" in POSEIDON_SPEC[ds]:
        return _SteadyNCReader(ds)
    if os.path.isdir(os.path.join(ASSEMBLED, f"{ds}_shards")):
        return _ShardReader(ds)
    return _NCReader(ds)


def _global_stats(reader, ds, n_stat=64):
    spec = POSEIDON_SPEC[ds]
    Cnat = reader.C
    ssum = np.zeros(Cnat, np.float64); ssq = np.zeros(Cnat, np.float64); cnt = 0
    for idx in range(min(n_stat, reader.n - N_TEST)):
        a = reader[idx].astype(np.float64)                          # (21,Cnat,H,W)
        ssum += a.sum(axis=(0, 2, 3)); ssq += (a ** 2).sum(axis=(0, 2, 3))
        cnt += a.shape[0] * a.shape[2] * a.shape[3]
    m = (ssum / cnt).astype(np.float32)
    s = (np.sqrt(np.maximum(ssq / cnt - (ssum / cnt) ** 2, 0.0)) + 1e-6).astype(np.float32)
    return m, s


def _to_slots(traj_nat, ds):
    spec = POSEIDON_SPEC[ds]
    T, Cnat, H, W = traj_nat.shape
    out = np.zeros((T, H, W, NCH), np.float32)
    for ci, slot in enumerate(spec["ch2slot"]):
        out[..., slot] = traj_nat[:, ci]
    return out


def build_poseidon_stream(cfg, global_batch, test_frac=None):
    req = list(cfg.datasets)
    datasets, specs, stats, counts = [], {}, {}, {}
    for ds in req:
        try:
            sp = POSEIDON_SPEC[ds]
            r = _open(ds)
            m_nat, s_nat = _global_stats(r, ds)
            cnt = r.n
        except Exception as e:
            print(f"  [poseidon] !! SKIP {ds} (unreadable: {type(e).__name__}: {e})", flush=True)
            continue
        specs[ds] = sp
        m8, s8 = np.zeros(NCH, np.float32), np.ones(NCH, np.float32)
        for ci, slot in enumerate(sp["ch2slot"]):
            m8[slot] = m_nat[ci]; s8[slot] = s_nat[ci]
        stats[ds] = (m8, s8); counts[ds] = cnt; datasets.append(ds)
        src = "shards" if os.path.isdir(os.path.join(ASSEMBLED, f"{ds}_shards")) else "nc"
        print(f"  [poseidon] {ds}: {cnt} traj, src={src}", flush=True)
    if not datasets:
        raise RuntimeError("no readable Poseidon datasets")
    fam_names = datasets

    # DOWNSTREAM N-shot transfer: n_shot>0 -> train on the FIRST n_shot trajectories, reserve exactly the
    # last `test_per_fam` as the (fixed) test set. Legacy pretraining (no n_shot) keeps the 240 reserve.
    n_shot = int(getattr(cfg, "n_shot", 0))
    cap = int(getattr(cfg, "test_per_fam", 64))
    reserve = cap if n_shot > 0 else N_TEST
    if n_shot > 0:
        print(f"  [poseidon] N-SHOT={n_shot} per family, test reserve={cap} (last {cap} traj)", flush=True)
    # uv_only: supervise VELOCITY (slots 0,1) as the task (matches Poseidon's NS-PwC=velocity task).
    # pad_zero (0-in-0-out): ALSO supervise the dataset's UNUSED slots to 0 — Poseidon supervises its dummy
    # channels (p->0, rho->const) so the full output stays VALID/stable (vs masking, which leaves them as
    # garbage that corrupts autoregressive re-feed). This actively "turns off" the dead channels -> AR-capable.
    _uv = bool(getattr(cfg, "uv_only", False))
    _pad0 = bool(getattr(cfg, "pad_zero", False))
    # pad_const: unused slots -> a NORMALIZED constant (e.g. 1.0) in+out, SUPERVISED (identity of unity).
    # Unlike pad_zero (->0, which "kills" the channel: zero activation/gradient), a non-zero const keeps the
    # channel ALIVE while still being trivially satisfiable -> valid full-channel output without dead slots.
    _pad_c = float(getattr(cfg, "pad_const", 0.0))
    def _unused(ds):
        used = set(POSEIDON_SPEC[ds]["ch2slot"])
        return [s for s in range(NCH) if s not in used]
    def _eff_cm(ds, base_cm):
        cm = (np.array([1, 1, 0, 0, 0, 0, 0, 0], np.float32) if _uv else np.array(base_cm, np.float32))
        if _pad0 or _pad_c:                                  # supervise unused slots (->0 or ->const)
            for s in _unused(ds):
                cm[s] = 1.0
        return cm
    def _fill_pad_const(trajn, ds):                          # override unused slots to const in NORMALIZED space
        if _pad_c:
            for s in _unused(ds):
                trajn[..., s] = _pad_c
        return trajn
    if _uv:   print(f"  [poseidon] UV-ONLY (velocity task){' + PAD-ZERO (0-in-0-out on unused slots)' if _pad0 else ''}{f' + PAD-CONST={_pad_c} (const-in-const-out, channel ALIVE)' if _pad_c else ''}", flush=True)
    # rho_const: fill the DENSITY slot (3) with a constant for families that don't have density (NS = incompressible
    # => physically ρ=const, like Poseidon's dummy rho). Combined with pad_zero, this is const-in-const-out
    # (supervised), not masking. Density present (CE) is untouched.
    _rho_c = float(getattr(cfg, "rho_const", 0.0))
    if _rho_c and _pad0:
        print(f"  [poseidon] RHO-CONST: incompressible density slot[3] -> {_rho_c} (const-in-const-out)", flush=True)
    def _fill_const(traj, ds):
        if _rho_c and 3 not in POSEIDON_SPEC[ds]["ch2slot"]:
            traj[..., 3] = _rho_c
        return traj

    # Parallel reader: the per-sample .nc read is a single serial cephfs hit; one generator thread can't
    # feed multi-GPU (GPUs starve, ~19 samples/s on 4xH100). Each `shard` opens its OWN reader handles and
    # uses its OWN reshuffle order (seeded by shard) -> K shards interleaved with num_parallel_calls=K run
    # K generator threads concurrently; netCDF/HDF5 releases the GIL during I/O so cephfs reads parallelize.
    # Each shard yields the same family-balanced round-robin, so the union preserves the sampling distribution.
    def _gen(shard=0):
        shard = int(shard)
        readers = {ds: _open(ds) for ds in datasets}
        ntr = {ds: (min(n_shot, counts[ds] - reserve) if n_shot > 0 else counts[ds] - reserve)
               for ds in datasets}
        rng = np.random.default_rng(int(getattr(cfg, "seed", 0)) * 100003 + shard)
        order = {ds: rng.permutation(ntr[ds]) for ds in datasets}
        ptr = {ds: 0 for ds in datasets}
        while True:
            for fi, ds in enumerate(datasets):
                if ptr[ds] >= ntr[ds]:
                    order[ds] = rng.permutation(ntr[ds]); ptr[ds] = 0
                idx = int(order[ds][ptr[ds]]); ptr[ds] += 1
                traj = _fill_const(_to_slots(readers[ds][idx], ds), ds)                # (21,H,W,8)
                m8, s8 = stats[ds]
                feed_cm = _eff_cm(ds, specs[ds]["c_mask"])            # channels FED (kept non-zero in input)
                score_cm = feed_cm.copy()                            # channels SCORED/TRAINED (loss + eval)
                for _fs in specs[ds].get("feed_only", []):           # fed-not-scored (e.g. Wave c) -> drop from loss/eval
                    score_cm[_fs] = 0.0
                trajn = ((traj - m8[None, None, None, :]) / s8[None, None, None, :]) * feed_cm[None, None, None, :]
                trajn = _fill_pad_const(trajn, ds)
                yield (trajn.astype(np.float32), score_cm, m8, s8, np.int32(fi))

    H = cfg.Nx
    _NTD = int(getattr(cfg, "nt_data", 0)) or cfg.Nt      # data frames (Nt may be padded past the data)
    sig = (tf.TensorSpec((_NTD, H, H, NCH), tf.float32), tf.TensorSpec((NCH,), tf.float32),
           tf.TensorSpec((NCH,), tf.float32), tf.TensorSpec((NCH,), tf.float32), tf.TensorSpec((), tf.int32))
    K = int(getattr(cfg, "reader_threads", 1))
    if K > 1:
        ds_tf = (tf.data.Dataset.range(K)
                 .interleave(lambda s: tf.data.Dataset.from_generator(_gen, output_signature=sig, args=(s,)),
                             cycle_length=K, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
                 .batch(global_batch, drop_remainder=True).prefetch(tf.data.AUTOTUNE))
        print(f"  [poseidon] PARALLEL READER: {K} interleaved generator threads", flush=True)
    else:
        ds_tf = (tf.data.Dataset.from_generator(_gen, output_signature=sig, args=(0,))
                 .batch(global_batch, drop_remainder=True).prefetch(4))

    xs, cms, ms, ss, fams = [], [], [], [], []
    for fi, dsn in enumerate(datasets):
        r = _open(dsn); m8, s8 = stats[dsn]
        feed_cm = _eff_cm(dsn, specs[dsn]["c_mask"])
        score_cm = feed_cm.copy()
        for _fs in specs[dsn].get("feed_only", []):
            score_cm[_fs] = 0.0
        for idx in range(r.n - reserve, r.n - reserve + cap):
            traj = _fill_const(_to_slots(r[idx], dsn), dsn)
            trajn = ((traj - m8[None, None, None, :]) / s8[None, None, None, :]) * feed_cm[None, None, None, :]
            trajn = _fill_pad_const(trajn, dsn)
            xs.append(trajn.astype(np.float32)); cms.append(score_cm); ms.append(m8); ss.append(s8); fams.append(fi)
    test = dict(traj=np.asarray(xs, np.float32), cm=np.asarray(cms, np.float32),
                mean=np.asarray(ms, np.float32), std=np.asarray(ss, np.float32),
                fam=np.asarray(fams, np.int32), fam_names=fam_names)
    print(f"  [poseidon] test materialized: {test['traj'].shape}", flush=True)
    return ds_tf, test
