"""PROSE-FD dataset loader with the unified 6-slot channel layout (Poseidon/PROSE-style).

Slots (semantic, fixed): [vx, vy, particle/smoke, density, pressure, scalar-h].
Each dataset fills its slots + a c_mask (loss only on active slots) + an operator-presence
descriptor (fed to the FamilyGate) + a geometry mask (fluid=1; ones if no geometry).
Input/output: T_in frames → predict the future window (non-autoregressive, matches PROSE-FD).

Native trajectory lengths differ per family (SWE ~101, PDEArena 56), so every family is
resampled to a common `t_num` frames by even index sampling — this is what lets multiple
families be concatenated into one batch.

descriptor dims (6): [convective, diffusive, reaction, elliptic, shock, hyperbolic-extra].

Datasets:
    shallow_water : slot 5,       mask [0,0,0,0,0,1], desc [1,0,0,0,1,0]  (advection + bores)
    pdearena_ns   : slots 0,1,2,  mask [1,1,1,0,0,0], desc [1,1,0,1,0,0]  (conv+visc+pressure)
    diff_react    : slots 2,3,    mask [0,0,1,1,0,0], desc [0,1,1,0,0,0]  (diffusion + reaction)
"""
from __future__ import annotations
import glob
import os

import h5py
import numpy as np

C_MAX = int(os.environ.get("PROSE_CMAX", "6"))   # env-gated: default 6 (task2/Phase-1 unchanged); set 7 for
                                                  # the SEPARATED-density experiment (com_ns rho=slot3 alone,
                                                  # incompressible const-rho marker -> slot 6). SPEC c_masks
                                                  # (6-wide) are zero-padded to C_MAX at use, so slot 6 is
                                                  # never a REAL/evaluated channel — only the dummy marker.
PROSE_ROOT = os.environ.get("PROSE_DATA_DIR", "/mnt/e/pdefoundation_data/prose")
SWE_PATH = os.path.join(PROSE_ROOT, "pdebench/2D/shallow-water/2D_rdb_NA_NA.h5")

# per-dataset spec: slot indices, c_mask (6,), operator-presence descriptor (6,), has_geometry
SPEC = {
    "shallow_water": dict(slots=[5],          c_mask=[0, 0, 0, 0, 0, 1], desc=[1, 0, 0, 0, 1, 0], geom=False),
    "pdearena_ns":   dict(slots=[0, 1, 2],    c_mask=[1, 1, 1, 0, 0, 0], desc=[1, 1, 0, 1, 0, 0], geom=False),
    # PDEArena NS UNCONDITIONED (NavierStokes-2D, 14-frame): same incompressible operators/slots as
    # pdearena_ns; the 6th PROSE fluid family. Native horizon 14 → built padded to t_num=20 but only the
    # first valid_t=14 frames are real (the rest are clip-repeat pad). valid_t drives the TEMPORAL MASK so
    # the padded output frames are excluded from the loss (PROSE's mixed_length + data_mask_len approach).
    "pdearena_uncond": dict(slots=[0, 1, 2],  c_mask=[1, 1, 1, 0, 0, 0], desc=[1, 1, 0, 1, 0, 0], geom=False, valid_t=14),
    "diff_react":    dict(slots=[2, 3],       c_mask=[0, 0, 1, 1, 0, 0], desc=[0, 1, 1, 0, 0, 0], geom=False),
    # compressible NS (PDEBench 2D CFD): fields [Vx,Vy,density,pressure] → slots [0,1,3,4].
    # operators: convective + diffusive(viscosity Eta) + shock + hyperbolic (compressible, no elliptic
    # pressure-projection — that's incompressible). The shock/hyper slots distinguish it from pdearena_ns.
    "com_ns":        dict(slots=[0, 1, 3, 4], c_mask=[1, 1, 0, 1, 1, 0], desc=[1, 1, 0, 0, 1, 1], geom=False),
    # incompressible NS (PDEBench ns_incom_inhom_2d, 512² → mean-pooled to 128): fields
    # [Vx,Vy,particles] → slots [0,1,2]. Same incompressible operators as pdearena_ns (conv+diff+
    # elliptic pressure-projection); a second incompressible-NS source for the PROSE benchmark.
    "incom_ns":      dict(slots=[0, 1, 2], c_mask=[1, 1, 1, 0, 0, 0], desc=[1, 1, 0, 1, 0, 0], geom=False),
    # CFDBench (incompressible NS, cavity/cylinder/tube): fields [Vx,Vy,boundary-mask] → slots [0,1,2].
    # The mask is STATIC geometry — kept as an input feature (model must see the obstacles) but EXCLUDED
    # from the loss (c_mask slot2=0), matching BCAT/PROSE-FD (dim:2). Incompressible operators
    # (conv+diff+elliptic), like incom_ns. The 6th PROSE/BCAT fluid family.
    "cfdbench":      dict(slots=[0, 1, 2], c_mask=[1, 1, 0, 0, 0, 0], desc=[1, 1, 0, 1, 0, 0], geom=False),
}

# DISJOINT-CHANNEL variant (env PROSE_DISJOINT=1; needs PROSE_CMAX=8): velocity stays SHARED at slots 0,1 (same
# physics across families → transfer preserved), but every family's NON-velocity scalar gets its OWN dedicated
# slot so different physics (smoke vs particles vs density vs height) NEVER collide in one channel — the shared
# slot-2 tracer was hard for the physics-based bank. slots: 0=vx,1=vy(shared); 2=pdearena_ns smoke;
# 3=incom particles; 4=com rho; 5=com pressure; 6=pdearena_uncond smoke; 7=SWE height. (cfdbench boundary →
# geom via cfdbench_geom, so its slot-2 is freed — no collision with pdearena smoke.) c_masks are 8-wide.
SPEC_DISJOINT = {
    "shallow_water":   dict(slots=[7],       c_mask=[0, 0, 0, 0, 0, 0, 0, 1], desc=[1, 0, 0, 0, 1, 0], geom=False),
    "pdearena_ns":     dict(slots=[0, 1, 2], c_mask=[1, 1, 1, 0, 0, 0, 0, 0], desc=[1, 1, 0, 1, 0, 0], geom=False),
    "pdearena_uncond": dict(slots=[0, 1, 6], c_mask=[1, 1, 0, 0, 0, 0, 1, 0], desc=[1, 1, 0, 1, 0, 0], geom=False, valid_t=14),
    "com_ns":          dict(slots=[0, 1, 4, 5], c_mask=[1, 1, 0, 0, 1, 1, 0, 0], desc=[1, 1, 0, 0, 1, 1], geom=False),
    "incom_ns":        dict(slots=[0, 1, 3], c_mask=[1, 1, 0, 1, 0, 0, 0, 0], desc=[1, 1, 0, 1, 0, 0], geom=False),
    "cfdbench":        dict(slots=[0, 1, 2], c_mask=[1, 1, 0, 0, 0, 0, 0, 0], desc=[1, 1, 0, 1, 0, 0], geom=False),
}
# PARTIAL-DISJOINT (env PROSE_DISJOINT=2; needs PROSE_CMAX=7): separate only GENUINELY-DIFFERENT physics, keep
# SIMILAR scalars shared. Full disjoint over-separated (split the two SMOKES → lost transfer). Here: velocity
# shared 0,1; passive SMOKE shared at slot 2 (pdearena_ns AND pdearena_uncond — same physics); incom PARTICLES
# get their OWN slot 3 (different enough — the @250 incom gain); com density/pressure → 4,5; SWE height → 6.
SPEC_DISJOINT2 = {
    "shallow_water":   dict(slots=[6],       c_mask=[0, 0, 0, 0, 0, 0, 1], desc=[1, 0, 0, 0, 1, 0], geom=False),
    "pdearena_ns":     dict(slots=[0, 1, 2], c_mask=[1, 1, 1, 0, 0, 0, 0], desc=[1, 1, 0, 1, 0, 0], geom=False),
    "pdearena_uncond": dict(slots=[0, 1, 2], c_mask=[1, 1, 1, 0, 0, 0, 0], desc=[1, 1, 0, 1, 0, 0], geom=False, valid_t=14),
    "com_ns":          dict(slots=[0, 1, 4, 5], c_mask=[1, 1, 0, 0, 1, 1, 0], desc=[1, 1, 0, 0, 1, 1], geom=False),
    "incom_ns":        dict(slots=[0, 1, 3], c_mask=[1, 1, 0, 1, 0, 0, 0], desc=[1, 1, 0, 1, 0, 0], geom=False),
    "cfdbench":        dict(slots=[0, 1, 2], c_mask=[1, 1, 0, 0, 0, 0, 0], desc=[1, 1, 0, 1, 0, 0], geom=False),
}
if os.environ.get("PROSE_DISJOINT") == "2":
    SPEC = SPEC_DISJOINT2
    print(f"  [prose] PARTIAL-DISJOINT ON (C_MAX={C_MAX}): velocity+smoke shared; particles/rho/p/height separate", flush=True)
elif os.environ.get("PROSE_DISJOINT"):
    SPEC = SPEC_DISJOINT
    print(f"  [prose] DISJOINT channels ON (C_MAX={C_MAX}): velocity shared 0,1; scalars per-family disjoint", flush=True)

# compressible NS is huge (each native file ~55GB / ~10k traj); the others are 1k-3.7k. To keep the
# multi-family mix balanced AND fit a 13GB prep node, take only this many traj per source file
# (4 valid 128-res Rand files × 4 (M,Eta) settings → ~6k diverse traj, comparable to pdearena's 3.7k).
COMNS_PER_FILE = 1500
INCOM_WIN = 10        # consecutive (dt=1) windows extracted per incom trajectory (1000 steps) — fixes
                      # the old single coarse-dt sample: 89 files × 4 traj × 10 ≈ 3.5k samples, all dt=1
# sliding-window counts per trajectory for the other long-T families → dt=1 + more samples (PROSE-style).
# com_ns (T=21) stays 1 (already short / plenty of traj); the rest spread a few consecutive windows.
SWE_WIN = 4           # SWE T=101 → 4 windows × ~1000 traj
DR_WIN = 4            # diff-react T=101 → 4 windows × ~1000 traj
PDEARENA_WIN = 3      # pdearena T=56 → 3 windows × ~3.7k traj


def _resample(T, t_num):
    """t_num CONSECUTIVE frame indices (dt=1, matching PROSE's t_step=1). Previously this evenly-spread
    t_num indices over the native length T — fine for short T (com_ns T=21 → dt~1) but for long
    trajectories it made the dt FAR too coarse: incom T=1000 → stride ~53 → consecutive cache frames are
    53 native steps apart → turbulent NS changes completely between frames → UNFORECASTABLE (incom stuck
    at ~61% for us AND ~80% for official PROSE on our cache). SWE/DR (T=101) → dt~5, pdearena (T=56) →
    dt~3 — all coarser than PROSE. Consecutive (dt=1) makes every family a proper short-horizon forecast,
    matching PROSE. (T<t_num → clip-repeats the last frame; we no longer mix 14-frame PDEArena.)"""
    return np.clip(np.arange(t_num), 0, T - 1)


def _consec_windows(T, t_num, max_win):
    """List of up to max_win CONSECUTIVE t_num-frame index windows, start-points evenly spaced over the
    usable range [0, T-t_num]. For long trajectories (incom T=1000) this yields many small-dt samples
    from one trajectory (fixes BOTH the dt AND the tiny sample count, ~340 → thousands)."""
    last = max(0, T - t_num)
    if last == 0 or max_win <= 1:
        return [np.clip(np.arange(t_num), 0, T - 1)]
    starts = np.unique(np.round(np.linspace(0, last, max_win)).astype(np.int64))
    return [np.arange(s, s + t_num) for s in starts]


def _to_6slot(data_dim, slots):
    """(n,T,128,128,d) → (n,T,128,128,6) placing channels into `slots`."""
    n, T, H, W, d = data_dim.shape
    out = np.zeros((n, T, H, W, C_MAX), np.float32)
    for j, s in enumerate(slots):
        out[..., s] = data_dim[..., j]
    return out


def load_swe(n_total, t_num, path=SWE_PATH):
    out = []
    with h5py.File(path, "r") as f:
        for k in sorted(f.keys()):
            if len(out) >= n_total:
                break
            d = np.asarray(f[k]["data"])           # (T,128,128,1)
            for idx in _consec_windows(d.shape[0], t_num, SWE_WIN):   # dt=1 sliding windows
                if len(out) >= n_total:
                    break
                out.append(d[idx].astype(np.float32))
    return np.stack(out, 0)[:n_total]


def load_pdearena(n_total, t_num, root=None, subdir="NavierStokes-2D-conditoned"):
    """Aggregate (vx,vy,u-smoke) trajectories across PDEArena NS files → (n,t_num,128,128,3).
    PDEARENA_ROOT env lets this read from the volume directly (files are large contiguous arrays,
    so direct volume reads are fine — unlike SWE's many small groups which need a local copy).
    subdir picks the set: "NavierStokes-2D-conditoned" (pdearena_ns; 56-frame, 116 files) or
    "NavierStokes-2D" (pdearena_uncond; 14-frame, 156 files @ 25 traj) — both 128², same fields.
    NOTE: for the 14-frame uncond set call with t_num<=14 (PROSE uses T0=10/T=4 there)."""
    root = root or os.environ.get("PDEARENA_ROOT") or os.path.join(PROSE_ROOT, "pdearena")
    fs = sorted(glob.glob(os.path.join(root, subdir, "*.h5")))
    out = []
    for p in fs:
        if len(out) >= n_total:
            break
        with h5py.File(p, "r") as f:
            grp = list(f.keys())[0]                # train / valid / test
            g = f[grp]
            T = g["vx"].shape[1]
            idx = _resample(T, t_num)
            # ONE bulk read per field (whole dataset → numpy), not per-trajectory — object-storage
            # FUSE has high per-read latency, so 3 reads/file ≫ faster than 96 reads/file.
            vx = np.asarray(g["vx"])[:, idx]       # (ntraj,t_num,128,128)
            vy = np.asarray(g["vy"])[:, idx]
            uu = np.asarray(g["u"])[:, idx]
            for i in range(vx.shape[0]):
                if len(out) >= n_total:
                    break
                out.append(np.stack([vx[i], vy[i], uu[i]], axis=-1).astype(np.float32))
    if not out:
        raise FileNotFoundError(f"no PDEArena files under {root}")
    return np.stack(out, 0)


def load_pdearena_uncond(n_total, t_num, root=None):
    """PDEArena NS UNCONDITIONED (NavierStokes-2D, 14-frame) → (n,t_num,128,128,3). 6th PROSE fluid
    family. t_num<=14 (the native horizon; PROSE uses T0=10/T=4 here)."""
    return load_pdearena(n_total, t_num, root, subdir="NavierStokes-2D")


def load_diffreact(n_total, t_num, root=None):
    """PDEBench 2D diffusion-reaction (FitzHugh-Nagumo): 1000 groups, data (T=101,128,128,2)
    = (activator, inhibitor) → (n,t_num,128,128,2)."""
    root = root or os.path.join(PROSE_ROOT, "pdebench/2D/diffusion-reaction")
    path = os.path.join(root, "2D_diff-react_NA_NA.h5")
    out = []
    with h5py.File(path, "r") as f:
        for k in sorted(f.keys()):
            if len(out) >= n_total:
                break
            d = np.asarray(f[k]["data"])           # (T,128,128,2)
            for idx in _consec_windows(d.shape[0], t_num, DR_WIN):   # dt=1 sliding windows
                if len(out) >= n_total:
                    break
                out.append(d[idx].astype(np.float32))
    return np.stack(out, 0)[:n_total]


def _comns_files(root=None):
    """Valid 128-res compressible-NS source files (skip 512-res, .part, and truncated <1MB stubs)."""
    root = root or os.path.join(PROSE_ROOT, "pdebench/2D/CFD")
    fs = sorted(glob.glob(os.path.join(root, "*", "*.hdf5")))
    out = []
    for p in fs:
        if "512" in os.path.basename(p) or p.endswith(".part"):
            continue
        try:
            if os.path.getsize(p) < 1_000_000:             # the inviscid 1e-08 128 file is a 96B stub
                continue
        except OSError:
            continue
        out.append(p)
    return out


def _comns_fields(g):
    """Return (Vx,Vy,density,pressure) h5 datasets, tolerant of capitalization."""
    keys = {k.lower(): k for k in g.keys()}
    return tuple(g[keys[k]] for k in ("vx", "vy", "density", "pressure"))


def load_com_ns(n_total, t_num, root=None):
    """Compressible NS → (n,t_num,128,128,4) = [Vx,Vy,density,pressure]. Reads only COMNS_PER_FILE
    traj per source file (slice read — never the whole 55GB file) so it fits a small-RAM node and
    keeps the multi-family mix balanced."""
    out = []
    for p in _comns_files(root):
        if len(out) >= n_total:
            break
        take = min(COMNS_PER_FILE, n_total - len(out))
        try:
            with h5py.File(p, "r") as f:
                g = f                                       # CFD fields live at the file root
                vx, vy, rho, pr = _comns_fields(g)
                T = vx.shape[1]; idx = _resample(T, t_num)
                vx = np.asarray(vx[:take])[:, idx]; vy = np.asarray(vy[:take])[:, idx]
                rho = np.asarray(rho[:take])[:, idx]; pr = np.asarray(pr[:take])[:, idx]
            out.append(np.stack([vx, vy, rho, pr], axis=-1).astype(np.float32))
        except (OSError, KeyError):
            continue
    if not out:
        raise FileNotFoundError("no valid compressible-NS files (2D/CFD)")
    return np.concatenate(out, 0)[:n_total]


def _incom_files(root=None):
    """Valid incompressible-NS source files (skip .part / truncated stubs)."""
    root = root or os.path.join(PROSE_ROOT, "pdebench/2D/NS_incom")
    fs = sorted(glob.glob(os.path.join(root, "ns_incom_inhom_2d_512-*.h5")))
    return [p for p in fs if not p.endswith(".part") and os.path.getsize(p) > 1_000_000]


def _pool4(a):
    """Mean-pool the two trailing-spatial 512 dims by 4 → 128 (PROSE convert: 512²→128²). a:(...,512,512,C)."""
    s = a.shape
    return a.reshape(s[:-3] + (128, 4, 128, 4, s[-1])).mean(axis=(-4, -2))


def load_incom_ns(n_total, t_num, root=None):
    """PDEBench incompressible NS → (n,t_num,128,128,3) = [Vx,Vy,particles]. Each file = 4 traj of
    1000 steps at 512²; we read ONLY the t_num resampled timesteps per traj (h5 slice, ~42MB) and
    mean-pool 512→128 — so the giant 512 files never load whole (RAM-safe)."""
    out = []
    for p in _incom_files(root):
        if len(out) >= n_total:
            break
        try:
            with h5py.File(p, "r") as f:
                vel, par = f["velocity"], f["particles"]
                ntraj, T = vel.shape[0], vel.shape[1]
                wins = _consec_windows(T, t_num, INCOM_WIN)   # consecutive (dt=1) sliding windows
                for i in range(ntraj):
                    for idx in wins:
                        if len(out) >= n_total:
                            break
                        a = np.concatenate([np.asarray(vel[i, idx]), np.asarray(par[i, idx])], axis=-1)
                        out.append(_pool4(a).astype(np.float32))     # (t_num,128,128,3)
        except (OSError, KeyError):
            continue
    if not out:
        raise FileNotFoundError("no valid incompressible-NS files (2D/NS_incom)")
    return np.stack(out, 0)[:n_total]


_LOADERS = {
    "shallow_water": lambda n, t: load_swe(n, t),
    "pdearena_ns":   lambda n, t: load_pdearena(n, t),
    "pdearena_uncond": lambda n, t: load_pdearena_uncond(n, t),
    "diff_react":    lambda n, t: load_diffreact(n, t),
    "com_ns":        lambda n, t: load_com_ns(n, t),
    "incom_ns":      lambda n, t: load_incom_ns(n, t),
}


def _write_incomns_shards(n_total, t_num, shard_dir, root=None):
    """Write incompressible NS as one shard per source file (4 traj each, ~15MB) — reads only the
    t_num resampled timesteps per traj + mean-pools 512→128, so the 512 files never load whole."""
    os.makedirs(shard_dir, exist_ok=True)
    files = _incom_files(root)
    if not files:
        raise FileNotFoundError("no valid incompressible-NS files (2D/NS_incom)")
    written, si = 0, 0
    for p in files:
        if written >= n_total:
            break
        sp = os.path.join(shard_dir, f"shard_{si:04d}.npy"); si += 1
        if os.path.exists(sp):
            written += int(_npy_shape(sp)[0]); continue
        try:
            with h5py.File(p, "r") as f:
                vel, par = f["velocity"], f["particles"]
                ntraj, T = vel.shape[0], vel.shape[1]
                wins = _consec_windows(T, t_num, INCOM_WIN)   # many CONSECUTIVE (dt=1) windows per traj
                arr = []
                for i in range(ntraj):
                    for idx in wins:
                        if written + len(arr) >= n_total:
                            break
                        a = np.concatenate([np.asarray(vel[i, idx]), np.asarray(par[i, idx])], axis=-1)
                        arr.append(_pool4(a).astype(np.float32))   # (t_num,128,128,3), consecutive frames
            if not arr:
                continue
            arr = np.ascontiguousarray(np.stack(arr, 0))
            np.save(sp, arr); written += arr.shape[0]
            print(f"  [incom_ns] {os.path.basename(p)} -> shard_{si-1:04d} ({written} traj)", flush=True)
        except (OSError, KeyError) as e:
            print(f"  [incom_ns] skip {os.path.basename(p)} ({e})", flush=True); continue
    with open(os.path.join(shard_dir, "_DONE"), "w") as f:
        f.write(str(written))
    return written


def _write_comns_shards(n_total, t_num, shard_dir, root=None):
    """Write compressible NS as shards — one sub-shard per ~750-traj chunk (≤~0.8GB, FUSE-safe), 2 per
    source file so every (M,Eta) physics setting appears in BOTH the train and test shard-split. Reads
    each chunk by h5 slice (never the whole 55GB file) → small-RAM safe. Resumes on existing shards."""
    os.makedirs(shard_dir, exist_ok=True)
    CHUNK = 750
    files = _comns_files(root)
    if not files:
        raise FileNotFoundError("no valid compressible-NS files (2D/CFD)")
    written, si = 0, 0
    for p in files:
        if written >= n_total:
            break
        per_file = min(COMNS_PER_FILE, n_total - written)
        try:
            with h5py.File(p, "r") as f:
                vx, vy, rho, pr = _comns_fields(f)
                navail = vx.shape[0]; T = vx.shape[1]; idx = _resample(T, t_num)
                per_file = min(per_file, navail)
                for start in range(0, per_file, CHUNK):
                    end = min(start + CHUNK, per_file)
                    sp = os.path.join(shard_dir, f"shard_{si:04d}.npy"); si += 1
                    if os.path.exists(sp):
                        written += int(_npy_shape(sp)[0]); continue
                    a = np.stack([np.asarray(vx[start:end])[:, idx], np.asarray(vy[start:end])[:, idx],
                                  np.asarray(rho[start:end])[:, idx], np.asarray(pr[start:end])[:, idx]],
                                 axis=-1).astype(np.float32)
                    np.save(sp, np.ascontiguousarray(a))
                    written += a.shape[0]
                    print(f"  [com_ns] {os.path.basename(p)} [{start}:{end}] -> shard_{si-1:04d} ({written} traj)", flush=True)
        except (OSError, KeyError) as e:
            print(f"  [com_ns] skip {os.path.basename(p)} ({e})", flush=True); continue
    # --- Turb (512-res) → mean-pool 512→128 (PROSE/BCAT use Rand + Turb for com_ns). Appended as extra
    #     shards (resume-safe: existing Rand shards untouched). Smaller chunk — 512² is 16× the 128² data. ---
    troot = root or os.path.join(PROSE_ROOT, "pdebench/2D/CFD")
    turb = [p for p in sorted(glob.glob(os.path.join(troot, "*", "*Turb*512*.hdf5")))
            if not p.endswith(".part") and os.path.getsize(p) > 1_000_000]
    TCHUNK = 60
    for p in turb:
        if written >= n_total:
            break
        try:
            with h5py.File(p, "r") as f:
                vx, vy, rho, pr = _comns_fields(f)
                navail = vx.shape[0]; T = vx.shape[1]; idx = _resample(T, t_num)
                per_file = min(COMNS_PER_FILE, navail, n_total - written)
                for start in range(0, per_file, TCHUNK):
                    end = min(start + TCHUNK, per_file)
                    sp = os.path.join(shard_dir, f"shard_{si:04d}.npy"); si += 1
                    if os.path.exists(sp):
                        written += int(_npy_shape(sp)[0]); continue
                    a = np.stack([_pool4(np.asarray(vx[start:end])[:, idx][..., None])[..., 0],
                                  _pool4(np.asarray(vy[start:end])[:, idx][..., None])[..., 0],
                                  _pool4(np.asarray(rho[start:end])[:, idx][..., None])[..., 0],
                                  _pool4(np.asarray(pr[start:end])[:, idx][..., None])[..., 0]],
                                 axis=-1).astype(np.float32)
                    np.save(sp, np.ascontiguousarray(a))
                    written += a.shape[0]
                    print(f"  [com_ns TURB] {os.path.basename(p)} [{start}:{end}] -> shard_{si-1:04d} ({written} traj)", flush=True)
        except (OSError, KeyError) as e:
            print(f"  [com_ns TURB] skip {os.path.basename(p)} ({e})", flush=True); continue
    with open(os.path.join(shard_dir, "_DONE"), "w") as f:
        f.write(str(written))
    return written


def _write_pdearena_shards(n_total, t_num, shard_dir, root=None, workers=8,
                           subdir="NavierStokes-2D-conditoned", max_win=None):
    """Write the PDEArena native cache as per-source-file SHARDS (~130MB each), NOT one giant .npy:
    a 45GB single-file write to the object-storage FUSE volume TRUNCATES (~9GB silently lost → the
    header/data mismatch that crashed training). Small shards write reliably, read/copy in parallel,
    and resume (skip shards already present). A `_DONE` marker records completeness.
    subdir selects conditioned (56-frame) vs uncond (NavierStokes-2D, 14-frame); max_win overrides
    PDEARENA_WIN (uncond is 14-frame → use max_win=1: the whole trajectory is one t_num<=14 window)."""
    from concurrent.futures import ThreadPoolExecutor
    root = root or os.environ.get("PDEARENA_ROOT") or os.path.join(PROSE_ROOT, "pdearena")
    if max_win is None:
        max_win = PDEARENA_WIN
    fs = sorted(glob.glob(os.path.join(root, subdir, "*.h5")))
    if not fs:
        raise FileNotFoundError(f"no PDEArena files under {root}")
    os.makedirs(shard_dir, exist_ok=True)

    def _meta(p):                                          # h5 metadata only; skip corrupt + non-128²
        try:
            with h5py.File(p, "r") as f:
                g = f[list(f.keys())[0]]; sh = g["vx"].shape
                if sh[2] != 128 or sh[3] != 128:           # PDEArena mixes 64² and 128² files
                    return None
                return (p, int(sh[0]), int(sh[1]))
        except (OSError, KeyError):
            return None

    with ThreadPoolExecutor(max_workers=workers) as ex:    # parallel metadata scan
        meta = [m for m in ex.map(_meta, fs) if m is not None]
    n_bad = len(fs) - len(meta)
    if n_bad:
        print(f"  [pdearena] skipped {n_bad} corrupt/truncated/non-128px file(s)", flush=True)
    sel, run = [], 0
    for p, ntraj, T in meta:
        if run >= n_total:
            break
        take = min(ntraj, n_total - run)
        sel.append((p, take, T)); run += take

    def _proc(item):                                       # write one shard (parallel), → rows written
        i, (p, take, T) = item
        sp = os.path.join(shard_dir, f"shard_{i:04d}.npy")
        if os.path.exists(sp):                             # resume: already written
            return int(_npy_shape(sp)[0])
        wins = _consec_windows(T, t_num, max_win)          # dt=1 sliding windows
        try:
            with h5py.File(p, "r") as f:
                g = f[list(f.keys())[0]]
                vx = np.asarray(g["vx"][:take]); vy = np.asarray(g["vy"][:take]); uu = np.asarray(g["u"][:take])
            samples = [np.stack([vx[:, w], vy[:, w], uu[:, w]], axis=-1) for w in wins]
            arr = np.ascontiguousarray(np.concatenate(samples, 0), dtype=np.float32)  # (take*nwin,t_num,128,128,3)
        except (OSError, KeyError):
            return 0
        np.save(sp, arr)                                   # ~130MB → reliable on object storage
        return int(arr.shape[0])

    items = list(enumerate(sel))
    w = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i in range(0, len(items), workers):
            w += sum(ex.map(_proc, items[i:i + workers]))
            print(f"  [pdearena] {min(i + workers, len(items))}/{len(items)} files ({w} traj)", flush=True)
    with open(os.path.join(shard_dir, "_DONE"), "w") as f:
        f.write(str(w))
    return w


def _write_pdearena_uncond_shards(n_total, t_num, shard_dir, root=None, workers=8):
    """PDEArena NS UNCONDITIONED (NavierStokes-2D, 14-frame): per-file shards via the shared writer.
    14-frame → max_win=1 (one t_num<=14 window per traj, no sliding). Build at t_num<=14."""
    return _write_pdearena_shards(n_total, t_num, shard_dir, root=root, workers=workers,
                                  subdir="NavierStokes-2D", max_win=1)


def _cached_native(dataset, n_total, t_num, cache_dir=None):
    """Return a family's NATIVE-channel array from the on-disk cache (single .npy for SWE/DR; a
    directory of shards for pdearena). Full load (no mmap — FUSE mmap is unreliable); training copies
    the cache to LOCAL first so a plain load on the big-RAM node is fine."""
    if cache_dir:
        path = _ensure_cache(dataset, n_total, t_num, cache_dir)
        if os.path.isdir(path):                            # pdearena shards → load + concat in order
            shards = sorted(glob.glob(os.path.join(path, "shard_*.npy")))
            return np.concatenate([np.load(s) for s in shards], axis=0)
        return np.load(path)
    return _LOADERS[dataset](n_total, t_num)


def _npy_shape(fp):
    """Read just the .npy header (no data) → shape. Header-only read is FUSE-safe (unlike mmap)."""
    import numpy.lib.format as fmt
    with open(fp, "rb") as f:
        fmt.read_magic(f)
        shape, _, _ = fmt.read_array_header_1_0(f)
    return shape


def _ensure_cache(dataset, n_total, t_num, cache_dir):
    """Write the cache if absent; return its path (single .npy for SWE/DR, shard DIR for pdearena).
    Does NOT load the array — a 13GB CPU prep node only writes + reads headers."""
    os.makedirs(cache_dir, exist_ok=True)
    if dataset in ("pdearena_ns", "pdearena_uncond", "com_ns", "incom_ns"):   # sharded (giant single .npy truncates)
        sd = os.path.join(cache_dir, f"{dataset}_n{n_total}_t{t_num}_shards")
        if not os.path.exists(os.path.join(sd, "_DONE")):
            writer = {"pdearena_ns": _write_pdearena_shards, "pdearena_uncond": _write_pdearena_uncond_shards,
                      "com_ns": _write_comns_shards, "incom_ns": _write_incomns_shards}[dataset]
            total = writer(n_total, t_num, sd)
            print(f"  [cache] wrote {sd}  ({total} traj, sharded)", flush=True)
        return sd
    fp = os.path.join(cache_dir, f"{dataset}_n{n_total}_t{t_num}.npy")
    if not os.path.exists(fp):
        arr = _LOADERS[dataset](n_total, t_num)
        np.save(fp, arr)
        print(f"  [cache] wrote {fp}  {arr.shape}  {os.path.getsize(fp)/1e9:.2f}GB", flush=True)
    return fp


def _load_one(dataset, n_total, t_num, cache_dir=None):
    return _to_6slot(_cached_native(dataset, n_total, t_num, cache_dir), SPEC[dataset]["slots"])


def build(dataset="shallow_water", n_total=400, t_num=20, t_step=5,
          split=(0.8, 0.1, 0.1), seed=0):
    """Single-family. Returns u (n,t_num,128,128,6), c_mask (6,), desc (6,), geom_mask, splits."""
    spec = SPEC[dataset]
    u = _load_one(dataset, n_total, t_num)
    n = u.shape[0]
    rng = np.random.default_rng(seed)
    u = u[rng.permutation(n)]
    n_tr = int(split[0] * n); n_va = int(split[1] * n)
    geom = np.ones((n, 128, 128, 1), np.float32) if not spec["geom"] else None
    return {
        "u": u, "c_mask": np.array(spec["c_mask"], np.float32),
        "desc": np.array(spec["desc"], np.float32), "geom_mask": geom,
        "tr": slice(0, n_tr), "va": slice(n_tr, n_tr + n_va), "te": slice(n_tr + n_va, n),
        "dataset": dataset,
    }


def build_multi(datasets, n_per, t_num=20, split=(0.8, 0.1, 0.1), seed=0, cache_dir=None):
    """Multi-family. Loads each family (n_per samples), concatenates, and emits PER-SAMPLE
    c_mask (N,6) and desc (N,6). Splits are stratified per family (each family's first 80%
    → train, etc.) so val/test cover every family. Returns index arrays (not slices).
    cache_dir: reuse/populate per-family native .npy caches (see _cached_native)."""
    us, cms, descs, fams, tr, va, te = [], [], [], [], [], [], []
    off = 0
    for fi, ds in enumerate(datasets):
        u = _load_one(ds, n_per, t_num, cache_dir)
        n = u.shape[0]
        spec = SPEC[ds]
        us.append(u)
        cms.append(np.tile(np.array(spec["c_mask"], np.float32), (n, 1)))
        descs.append(np.tile(np.array(spec["desc"], np.float32), (n, 1)))
        fams.append(np.full(n, fi, np.int32))
        n_tr = int(split[0] * n); n_va = int(split[1] * n)
        idx = off + np.arange(n)
        tr.append(idx[:n_tr]); va.append(idx[n_tr:n_tr + n_va]); te.append(idx[n_tr + n_va:])
        off += n
        print(f"  [build_multi] {ds}: {n} samples", flush=True)
    u = np.concatenate(us, 0)
    c_mask = np.concatenate(cms, 0); desc = np.concatenate(descs, 0); fam = np.concatenate(fams, 0)
    N = u.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N); inv = np.argsort(perm)
    u = u[perm]; c_mask = c_mask[perm]; desc = desc[perm]; fam = fam[perm]
    remap = lambda a: inv[np.concatenate(a)]            # old global idx → shuffled position
    geom = np.ones((N, 128, 128, 1), np.float32)
    return {
        "u": u, "c_mask": c_mask, "desc": desc, "geom_mask": geom, "fam": fam,
        "tr": remap(tr), "va": remap(va), "te": remap(te),
        "datasets": list(datasets), "fam_names": list(datasets),
    }
