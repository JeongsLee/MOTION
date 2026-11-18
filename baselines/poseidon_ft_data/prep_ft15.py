"""REVISION-R1 (user 2026-10-23): build the fine-tuning files of the 15 Poseidon downstream tasks, shared by MOTION-IVP-S
and Poseidon-T, so that both models see IDENTICAL trajectories.

Each file /eu/data/poseidon_ft/ft/<task>.<nc|h5> holds, in this row order,
  [the first NTR training trajectories (original index 0..NTR-1)] + [official val] + [official test],
with the official scOT split val = [N_max-N_val-N_test, N_max-N_test), test = [N_max-N_test, N_max). Variable names and
layouts are the scOT ones, so scOT reads the file with N_max := NTR+N_val+N_test (its own start arithmetic then lands on
exactly these rows). Extra variables for the MOTION adapter (FNS-KF forcing, SE-AF rho/mask, Helmholtz u/a/bcf) are
added next to them. The MOTION split manifests (/corpus/meta/splits/FT-<task>.json) list the same rows.
"""
import os, sys, json
import numpy as np, netCDF4, h5py

SRC = "/eu/data/poseidon_ft"; DST = f"{SRC}/ft"; MAN = os.environ.get("MAN_DIR", "/corpus/meta/splits")
NTR = int(os.environ.get("NTR", "256"))
# task: (source file under SRC, N_max, N_val, N_test)
TASKS = {
    "NS-PwC": ("NS-PwC/NS-PwC_sub.nc", 20000, 120, 240), "NS-SVS": ("NS-SVS/NS-SVS_sub.nc", 20000, 120, 240),
    "NS-BB": ("NS-BB/NS-BB_sub.nc", 20000, 120, 240), "NS-SL": ("NS-SL/NS-SL_sub.nc", 40000, 120, 240),
    "FNS-KF": ("FNS-KF/FNS-KF_sub.nc", 20000, 120, 240), "CE-RPUI": ("CE-RPUI/CE-RPUI_sub.nc", 10000, 120, 240),
    "ACE": ("ACE/ACE_sub.nc", 15000, 60, 240),
    "CE-RM": ("CE-RM/CE-RM.nc", 1260, 100, 130), "GCE-RT": ("GCE-RT/GCE-RT.nc", 1260, 100, 130),
    "Wave-Gauss": ("Wave-Gauss/Wave-Gauss.nc", 10512, 60, 240), "Wave-Layer": ("Wave-Layer/Wave-Layer.nc", 10512, 60, 240),
    "SE-AF": ("SE-AF/SE-AF.nc", 10869, 120, 240), "Poisson-Gauss": ("Poisson-Gauss/Poisson-Gauss.nc", 20000, 120, 240),
    "Helmholtz": ("Helmholtz/Helmholtz.h5", 19675, 128, 512),
}
MANIFEST_ALIASES = {"NS-PwC": ["NS-Tracer-PwC"]}      # NS-Tracer-PwC = channel 2 of the NS-PwC file


def source_rows(t, Nmax, Nv, Nt):
    """Original indices wanted -> row indices in the source file."""
    orig = list(range(NTR)) + list(range(Nmax - Nv - Nt, Nmax))
    sub = TASKS[t][0].endswith("_sub.nc")
    if not sub:
        return orig, orig
    meta = json.load(open(f"{SRC}/{t}/{t}_sub.json"))
    n0, v0 = meta["n_train_avail"], meta["valtest_orig"][0]
    assert NTR <= n0 and meta["valtest_orig"] == [Nmax - Nv - Nt, Nmax], meta
    return orig, list(range(NTR)) + [n0 + (g - v0) for g in orig[NTR:]]


def copy_nc(src, dst, rows, extra=None):
    with netCDF4.Dataset(src) as a, netCDF4.Dataset(dst, "w") as o:
        sd = [v for v in a.variables.values() if v.ndim >= 3][0].dimensions[0]   # sample dimension name
        for d, dim in a.dimensions.items():
            o.createDimension(d, len(rows) if d == sd else dim.size)
        for k, v in a.variables.items():
            if v.dimensions and v.dimensions[0] == sd:
                ov = o.createVariable(k, v.dtype, v.dimensions, chunksizes=(1,) + tuple(min(s, 128) for s in v.shape[1:]))
                for j, r in enumerate(rows):
                    ov[j] = v[r]
            else:
                ov = o.createVariable(k, v.dtype, v.dimensions); ov[:] = v[:]
        if extra:
            extra(a, o, rows)


def extra_fnskf(a, o, rows):                          # scOT KolmogorovFlow forcing (ij grid on [0,1])
    x = np.linspace(0, 1, 128, dtype=np.float32)
    X, Y = np.meshgrid(x, x, indexing="ij"); f = (0.1 * np.sin(2 * np.pi * (X + Y))).astype(np.float32)
    sv = a.variables["solution"]; dd = (sv.dimensions[0], sv.dimensions[-2], sv.dimensions[-1])
    v = o.createVariable("forcing", "f4", dd, chunksizes=(1, 128, 128))
    for j in range(len(rows)):
        v[j] = f


def extra_seaf(a, o, rows):                           # scOT Airfoil: channel 0 = airfoil mask (input), 1 = density
    s = a.variables["solution"]; dd = (s.dimensions[0], s.dimensions[-2], s.dimensions[-1])
    m = o.createVariable("mask", "f4", dd, chunksizes=(1, 128, 128))
    r = o.createVariable("rho", "f4", dd, chunksizes=(1, 128, 128))
    for j, k in enumerate(rows):
        m[j] = s[k, 0]; r[j] = s[k, 1]


def copy_helmholtz(src, dst, rows):
    with h5py.File(src, "r") as a, h5py.File(dst, "w") as o:
        n = len(rows)
        U = o.create_dataset("u", (n, 128, 128), "f4"); A = o.create_dataset("a", (n, 128, 128), "f4")
        B = o.create_dataset("bcf", (n, 128, 128), "f4")
        for j, k in enumerate(rows):
            g = a[f"Sample_{k}"]; G = o.create_group(f"Sample_{j}")
            for key in g:
                G.create_dataset(key, data=g[key][()])
            U[j] = g["u"][()]; A[j] = g["a"][()] - 1.0          # scOT feeds a - 1
            B[j] = np.full((128, 128), float(np.array(g["bc"])), np.float32)
        return len(a.keys())


os.makedirs(DST, exist_ok=True); os.makedirs(MAN, exist_ok=True)
summary = {}
def prep(t):
    f, Nmax, Nv, Nt = TASKS[t]; src = f"{SRC}/{f}"
    orig, rows = source_rows(t, Nmax, Nv, Nt)
    ext = ".h5" if f.endswith(".h5") else ".nc"; dst = f"{DST}/{t}{ext}"
    if os.path.exists(dst):
        print(f"{t}: exists, skipped", flush=True)
    elif t == "Helmholtz":
        copy_helmholtz(src, dst, rows)
    else:
        copy_nc(src, dst, rows, {"FNS-KF": extra_fnskf, "SE-AF": extra_seaf}.get(t))
    n = len(rows)
    sp = {"train": [f"0000:{i}" for i in range(NTR)], "val": [f"0000:{i}" for i in range(NTR, NTR + Nv)],
          "test": [f"0000:{i}" for i in range(NTR + Nv, n)]}
    for name in [t] + MANIFEST_ALIASES.get(t, []):
        man = {"family": f"FT-{name}", "K": 2, "unit": "trajectory", "stage": "downstream",
               "source": f"official:poseidon (scOT N_max {Nmax}, val {Nv}, test {Nt}); train = first {NTR}",
               "seed": 0, "counts": {k: len(v) for k, v in sp.items()}, "splits": sp}
        json.dump(man, open(f"{MAN}/FT-{name}.json", "w"), indent=1)
    meta = {"task": t, "file": dst, "rows": n, "ntr": NTR, "n_val": Nv, "n_test": Nt, "orig_N_max": Nmax,
            "scot_N_max": n, "orig_index": orig}
    json.dump(meta, open(f"{DST}/{t}.json", "w"))
    summary[t] = {"rows": n, "size_GB": round(os.path.getsize(dst) / 1e9, 2)}
    print(t, summary[t], flush=True)


failed = []
for t in (sys.argv[1:] or list(TASKS)):
    try:
        prep(t)
    except Exception as e:
        print(f"FAILED {t}: {type(e).__name__}: {e}", flush=True); failed.append(t)
        for ext in (".nc", ".h5"):
            if os.path.exists(f"{DST}/{t}{ext}"): os.remove(f"{DST}/{t}{ext}")
print("PREP_DONE failed:", failed, json.dumps(summary), flush=True)
