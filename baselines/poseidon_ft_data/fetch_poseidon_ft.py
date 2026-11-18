"""REVISION-R1 (user 2026-10-23): fetch only the trajectories needed to fine-tune / validate / test on the 15 Poseidon
downstream tasks (14 HF repos; NS-Tracer-PwC = channel 2 of NS-PwC).

Split rule (scOT/problems/base.py): train = first n trajectories, val = [N_max-N_val-N_test, N_max-N_test),
test = [N_max-N_test, N_max). Chunks are concatenated in LEXICOGRAPHIC file order (assemble_data.py).
  - repos with <= 4 chunks or one file: download everything; assemble with the repo's own assemble_data.py.
  - large repos (NS-*, FNS-KF, CE-RPUI): download the first and the last chunk in that order and write
    <task>_sub.nc = [chunk-0 trajectories] + [val] + [test], with <task>_sub.json giving the original indices.
The total sample count is checked against N_max from the file sizes before any subset is written.
Output: /eu/data/poseidon_ft/<task>/ ; existing full files in /eu/data/poseidon/_assembled are reused (symlink).
"""
import os, sys, json, shutil, subprocess
from huggingface_hub import HfApi, hf_hub_download
import netCDF4

OUT = os.environ.get("OUT", "/eu/data/poseidon_ft"); RAW = os.environ.get("RAW", "/eu/data/poseidon_ft/_raw")
OLD = "/eu/data/poseidon/_assembled"
SPEC = {  # repo: (N_max, N_val, N_test, mode)
    "NS-PwC": (20000, 120, 240, "sub"), "NS-SVS": (20000, 120, 240, "sub"), "NS-BB": (20000, 120, 240, "sub"),
    "NS-SL": (40000, 120, 240, "sub"), "FNS-KF": (20000, 120, 240, "sub"), "CE-RPUI": (10000, 120, 240, "sub"),
    "CE-RM": (1260, 100, 130, "full"), "GCE-RT": (1260, 100, 130, "full"), "Wave-Gauss": (10512, 60, 240, "full"),
    "Wave-Layer": (10512, 60, 240, "full"), "ACE": (15000, 60, 240, "sub"), "SE-AF": (10869, 120, 240, "file"),
    "Poisson-Gauss": (20000, 120, 240, "file"), "Helmholtz": (19675, 128, 512, "file")}
api = HfApi()
tasks = sys.argv[1:] or list(SPEC)
def run(t):
    Nmax, Nv, Nt, mode = SPEC[t]; od = f"{OUT}/{t}"; os.makedirs(od, exist_ok=True)
    for f in os.listdir(od):                          # drop links to old files that do not hold N_max samples (10-23 fix)
        q = f"{od}/{f}"
        if os.path.islink(q) and f.endswith(".nc"):
            with netCDF4.Dataset(q) as nc: n_ = nc.dimensions["sample"].size
            if n_ < Nmax: print(f"  old {os.path.realpath(q)} has {n_} < {Nmax} samples: re-fetch", flush=True); os.remove(q)
    if any(f.endswith((".nc", ".h5")) for f in os.listdir(od)):
        print(f"=== {t}: output exists, skipped ({os.listdir(od)})", flush=True); return
    print(f"=== {t} mode={mode} N_max={Nmax} val={Nv} test={Nt}", flush=True)
    n_old = 0
    if os.path.exists(f"{OLD}/{t}.nc"):
        with netCDF4.Dataset(f"{OLD}/{t}.nc") as nc: n_old = nc.dimensions["sample"].size
    if n_old >= Nmax:
        os.symlink(f"{OLD}/{t}.nc", f"{od}/{t}.nc") if not os.path.exists(f"{od}/{t}.nc") else None
        print(f"  reuse existing {OLD}/{t}.nc", flush=True); return
    info = {f.path: f.size for f in api.list_repo_tree(f"camlab-ethz/{t}", repo_type="dataset", recursive=True) if hasattr(f, "size")}
    data = sorted(p for p in info if p.endswith((".nc", ".h5")) and not p.startswith("c_"))
    extra = sorted(p for p in info if p.startswith("c_")) + (["assemble_data.py"] if "assemble_data.py" in info else [])
    rd = f"{RAW}/{t}"; os.makedirs(rd, exist_ok=True)
    get = lambda p: hf_hub_download(f"camlab-ethz/{t}", p, repo_type="dataset", local_dir=rd)
    if mode == "file":
        for p in data: shutil.move(get(p), f"{od}/{p}")
        print(f"  done {data}", flush=True); return
    if mode == "full":
        for p in data + extra: get(p)
        subprocess.run([sys.executable, f"{rd}/assemble_data.py", "--input_dir", rd, "--output_file", f"{od}/{t}.nc"], check=True)
        with netCDF4.Dataset(f"{od}/{t}.nc") as nc: n = nc.dimensions["sample"].size
        print(f"  assembled {n} samples (N_max {Nmax})", flush=True); assert n >= Nmax
        shutil.rmtree(rd); return
    first, last = data[0], data[-1]; p0, pl = get(first), get(last)
    var = first.split("_")[0]
    with netCDF4.Dataset(p0) as a, netCDF4.Dataset(pl) as b:
        n0, nl = a.dimensions["sample"].size, b.dimensions["sample"].size
        per = info[first] / n0; total = sum(info[p] for p in data) / per
        print(f"  chunk0 {first} n={n0}; last {last} n={nl}; estimated total {total:.2f}", flush=True)
        assert abs(total - round(total)) < 0.05 and round(total) == Nmax, f"total {total} != N_max {Nmax}"
        off = Nmax - nl; vi = list(range(Nmax - Nv - Nt, Nmax)); assert vi[0] >= off
        with netCDF4.Dataset(f"{od}/{t}_sub.nc", "w") as o:
            for d, dim in a.dimensions.items(): o.createDimension(d, (n0 + len(vi)) if d == "sample" else dim.size)
            v = a.variables[var]; ov = o.createVariable(var, v.dtype, v.dimensions, chunksizes=(1, 1) + v.shape[2:])
            for i in range(n0): ov[i] = v[i]
            for j, g in enumerate(vi): ov[n0 + j] = b.variables[var][g - off]
    json.dump({"task": t, "var": var, "orig_N_max": Nmax, "N_val": Nv, "N_test": Nt, "n_train_avail": n0,
               "sub_N_max": n0 + Nv + Nt, "train_orig": [0, n0], "valtest_orig": [vi[0], vi[-1] + 1],
               "chunks": [first, last]}, open(f"{od}/{t}_sub.json", "w"), indent=1)
    print(f"  wrote {t}_sub.nc: {n0} train + {Nv} val + {Nt} test", flush=True); shutil.rmtree(rd)


failed = []
for t in tasks:
    try: run(t)
    except Exception as e:
        print(f"  FAILED {t}: {type(e).__name__}: {e}", flush=True); failed.append(t)
print("FETCH_DONE failed:", failed, flush=True)
