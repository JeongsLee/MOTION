"""REVISION-R1 10-24: evaluation files for the six Poseidon PRETRAINING families (check of the pretrained Poseidon-T
numbers). Only the last chunk (lexicographic order) is needed: it holds the official val + test rows. File layout as in
prep_ft15.py: [1 placeholder train row] + [val 120] + [test 240]; meta json with ntr = 1."""
import os, json, numpy as np, netCDF4
from huggingface_hub import HfApi, hf_hub_download
OUT = os.environ.get("PRE_OUT", "/eu/data/poseidon_ft/pre"); os.makedirs(OUT, exist_ok=True)
SPEC = {"NS-Sines": 20000, "NS-Gauss": 20000, "CE-RP": 10000, "CE-CRP": 10000, "CE-KH": 10000, "CE-Gauss": 10000}
NV, NT = 120, 240
api = HfApi()
for t, Nmax in SPEC.items():
    info = {f.path: f.size for f in api.list_repo_tree(f"camlab-ethz/{t}", repo_type="dataset", recursive=True) if hasattr(f, "size")}
    data = sorted(p for p in info if p.endswith(".nc")); first, last = data[0], data[-1]
    p0 = hf_hub_download(f"camlab-ethz/{t}", first, repo_type="dataset", local_dir=f"/tmp/raw/{t}")
    pl = hf_hub_download(f"camlab-ethz/{t}", last, repo_type="dataset", local_dir=f"/tmp/raw/{t}")
    with netCDF4.Dataset(p0) as a, netCDF4.Dataset(pl) as b:
        n0, nl = a.dimensions["sample"].size, b.dimensions["sample"].size
        total = sum(info[p] for p in data) / (info[first] / n0)
        assert abs(total - round(total)) < 0.05 and round(total) == Nmax, (t, total)
        var = first.split("_")[0]; v = b.variables[var]; off = Nmax - nl
        rows = [0] + list(range(Nmax - NV - NT - off, nl))             # placeholder + val + test (local rows)
        with netCDF4.Dataset(f"{OUT}/{t}.nc", "w") as o:
            for d, dim in b.dimensions.items(): o.createDimension(d, len(rows) if d == "sample" else dim.size)
            ov = o.createVariable(var, v.dtype, v.dimensions, chunksizes=(1, 1) + v.shape[2:])
            for j, r in enumerate(rows): ov[j] = v[r]
    json.dump({"task": t, "rows": len(rows), "ntr": 1, "n_val": NV, "n_test": NT, "orig_N_max": Nmax,
               "orig_valtest": [Nmax - NV - NT, Nmax], "chunk": last}, open(f"{OUT}/{t}.json", "w"))
    print(t, "n_last", nl, "rows", len(rows), flush=True)
    os.system(f"rm -rf /tmp/raw/{t}")
print("PREP_PRE6_DONE", flush=True)
