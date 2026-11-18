#!/bin/bash
set -e
export PYTHONUNBUFFERED=1
pip install -q netCDF4 h5py 2>&1 | tail -1
ls /corpus/meta/splits | head -5; ls /corpus/meta/splits | wc -l
python -u /work/runs/poseidon_ft_data/prep_ft15.py $TASKS 2>&1 | tee /eu/data/poseidon_ft/prep_ft15.log
ls -la /eu/data/poseidon_ft/ft; ls /corpus/meta/splits | grep '^FT-'
python - <<'PY'
import json, glob, h5py, numpy as np
for m in sorted(glob.glob("/corpus/meta/splits/FT-*.json")):
    d = json.load(open(m)); print(m.split("/")[-1], d["counts"])
for f in sorted(glob.glob("/eu/data/poseidon_ft/ft/*.*")):
    if f.endswith(".json"): continue
    with h5py.File(f, "r") as h:
        ks = [k for k in h if hasattr(h[k], "shape")]
        print(f.split("/")[-1], {k: (h[k].shape, float(np.asarray(h[k][-1]).std())) for k in ks})
PY
