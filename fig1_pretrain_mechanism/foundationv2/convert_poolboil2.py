"""BubbleML 2.0 FC72 subcooled (20 traj, 512^2, 2001 frames) -> 4-field 256^2 nc for the
combo corpus (train generation; 1.0 stays the val/test generation). Same conventions as
convert_poolboil01: drop 300 transient frames (docs), ::2 spatial, channels
[velx, vely, temperature, dfun] matching registry slots (0,1,5,6)."""
import glob
import os

import h5py
import numpy as np

SRC = sorted(glob.glob("/corpus/raw/bubbleml2/PoolBoiling-Subcooled-FC72-2D/Twall_*.hdf5"))
OUT = "/corpus/raw/bubbleml/PoolBoil-Sub2.nc"
assert len(SRC) == 20, SRC
KEYS = ("velx", "vely", "temperature", "dfun")
DROP = 300

twall = []
with h5py.File(OUT, "w") as g:
    ds = None
    for i, fp in enumerate(SRC):
        with h5py.File(fp, "r") as f:
            a = np.stack([np.asarray(f[k][DROP:, ::2, ::2], np.float32) for k in KEYS], 1)
        if ds is None:
            ds = g.create_dataset("solution", shape=(len(SRC),) + a.shape, dtype=np.float32)
        ds[i] = a
        twall.append(float(os.path.basename(fp).replace("Twall_", "").split(".")[0]))
        print(os.path.basename(fp), a.shape, "Twall", twall[-1], flush=True)
    g.create_dataset("twall", data=np.array(twall, np.float32))
    print("wrote", OUT, ds.shape, flush=True)
print("CONVERT_DONE", flush=True)
