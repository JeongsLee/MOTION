"""BubbleML PoolBoiling-Subcooled 2D, dt=0.1 release (15 trajectories, 2001 frames)
-> assembled grid_nchw h5 at the registry path (replaces the dt=1.0 build).

Native time stride (dt=0.1; the dt=1.0 build strided 10x on top of a 10x coarser
release = effective dt 10). The first 300 frames are dropped per the BubbleML docs
("drop the first 300 timesteps for dataset discretized to 0.1 unit" -- pre-boiling
transient). Spatial 2x downsample (384->192), fp32.
Channels out: [velx, vely, pressure, temperature, dfun] ; Twall parsed per file."""
import glob
import os

import h5py
import numpy as np

SRC = sorted(glob.glob("/corpus/raw/bubbleml_fine/PoolBoiling-SubCooled-FC72-2D-0.1/Twall-*.hdf5"))
OUT = "/corpus/raw/bubbleml/PoolBoil-Sub.nc"
BAK = "/corpus/raw/bubbleml/PoolBoil-Sub-5ch.nc"
assert len(SRC) == 15, SRC
KEYS = ("velx", "vely", "temperature", "dfun")   # NO pressure: unpredictable (rel 2.4) + off-protocol (Walrus 4-field canon)
DROP = 300

if os.path.exists(OUT) and not os.path.exists(BAK):
    os.rename(OUT, BAK)
    print(f"backed up dt=1.0 build -> {BAK}", flush=True)

twall = []
with h5py.File(OUT, "w") as g:
    ds = None
    for i, fp in enumerate(SRC):
        with h5py.File(fp, "r") as f:
            a = np.stack([np.asarray(f[k][DROP:, ::2, ::2], np.float32) for k in KEYS], 1)
        if ds is None:                                   # stream per trajectory: CPU-RAM safe
            ds = g.create_dataset("solution", shape=(len(SRC),) + a.shape, dtype=np.float32)
        ds[i] = a
        twall.append(float(os.path.basename(fp).split("-")[1].split(".")[0]))
        print(os.path.basename(fp), a.shape, "Twall", twall[-1], flush=True)
    g.create_dataset("twall", data=np.array(twall, np.float32))
    print("wrote", OUT, ds.shape, flush=True)   # inside the with-block: ds is invalid after close
print("CONVERT_DONE", flush=True)
