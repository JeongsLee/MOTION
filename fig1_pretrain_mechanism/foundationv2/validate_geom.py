"""Validate the canonical 8-channel geom emitted by mesh adapters.

For each mesh family: load ONE unit and print geom.shape plus per-channel
[min,max,mean]. Runs on the cluster (raw data at /corpus); do NOT run locally.
Canonical channels: [SDF, mask(loader-filled), nx, ny, nz, mean_curv, gauss_curv, interior].
"""
import numpy as np

from data import adapters
from data.registry import FAMILIES

_CHAN = ["sdf", "mask", "nx", "ny", "nz", "mean_c", "gauss_c", "interior"]

_FAMILIES = ["airfrans", "shapenet_car", "drivaernet_pressure",
             "geofno_airfoil", "geofno_pipe", "geofno_elasticity"]

for name in _FAMILIES:
    try:
        spec = FAMILIES[name]
        _, ref = next(iter(adapters.units(spec)))          # first unit ref
        sample = adapters.load(spec, ref)
        g = sample.geom
        if g is None:
            print(f"{name}: geom=None")
            continue
        g = np.asarray(g, np.float32)
        print(f"{name}: geom.shape={g.shape}")
        for c in range(g.shape[1]):
            col = g[:, c]
            lbl = _CHAN[c] if c < len(_CHAN) else f"ch{c}"
            print(f"    ch{c} {lbl:>9}: [{col.min():.4g}, {col.max():.4g}] "
                  f"mean={col.mean():.4g}")
    except Exception as e:
        print(f"{name}: SKIP ({type(e).__name__}: {e})")

print("VALIDATE_GEOM_DONE")
