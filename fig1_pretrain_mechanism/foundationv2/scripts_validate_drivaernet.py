"""One-off validation of the mesh_drivaernet adapter (run on a CPU job with /corpus mounted).

Extracts just 2 VTKs from one pressure zip into /tmp (does NOT touch /corpus), then exercises
the real adapter helpers: pyvista read -> _vtk_pressure -> normalize_coords, and a full
adapters.load through a hand-built ref. Prints shapes/stats so we can confirm before a full
materialize of ~10k designs.
"""
import glob
import os
import zipfile

import numpy as np

os.environ.setdefault("CORPUS_ROOT", "/corpus")
CORP = os.environ["CORPUS_ROOT"]
PRES = os.path.join(CORP, "raw", "mesh", "drivaernet", "pressure")
TMP = "/tmp/danval"

os.makedirs(TMP, exist_ok=True)
z = sorted(glob.glob(os.path.join(PRES, "*.zip")))[0]
print("ZIP", os.path.basename(z), flush=True)
with zipfile.ZipFile(z) as zf:
    mem = [m for m in zf.namelist() if m.endswith(".vtk")][:2]
    zf.extractall(TMP, members=mem)

import pyvista as pv  # noqa: E402

from data.adapters import _drivaernet_load, _vtk_pressure  # noqa: E402
from data.registry import FAMILIES  # noqa: E402
from data.schema import normalize_coords  # noqa: E402
import data.symbolic as SY  # noqa: E402

vtks = sorted(glob.glob(os.path.join(TMP, "PressureVTK", "*", "*.vtk")))
print("found", len(vtks), "vtk", flush=True)
for v in vtks[:2]:
    m = pv.read(v)
    p = _vtk_pressure(m)
    c, _ = normalize_coords(np.asarray(m.points))
    print(os.path.basename(v), "| N", m.points.shape[0],
          "| point_data", list(m.point_data.keys()),
          "| cell_data", list(m.cell_data.keys()), flush=True)
    print("   pressure mean %.3f std %.3f rng [%.2f, %.2f] | coord rng [%.3f, %.3f]"
          % (float(p.mean()), float(p.std()), float(p.min()), float(p.max()),
             float(c.min()), float(c.max())), flush=True)

# full adapter path through a hand-built ref (uid, vtk_path)
spec = FAMILIES["drivaernet_pressure"]
cond = SY.encode(spec)
s = _drivaernet_load(spec, vtks[0], cond)
s.validate()
print("UNIFIED sample: K", s.K, "mode", s.mode, "fields", s.fields.shape,
      "coords", s.coords.shape, "cmask", s.cmask.tolist(), flush=True)
print("DANVAL_OK", flush=True)
