"""Validate Poisson-Gauss source-conditioning end-to-end (CPU job, /corpus mounted).

Loads one Poisson-Gauss unit through the real adapter (solution -> target field, source ->
normalized geom conditioning channel), then runs loader.make_example to confirm the steady
example carries: zeroed input frames (no solution leak), the source in geom, and a solution
target. Prints shapes/stats.
"""
import os

import numpy as np

os.environ.setdefault("CORPUS_ROOT", "/corpus")

from data import adapters, loader  # noqa: E402
from data.registry import FAMILIES  # noqa: E402

spec = FAMILIES["Poisson-Gauss"]
print("nc_keys:", spec.nc_keys, "steady:", spec.time, flush=True)

uid, ref = next(iter(adapters.units(spec)))
s = adapters.load(spec, ref)
s.validate()
print("SAMPLE uid", uid, "| K", s.K, "mode", s.mode, "fields", s.fields.shape,
      "| geom(source)", None if s.geom is None else s.geom.shape,
      "| cmask", s.cmask.tolist(), flush=True)
print("  field(u) slot4 std %.3f | source geom std %.3f mean %.3f"
      % (float(s.fields[..., 4].std()),
         float(s.geom.std()), float(s.geom.mean())), flush=True)

# full training-example path
d = loader._as_dict(s)
ex = loader.make_example(d, t_in=10, n_colloc=2048, rng=np.random.default_rng(0))
print("EXAMPLE steady", ex["steady"], "| x_frames", ex["x_frames"].shape,
      "allzero", bool(np.all(ex["x_frames"] == 0)),
      "| geom", None if ex.get("geom") is None else ex["geom"].shape,
      "| y_q", ex["y_q"].shape, "| coords_q", ex["coords_q"].shape, flush=True)
print("  geom(source) matches Ncell:", ex["geom"].shape[0] == ex["coords_node"].shape[0],
      "| y_q std %.3f (normalized u)" % float(ex["y_q"][:, 4].std()), flush=True)
print("POISSON_VAL_OK", flush=True)
