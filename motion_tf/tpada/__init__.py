"""TP-ADA: tensor-product Anti-Derivative Approximator numerics (Phase 0).

Pure-numpy table precomputation and synthesis. All tables are constants at
model-build time; the training framework (TF/JAX/torch) only sees einsums
against fixed matrices.
"""
from .basis import Axis, AxisSpec, null_projector
from .synthesis import apply_bc, eval_grid, eval_points, project

__all__ = ["Axis", "AxisSpec", "null_projector", "project", "eval_grid",
           "eval_points", "apply_bc"]
