"""PHASE-1 Task-1 (UNIFIED baseline) + DUAL-BRANCH interpreter: local conv ∥ non-local multigrid.

Baseline = `prose_150M_fluid5_6x_unified` (the UNIFIED production arch — transformer=encoder + single
UnifiedExpert; the router collapsed, see [[router-collapse-unified-redesign]]). Its head currently
interprets the physics-operator bank with a dilated-conv (LOCAL) + a single global-MEAN scalar (a weak
non-local proxy). NS's non-locality is the elliptic/pressure (Poisson) coupling — per-timestep, global,
and the 1-time/input-only transformer encoder can't supply it.

This run swaps the weak global-mean for a MULTIGRID V-CYCLE non-local branch IN the UnifiedExpert head
(`unified_multigrid`), keeping the local dilated-conv branch → the head has BOTH local + non-local,
summed. Non-spectral (orthogonal to ADA time-Fourier), general (multigrid=elliptic/parabolic; local branch
keeps hyperbolic/shock & compressible → shouldn't break com_ns), per-panel on the evolving state. Only this
one change vs the unified baseline → clean isolation. Compare to the unified Task-1 train result
(prose_150M_fluid5_6x_unified ckpt_15000, PROSE-exact class-avg 6.31). Target: pdearena_ns + incom_ns drop
WITHOUT regressing com_ns (the failure mode of the earlier incompressible-biased changes)."""
from .prose_150M_fluid5_6x_unified import Cfg as _Base


class Cfg(_Base):
    unified_multigrid = True
    unified_mg_levels = 3      # 32→16→8→4 (coarsest ≈ global RF)
    unified_mg_hidden = 384    # lean V-cycle width (structure, not capacity)
    save_dir = "results/prose_150M_fluid5_6x_unified_mg"
