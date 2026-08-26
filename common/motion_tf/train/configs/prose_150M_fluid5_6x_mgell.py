"""PHASE-1 Task-1 ABLATION — MULTIGRID elliptic operator (inductive bias, NOT capacity).

NS is slow to converge partly because its NON-LOCAL coupling (the elliptic ∇·u=0 pressure/Poisson) is only
APPROXIMATED in our model by the EllipticExpert's dilated-conv(1,2,4,8)+global-MEAN (a single global scalar
— a weak proxy). This run swaps that proxy for a real MULTIGRID V-CYCLE (avg-pool restrict 32→16→8→4,
bilinear prolong + fine-level skip, periodic smoothers) — the actual numerical method for elliptic Poisson.
NON-spectral (real-space hierarchy → orthogonal to ADA's TIME-Fourier basis; no "double sin" clash that an
FNO spatial-spectral layer would cause), O(N), and deliberately PARAMETER-LEAN (V-cycle width capped at 384
vs expert_hidden=960) so any gain is the STRUCTURE, not capacity (contrast: enc_expand helped but added +41M).

ONLY the elliptic expert's global-coupling structure changes (loss/norm/arch otherwise = baseline) → clean
isolation vs the Task-1 curve (overall 11.60% @15k; pdearena_ns 25.19, incom_ns 5.98). Target: the
incompressible/turbulent families (pdearena_ns, incom_ns) converge faster/lower via true global coupling."""
from .prose_150M_fluid5_6x import Cfg as _Base


class Cfg(_Base):
    elliptic_multigrid = True
    elliptic_mg_levels = 3     # 32→16→8→4 (coarsest 4² ≈ global receptive field)
    save_dir = "results/prose_150M_fluid5_6x_mgell"
