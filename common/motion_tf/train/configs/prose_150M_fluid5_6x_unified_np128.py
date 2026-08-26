"""PHASE-1 Task-1 (UNIFIED) — increase TEMPORAL resolution: N_p 64→128, n_modes/max_order 32→64 (stays at
Nyquist=N_p/2). Hypothesis (user): turbulence varies FAST in time; the 32-mode ADA basis may over-smooth the
TIME trajectory → NS suffers. More panels/modes let the trajectory follow faster NS dynamics. Distinct axis
from the feature bottleneck (n_latent). ~2× rollout compute/mem (N_p doubles); 32² latent so fits 80GB.
vs unified Task-1 baseline (@15k overall 10.43 / pdearena 22.10)."""
from .prose_150M_fluid5_6x_unified import Cfg as _Base


class Cfg(_Base):
    N_p = 128
    n_modes = 64
    max_order = 64
    save_dir = "results/prose_150M_fluid5_6x_unified_np128"
