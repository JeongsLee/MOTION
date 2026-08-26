"""Phase-1 Task-1 BEST recipe (114M unified-lup masking) + TIME-DERIVATIVE feature. The physics bank is all
SPATIAL (Δ,∇,−u·∇,ω,δ); time_deriv_feat adds the OBSERVED initial ∂u/∂t (last − prev input frame) as an
explicit encoder-input channel-block so the model gets the current rate-of-change directly. Only that flag
differs from the 114M masking best (10.49 / class-avg 6.35) → clean test of whether the time-derivative helps
(esp turbulent pdearena). Multi-frame T_in=10 so the derivative is available."""
from .prose_150M_fluid5_6x_unified import Cfg as _Base


class Cfg(_Base):
    time_deriv_feat = True
    save_dir = "results/prose_150M_fluid5_6x_unified_tderiv"
