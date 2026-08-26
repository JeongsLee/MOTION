"""RECOMMENDED final: lean UnifiedExpert + SHOCK. The lean unified (Δ,∇,−u·∇,ω,δ + dilated-conv head)
beat baseline-lup at 114M on 4/5 families — its ONE weakness was com_ns (compressible, where shock-capturing
matters). The full 8-expert sum (unified_all, 224M) UNDERPERFORMED the lean one → "add everything" is a loss.
So add ONLY shock, as a feature in the SAME single expert (not a separate summed expert): bank gains the
conservation-form −∇·F̂ (state-gated upwind). Stays ~single-expert/efficient. Drop reaction/wave/forcing
(redundant/no benefit). Goal: keep the lean unified's wins + close the com_ns gap with shock."""
from .prose_150M_fluid5_6x_unified import Cfg as _Base


class Cfg(_Base):
    # experts=("unified",), unified_expert=True, lambda_gate=0 inherited from the unified base
    unified_shock = True           # add the shock-capturing −∇·F̂ feature to the unified bank
    save_dir = "results/prose_150M_fluid5_6x_unified_shock"
