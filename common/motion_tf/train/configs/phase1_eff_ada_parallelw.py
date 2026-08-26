"""Phase-1 ADA — PARALLEL-W variant: drop the Euler state-feedback recurrence (z fixed at z0 for every
panel) so the N_p panels are independent → run in parallel, no swap_memory, no BPTT-through-z. Per-panel
variation comes from step_embed; nonlinearity/temporal history is delegated to the full-T_in transformer
encoder (SOTA-style). Operator bank evaluated at the IC; ADA Fourier basis integrates the panel W's.
Same phase1_eff_ada figure protocol otherwise (6 families, eff-batch 32, 1M views, same eval/metric) →
clean A/B vs the recursive baseline on BOTH speed and accuracy.
"""
import os
from .phase1_eff_ada import Cfg as _Base


class Cfg(_Base):
    parallel_w = True
    step_embed = True        # panel-index conditioning — REQUIRED so the N_p panels differ (no recurrence)
    w_feedback = False       # no feedback/recurrence (parallel_w is the non-feedback path)
    swap_memory = False      # not needed: panels independent, no BPTT through z
    save_dir = "results/" + os.environ.get("PW_SAVE", "phase1_int_ada_152m_6fam_parallelw")
