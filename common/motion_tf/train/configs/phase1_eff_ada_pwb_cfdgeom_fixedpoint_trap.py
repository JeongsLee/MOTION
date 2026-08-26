"""Phase-1 ADA — fixedpoint K=2 with TRAPEZOIDAL state reconstruction. The fixedpoint feeds the bank a state
reconstructed by integrating W; the original uses Euler/left-Riemann cumsum (1st order). This replaces it with
the TRAPEZOIDAL rule (2nd order): z(t_i) = z0 + dt·(Σ_{j<i}W + ½(W_i−W_0)). A more accurate explicit integration
→ the reconstructed evolving state (and thus the advection velocity the bank sees) is more accurate per Picard
sweep → potentially better PDEArena recovery at the same K. A/B vs the Euler fixedpoint (job-zumxp5nt59gb).
Launch (2 GPU): bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwb_cfdgeom_fixedpoint_trap
"""
import os
from .phase1_eff_ada_pwbatched_cfdgeom import Cfg as _Base


class Cfg(_Base):
    fixedpoint_iters = 2
    fixedpoint_trapezoid = True   # 2nd-order trapezoidal reconstruction (vs 1st-order Euler cumsum)
    save_dir = "results/" + os.environ.get("FPT_SAVE", "phase1_int_ada_152m_6fam_pwb_cfdgeom_fixedpoint_trap")
