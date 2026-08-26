"""Phase-1 ADA — ALTERNATIVE B: "W-history recursion". Drop the Euler STATE feedback (z stays z0 → no
redundant double-integration of W, no baked-in 1st-order scheme) but KEEP a recursion in W-SPACE: each panel's
operator is conditioned on the PREVIOUS panels' W (w_feedback / wfb_proj), so W_i = g(z0, physics_bank, W_<i).
The single ADA basis integration of {W_i} produces the solution. This is the conceptually-clean recursion the
double-integration critique points to: recursion provides state/history dependence (helps the nonlinear
PDEArena families) WITHOUT the redundant Euler trajectory. Sequential (not batchable) → slow, like baseline.
A/B vs pw_batched (no recursion) and vs the recursive baseline (Euler z-recursion).
Launch (2 GPU): bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_wrec
"""
import os
from .phase1_eff_ada import Cfg as _Base


class Cfg(_Base):
    parallel_w = True        # z FIXED at z0 (no Euler state-feedback / no double integration)
    w_feedback = True        # recursion in W-space: panel i sees previous panels' W (wfb_proj)
    step_embed = True         # panel-index conditioning (panel 0 cp=0 → needs explicit index signal)
    pw_batched = False        # W-history recursion is sequential — cannot batch (gated off; _pwb needs no w_feedback)
    swap_memory = False
    save_dir = "results/" + os.environ.get("WREC_SAVE", "phase1_int_ada_152m_6fam_wrec")
