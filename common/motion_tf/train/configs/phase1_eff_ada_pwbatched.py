"""Phase-1 ADA — PARALLEL-W + BATCHED panels (speed variant of phase1_eff_ada_parallelw).
Same no-recurrence semantics (z fixed at z0, per-panel variation only via step_embed), but the N_p panels
are folded into the batch dim and the unified expert is called ONCE on (B·N_p, lat, lat, ·) instead of an
N_p-iteration while_loop. The latent grid is tiny (32²) → one fused batched conv beats 64 small panel
kernels (launch overhead + GPU under-utilization). MATH-IDENTICAL to parallel_w; this is purely the speed
fix. Clean A/B vs the recursive 4.43% baseline (accuracy) and vs the parallel_w while_loop (speed).
Launch (2 GPU): bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwbatched
"""
import os
from .phase1_eff_ada import Cfg as _Base


class Cfg(_Base):
    parallel_w = True
    pw_batched = True        # fold N_p panels into the batch dim → single batched expert call (the speedup)
    step_embed = True        # panel-index conditioning — REQUIRED so the N_p panels differ (no recurrence)
    w_feedback = False       # no recurrence (a feedback loop cannot be batched)
    swap_memory = False
    save_dir = "results/" + os.environ.get("PWB_SAVE", "phase1_int_ada_152m_6fam_pwbatched")
