"""Phase-1 ADA — cfdgeom + FIXED-POINT K=1, NO stop_gradient. One Picard sweep, but the reconstruction stays
DIFFERENTIABLE (backward flows W→state→W) so the first-eval W IS optimized for the output — isolates whether
fp1sg's slow early start was caused by the stop_gradient detach. Cost ~1.5 s/step (2 bank evals + 2-deep BPTT,
vs fp1sg's ~1.15 with 1-deep backward). A/B: fp1 (diff) vs fp1sg (detached) vs fixedpoint K=2.
Launch (2 GPU): bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwb_cfdgeom_fp1
"""
import os
from .phase1_eff_ada_pwbatched_cfdgeom import Cfg as _Base


class Cfg(_Base):
    fixedpoint_iters = 1         # one Picard sweep
    fixedpoint_stopgrad = False  # DIFFERENTIABLE reconstruction (vs fp1sg) → first-eval W optimized
    save_dir = "results/" + os.environ.get("FP1D_SAVE", "phase1_int_ada_152m_6fam_pwb_cfdgeom_fp1")
