"""Phase-1 ADA — cfdgeom + FIXED-POINT K=1 + stop_gradient (the CHEAPER, STABLER fixed-point). One Picard
sweep (2 bank evals → ~1.5 s/step vs K=2's ~2.0) — the hypothesis being that the FIRST W-reconstruction
already un-freezes advection (evolved velocity), so one sweep captures most of the recursion's PDEArena gain.
stop_gradient on the reconstructed state: backward graph stays 1-deep (cheaper) AND breaks the W→state→W
amplification that spiked the K=2 run at step 1000 (more stable). A/B vs the K=2 fixedpoint (does K=1 retain
the gain at lower cost?) and vs baseline (recover recursion accuracy at ~60% the cost, fully parallel).
Launch (2 GPU): bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwb_cfdgeom_fp1sg
"""
import os
from .phase1_eff_ada_pwbatched_cfdgeom import Cfg as _Base


class Cfg(_Base):
    fixedpoint_iters = 1        # ONE Picard sweep (2 bank evals) → ~1.5 s/step
    fixedpoint_stopgrad = True  # detach the W-reconstructed state → cheaper backward + stable (no spike)
    save_dir = "results/" + os.environ.get("FP1_SAVE", "phase1_int_ada_152m_6fam_pwb_cfdgeom_fp1sg")
