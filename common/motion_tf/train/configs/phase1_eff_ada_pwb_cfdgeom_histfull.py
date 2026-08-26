"""Phase-1 ADA — cfdgeom + HISTORY-AUGMENT, FULL input (K=10). Same as the hist variant but the history
encoder sees ALL T_in input frames (history_k=10) instead of the last 3. Still AUGMENTS z0 (z0 = [current
encode | full-history features]) so it avoids the bank_full_history bottleneck (which REPLACED z0). A/B vs
hist (K=3): does the full observed window help the frozen-bank no-recursion model more than a short history?
Launch (2 GPU): bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwb_cfdgeom_histfull
"""
import os
from .phase1_eff_ada_pwbatched_cfdgeom import Cfg as _Base


class Cfg(_Base):
    history_augment = True
    history_k = 10              # ALL input frames (vs hist's 3)
    history_dim = 8            # a bit more capacity for the richer 10-frame history (current block = 16−8 = 8)
    save_dir = "results/" + os.environ.get("HF_SAVE", "phase1_int_ada_152m_6fam_pwb_cfdgeom_histfull")
