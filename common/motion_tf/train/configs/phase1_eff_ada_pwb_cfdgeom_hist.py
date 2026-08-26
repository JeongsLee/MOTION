"""Phase-1 ADA — cfdgeom + HISTORY-AUGMENT. The no-recursion bank sees only z0 (frozen IC). Augment z0 with a
learned encoding of the last K input frames (a temporal conv): z0 = [current-frame encode | K-frame history
features]. The operators then see the OBSERVED recent trend (velocity/accel/higher-order) — richer than the
fixed ∂u/∂t (banktd), and ADDED not replacing the state (avoids the full-history bottleneck that hurt). A/B vs
cfdgeom: does observed short-history context help the frozen-bank no-recursion model?
Launch (2 GPU): bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwb_cfdgeom_hist
"""
import os
from .phase1_eff_ada_pwbatched_cfdgeom import Cfg as _Base


class Cfg(_Base):
    history_augment = True
    history_k = 3                # last 3 input frames
    history_dim = 6             # history feature channels (of the 16 latent; 10 for current-frame)
    save_dir = "results/" + os.environ.get("HIST_SAVE", "phase1_int_ada_152m_6fam_pwb_cfdgeom_hist")
