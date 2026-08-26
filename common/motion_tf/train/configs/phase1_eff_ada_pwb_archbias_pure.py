"""Phase-1 ADA — archbias + PURE learned upsample (drop bilinear low-pass ceiling). Extends the session-best
archbias (learned multi-bias-heads, Order A, no explicit ops, no recursion; final @1M: ov 4.63, cond-pdea
13.74 TIED with recursive, residual ALL in pdearena_uncond +1.75). The remaining uncond gap is long-horizon
HIGH-FREQUENCY (turbulent Rayleigh-Taylor cascade) that the bilinear low-pass coarse->full W upsample smears.

Change vs archbias: pure_learned_upsample=True → the coarse(latent_factor=4)->full W upsample is a SINGLE
glorot-init learned sub-pixel conv (depth_to_space), NO bilinear base. Removes the low-pass ceiling that caps
high-freq synthesis. Hypothesis: helps pdearena_uncond (and cond) high-freq; risk = no bilinear anchor → may
need to learn the upsample from scratch (slower early). A/B vs archbias on the uncond residual.
Launch (2 GPU): bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwb_archbias_pure
"""
import os
from .phase1_eff_ada_pwb_archbias import Cfg as _Base


class Cfg(_Base):
    learned_upsample = False         # turn off bilinear+residual mode (pure replaces it)
    pure_learned_upsample = True     # SINGLE glorot-init sub-pixel conv, no bilinear low-pass base
    save_dir = "results/" + os.environ.get("AB_SAVE", "phase1_int_ada_152m_6fam_pwb_archbias_pure")
