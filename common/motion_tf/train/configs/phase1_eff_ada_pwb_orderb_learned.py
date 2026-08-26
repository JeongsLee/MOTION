"""Phase-1 ADA — ORDER-B with LEARNED bias-head front-end (not explicit operators). Combines the archbias
finding (learned architecture-bias >> explicit operators) with Order B (bias at the raw front-end, before the
transformer): bias-typed CONV heads (multi-dilation transport + 1×1 pointwise/source) are applied to the CLEAN
raw IC → learned physics-bias features → prepended to the encoder input → transformer → generic W-head → ADA.
vs archbias (Order A: transformer → learned bias-heads on latent): does the learned bias work BETTER at the
raw front-end (Order B) or after the transformer (Order A)? No explicit ops, no recursion.
Launch (2 GPU): bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwb_orderb_learned
"""
import os
from .phase1_eff_ada_pwbatched_cfdgeom import Cfg as _Base


class Cfg(_Base):
    learned_frontend = True      # learned bias-head convs on raw IC → encoder input (Order B, learned)
    lfe_dim = 16
    unified_ops = False          # expert = generic W-head (bias is in the learned front-end)
    save_dir = "results/" + os.environ.get("OBL_SAVE", "phase1_int_ada_152m_6fam_pwb_orderb_learned")
