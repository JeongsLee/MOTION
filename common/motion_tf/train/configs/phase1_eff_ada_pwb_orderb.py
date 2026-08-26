"""Phase-1 ADA — ORDER-B: physics-bias FRONT-END → transformer → W → ADA. The physics operators (Δ,∇,−u·∇,ω,∇·)
are computed on the CLEAN physical IC (real velocity for advection) and prepended to the encoder input, so the
architecture-level inductive bias acts at the PHYSICAL scale (raw fields) BEFORE the transformer's global
learned mixing — not on the transformer's abstract latent (Order A, which entangles channels). The expert is a
generic head (unified_ops=False): physics lives in the encoder input, the transformer + head + ADA do the
rest. No hidden-z recursion (pw_batched parallel). Matches the NTO-ADA paper paradigm (bias in the front-end
architecture). A/B vs archbias (Order A: transformer→bias-head on latent) and cfdgeom (explicit-ops baseline).
Launch (2 GPU): bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwb_orderb
"""
import os
from .phase1_eff_ada_pwbatched_cfdgeom import Cfg as _Base


class Cfg(_Base):
    physics_frontend = True      # physics operators on clean raw IC → into the encoder input
    unified_ops = False          # expert = generic W-head (physics is front-end, not in the expert)
    save_dir = "results/" + os.environ.get("OB_SAVE", "phase1_int_ada_152m_6fam_pwb_orderb")
