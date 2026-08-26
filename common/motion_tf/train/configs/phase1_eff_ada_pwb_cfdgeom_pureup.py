"""Phase-1 ADA — cfdgeom + PURE LEARNED UPSAMPLE. Drops the bilinear (low-pass) base of the coarse 32²→128²
W upsample entirely; the upsample becomes a SINGLE learned sub-pixel conv (depth_to_space), glorot-init. The
bilinear ceiling that smears high-frequency content — a prime suspect for the turbulent PDEArena weakness —
is removed. Stacks on cfdgeom (pw_batched + cfdbench_geom) so the A/B vs cfdgeom isolates the upsample change.
Stability: W is small pre-upsample (W_scale≈0.05) + residual decode (field=IC+δ) → glorot-init upsample of a
small W keeps the start near IC. Target: does freeing the upsampler from bilinear recover PDEArena high-freq?
Launch (2 GPU): bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwb_cfdgeom_pureup
"""
import os
from .phase1_eff_ada_pwbatched_cfdgeom import Cfg as _Base


class Cfg(_Base):
    pure_learned_upsample = True    # drop bilinear base → single learned sub-pixel upsample (glorot-init)
    save_dir = "results/" + os.environ.get("PUP_SAVE", "phase1_int_ada_152m_6fam_pwb_cfdgeom_pureup")
