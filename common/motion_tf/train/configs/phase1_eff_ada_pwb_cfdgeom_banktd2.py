"""Phase-1 ADA — cfdgeom + BANK_TIME_DERIV with 2 differences (banktd2). z0 = ic_enc([last, u[-1]−u[-2],
u[-2]−u[-3]]) — TWO successive velocity-rates so the operator bank can infer ∂²u/∂t² (ACCELERATION =
buoyancy), which the single-difference banktd (n=1) cannot give. Tests whether richer time-derivative info
improves the buoyancy-driven PDEArena families further. A/B vs the n=1 banktd (recorded) and cfdgeom.
Launch (2 GPU): bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwb_cfdgeom_banktd2
"""
import os
from .phase1_eff_ada_pwbatched_cfdgeom import Cfg as _Base


class Cfg(_Base):
    bank_time_deriv = True
    bank_deriv_n = 2           # 2 first-differences (3 frames) → bank can infer acceleration (buoyancy)
    save_dir = "results/" + os.environ.get("BTD2_SAVE", "phase1_int_ada_152m_6fam_pwb_cfdgeom_banktd2")
