"""Phase-1 ADA — cfdgeom + BANK_TIME_DERIV. The operator-bank state z0 is normally just the LAST input frame,
so the spatial operators (Δ,∇,−u·∇,ω,δ) are blind to the input time-history — yet buoyancy (dropped buo_y) is
an ACCELERATION visible only across frames. This feeds the observed ∂u/∂t ≈ (last − prev input frame) INTO z0
(via ic_enc), so the bank state itself carries the rate-of-change and the operators can build a buoyancy/source
tendency. Targets a SECOND root cause of the PDEArena weakness (bank temporal-blindness), orthogonal to
fixedpoint (frozen advection). Stacks on cfdgeom; A/B vs cfdgeom isolates the temporal-state effect.
Launch (2 GPU): bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwb_cfdgeom_banktd
"""
import os
from .phase1_eff_ada_pwbatched_cfdgeom import Cfg as _Base


class Cfg(_Base):
    bank_time_deriv = True     # z0 = ic_enc([last frame, last−prev]) → bank sees ∂u/∂t (accel/buoyancy)
    save_dir = "results/" + os.environ.get("BTD_SAVE", "phase1_int_ada_152m_6fam_pwb_cfdgeom_banktd")
