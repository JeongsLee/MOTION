"""Phase-1 ADA — cfdgeom + EXTRAPOLATION RESIDUAL. Upgrades the residual anchor from "ic (last frame,
constant)" to a learned temporal Taylor extrapolation of the input history: base(t) = ic + v0·t + ½·a0·t²,
where v0 (velocity) and a0 (ACCELERATION = forcing/buoyancy signature) are learned from the input frame
1st/2nd differences. Captures the FORCED secular drift (buoyancy plume rise) directly from the 10 IC frames —
the component the conservative bank + ADA basis represent poorly — so ADA only models the deviation. No
recursion needed for the drift. zero-init → starts == cfdgeom. A/B vs cfdgeom (does direct extrapolation of
the observed forcing beat modeling it through W+ADA?).
Launch (2 GPU): bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwb_cfdgeom_extrap
"""
import os
from .phase1_eff_ada_pwbatched_cfdgeom import Cfg as _Base


class Cfg(_Base):
    extrap_residual = True
    save_dir = "results/" + os.environ.get("EX_SAVE", "phase1_int_ada_152m_6fam_pwb_cfdgeom_extrap")
