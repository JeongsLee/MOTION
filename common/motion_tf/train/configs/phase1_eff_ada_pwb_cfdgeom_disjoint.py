"""Phase-1 ADA — cfdgeom + DISJOINT per-family channels. The shared slot-2 tracer packs DIFFERENT physics
(pdearena smoke / incom particles / com density) into one channel — hard for the physics-based bank. This
keeps velocity SHARED at slots 0,1 (same physics → transfer) but gives every family's scalar its OWN slot
(8 channels total) so no cross-family physics collision. Tests whether de-conflicting the scalar channels
improves learning. Requires env PROSE_CMAX=8 + PROSE_DISJOINT=1 (set in the launch cmd). A/B vs cfdgeom.
Launch (2 GPU): PROSE_CMAX=8 PROSE_DISJOINT=1 bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwb_cfdgeom_disjoint
"""
import os
from .phase1_eff_ada_pwbatched_cfdgeom import Cfg as _Base


class Cfg(_Base):
    n_channels = 8             # disjoint layout uses 8 channels (velocity shared 0,1 + per-family scalars)
    save_dir = "results/" + os.environ.get("DJ_SAVE", "phase1_int_ada_152m_6fam_pwb_cfdgeom_disjoint")
