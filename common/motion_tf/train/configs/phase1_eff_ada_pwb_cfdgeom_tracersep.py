"""Phase-1 ADA — cfdgeom + PARTIAL-DISJOINT (tracersep). Fix of the full-disjoint over-separation: separate
only GENUINELY-different physics, keep SIMILAR scalars SHARED. velocity shared 0,1; SMOKE shared at slot 2
(pdearena_ns AND uncond — same passive-scalar physics → transfer kept); incom PARTICLES get own slot 3 (the
@250 incom gain); com rho/pressure → 4,5; SWE height → 6. 7 channels. A/B vs cfdgeom (does de-conflicting
particles-from-smoke help WITHOUT losing smoke transfer?) and vs full disjoint (which over-separated).
Launch: PROSE_CMAX=7 PROSE_DISJOINT=2 bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwb_cfdgeom_tracersep
"""
import os
from .phase1_eff_ada_pwbatched_cfdgeom import Cfg as _Base


class Cfg(_Base):
    n_channels = 7
    save_dir = "results/" + os.environ.get("TS_SAVE", "phase1_int_ada_152m_6fam_pwb_cfdgeom_tracersep")
