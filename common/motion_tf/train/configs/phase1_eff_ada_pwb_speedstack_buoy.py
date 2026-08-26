"""Phase-1 ADA — speedstack + BUOYANCY SOURCE operator. The speed-first no-recursion champion (pw_batched +
cfdgeom + banktd + panel_attn) plus the explicit scalar→velocity Boussinesq source branch in the bank — gives
the fast model a FORCING operator (the missing piece behind pdearena losing to BCAT). All four levers stacked,
still ~1s no-recursion. A/B vs speedstack (does the source operator close pdearena at speed?) and vs the
recursive buoyancy variant.
Launch (2 GPU): bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwb_speedstack_buoy
"""
import os
from .phase1_eff_ada_pwb_speedstack import Cfg as _Base


class Cfg(_Base):
    unified_buoyancy = True          # explicit scalar→velocity source operator (forcing)
    unified_buoyancy_hidden = 32
    save_dir = "results/" + os.environ.get("SSB_SAVE", "phase1_int_ada_152m_6fam_pwb_speedstack_buoy")
