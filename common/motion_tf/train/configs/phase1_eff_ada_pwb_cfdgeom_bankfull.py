"""Phase-1 ADA — cfdgeom + BANK_FULL_HISTORY. The operator-bank state z0 normally encodes only the LAST input
frame; here it encodes ALL Tin=10 frames (ic_enc 1×1 mixes the stacked frames per pixel), so the structured
operators (Δ,∇,−u·∇,ω,δ) act on a state summarizing the WHOLE input trajectory — the richest temporal info
(accel, jerk, evolution pattern), strictly more than the 2-frame bank_time_deriv. Targets the bank's temporal
blindness (PDEArena buoyancy/forcing). Caveat: z0 becomes an abstract 10-frame summary (advection of it is
heuristic, not a literal current-state operator). Stacks on cfdgeom; A/B vs cfdgeom (and vs banktd).
Launch (2 GPU): bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwb_cfdgeom_bankfull
"""
import os
from .phase1_eff_ada_pwbatched_cfdgeom import Cfg as _Base


class Cfg(_Base):
    bank_full_history = True   # z0 = ic_enc(all Tin frames) → bank sees the full input trajectory
    save_dir = "results/" + os.environ.get("BFH_SAVE", "phase1_int_ada_152m_6fam_pwb_cfdgeom_bankfull")
