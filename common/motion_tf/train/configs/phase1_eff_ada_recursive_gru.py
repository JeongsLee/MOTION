"""Phase-1 ADA — RECURSIVE baseline + LEARNED GRU integrator. Keeps the physics bank (W = operators on z) but
replaces the fixed forward-Euler state-update (z+=W·dt) with a learned GRU-style gated update
z' = z + u⊙cand (u=update gate, cand=zero-init candidate of [W, r⊙z]). The integrator is LEARNED (generalizes
Euler/AB2's fixed scheme; the gate can damp → stabilizes) while the operators stay physical. zero-init candidate
→ z'=z at start (validated identity start). A/B vs recursive baseline (Euler) and rec_ab2 (fixed 2nd-order):
fixed-Euler vs fixed-2nd-order vs learned integrator. Sequential (recursive), ~baseline speed + GRU convs.
Launch (2 GPU): bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_recursive_gru
"""
import os
from .phase1_eff_ada import Cfg as _Base


class Cfg(_Base):
    recursive_gru = True       # learned GRU gated z-update (vs fixed Euler/AB2)
    save_dir = "results/" + os.environ.get("GRU_SAVE", "phase1_int_ada_152m_6fam_recursive_gru")
