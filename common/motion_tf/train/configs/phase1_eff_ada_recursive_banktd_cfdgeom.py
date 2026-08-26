"""Phase-1 ADA — RECURSIVE baseline + banktd + cfdgeom, NO panel_attn (the combo minus attention). The full
combo (recursive + banktd + cfdgeom + attn) UNDERPERFORMED the plain recursive baseline — suspected the
panel_attn-in-recursive path interferes with the z-recursion's already-coupled trajectory. This isolates that
by dropping attn: keep only the recursion-ORTHOGONAL physics/data gains (banktd ∂u/∂t buoyancy + cfdgeom slot-2
cleanup) on the full-accuracy recursive baseline. A/B vs recursive baseline (4.43%) and vs the full combo.
Launch (2 GPU): bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_recursive_banktd_cfdgeom
"""
import os
from .phase1_eff_ada import Cfg as _Base


class Cfg(_Base):
    cfdbench_geom = True       # CFDBench boundary → geom_mask (slot-2 cleanup)
    bank_time_deriv = True     # z0 = ic_enc([last, ∂u/∂t]) → bank sees acceleration (buoyancy)
    # panel_attn OFF (the suspected interferer in the recursive path)
    save_dir = "results/" + os.environ.get("RBC_SAVE", "phase1_int_ada_152m_6fam_recursive_banktd_cfdgeom")
