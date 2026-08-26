"""Phase-1 ADA — rec_btd_cfd + explicit BUOYANCY SOURCE operator. pdearena loses to BCAT because it's a FORCED
system (buoyancy: scalar→vertical-velocity source) and our operator bank had only CONSERVATIVE transport
(Δ,∇,−u·∇,ω,∇·) — no source. This enables the unified expert's buoyancy branch: a learned pointwise
scalar→velocity Boussinesq source (W += by_out(by_l1([z, ∂_y z, h_geom])), zero-init), giving the bank an
explicit forcing operator. Stacks on the recursive accuracy champion (banktd ∂u/∂t input-side + cfdgeom).
banktd (acceleration info) + buoyancy operator (explicit source) are complementary. Target: close the
pdearena gap. A/B vs rec_btd_cfd and the recursive baseline.
Launch (2 GPU): bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_recursive_btd_cfd_buoy
"""
import os
from .phase1_eff_ada_recursive_banktd_cfdgeom import Cfg as _Base


class Cfg(_Base):
    unified_buoyancy = True          # explicit scalar→velocity source operator in the bank
    unified_buoyancy_hidden = 32
    save_dir = "results/" + os.environ.get("BUOY_SAVE", "phase1_int_ada_152m_6fam_rec_btd_cfd_buoy")
