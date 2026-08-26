"""Phase-1 ADA — cfdgeom + SLOT-STRUCTURED latent. Instead of a fully-entangled 6→16 learned latent (where the
operators' vel_slots=(0,1) are a learned proxy and smoke is mixed), keep the latent's channel identity CLEAN:
the first C latent slots ARE the physical channels (vx,vy,scalar,rho,p,height passed through), the remaining
D−C are learned extra capacity. Then advection reads REAL velocity (slots 0,1) and any scalar/source reads the
REAL scalar (slot 2) — disentangling the operators from the learned mix. A/B vs cfdgeom: does clean physical
channels for the physics operators improve the velocity-driven families (esp. the entangled advection)?
Launch (2 GPU): bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwb_cfdgeom_slotstruct
"""
import os
from .phase1_eff_ada_pwbatched_cfdgeom import Cfg as _Base


class Cfg(_Base):
    slot_structured = True       # latent = [physical C | learned D−C]; operators read clean physical channels
    save_dir = "results/" + os.environ.get("SST_SAVE", "phase1_int_ada_152m_6fam_pwb_cfdgeom_slotstruct")
