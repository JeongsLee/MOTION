"""Phase-1 ADA — ARCHITECTURE-BIAS HEADS (paper paradigm, no explicit operators, no recursion). Motivated by
the NTO-ADA paper: the encoder's INDUCTIVE BIAS lives in the architecture (not in an explicit Δ/∇/advection
operator bank on a recursively-evolved hidden z). Here we DROP the explicit operator bank (unified_ops=False)
and replace it with MULTIPLE architectural-bias heads, each emitting W directly in PARALLEL (pw_batched,
step_embed — no hidden-z recursive differencing):
  • adapter   : multi-dilation conv → transport/multi-scale bias
  • reaction  : pointwise 1×1 → reaction/source bias
  • buoyancy  : cross-channel ∂_y → scalar→velocity (Boussinesq) bias
  • wave      : cross-channel ∇ → wave/hyperbolic bias
slot_structured keeps the latent channels CLEAN so each head reads/writes real physical quantities (e.g.
buoyancy reads clean smoke → vy). Scale ~152M (heads add modest params; encoder dominates). A/B vs cfdgeom
(explicit-ops baseline): does architecture-level bias (paper style) beat explicit-ops-on-frozen-z?
Launch (2 GPU): bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwb_archbias
"""
import os
from .phase1_eff_ada_pwbatched_cfdgeom import Cfg as _Base


class Cfg(_Base):
    unified_ops = False          # DROP explicit Δ/∇/advection operators → architecture bias only
    slot_structured = True       # clean physical channels → heads target real quantities
    unified_adapter = True       # transport/multi-scale bias head
    unified_reaction = True      # reaction/source pointwise bias head
    unified_buoyancy = True      # scalar→velocity (Boussinesq) source bias head
    unified_wave = True          # cross-channel ∇ (wave) bias head
    save_dir = "results/" + os.environ.get("AB_SAVE", "phase1_int_ada_152m_6fam_pwb_archbias")
