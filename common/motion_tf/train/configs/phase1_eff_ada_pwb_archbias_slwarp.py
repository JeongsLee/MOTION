"""Phase-1 ADA — combo + VELOCITY-COUPLED COMPOSITIONAL semi-Lagrangian warp (strengthen the Lagrangian bank).
combo's warp does ONE backward warp of the IC by a 2nd-order Taylor displacement (parabolic path, decoupled
from a real flow). Here warp_steps=4 → trace the backward characteristic in 4 sub-steps through the predicted
velocity field u(x) (warp_v), sampling u at the moving departure point each step (d_k=d_{k-1}+u(x−d_{k-1})·t/K)
→ CURVED, self-consistent transport. Pure grid-sample → BOUNDED (cannot diverge like the bank-recursion
combo-tf), single-shot (4 cheap warps, no while_loop / no bank re-eval). zero-init → identity at load == combo.
A/B vs combo at 1M, fair frames-10-19 metric. Targets the turbulent/buoyant transport (pdearena, incom tracer).
Launch (2 GPU): AB_SAVE=phase1_int_ada_152m_6fam_pwb_archbias_slwarp bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwb_archbias_slwarp
"""
import os
from .phase1_eff_ada_pwb_archbias_combo import Cfg as _Base


class Cfg(_Base):
    warp_steps = 4               # K-substep semi-Lagrangian backward-characteristic tracing (1 = legacy Taylor)
    save_dir = "results/" + os.environ.get("AB_SAVE", "phase1_int_ada_152m_6fam_pwb_archbias_slwarp")
