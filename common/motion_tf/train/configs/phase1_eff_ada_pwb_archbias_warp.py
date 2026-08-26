"""Phase-1 ADA — optmatch + SEMI-LAGRANGIAN WARPING head, aimed at the pdearena weakness.
Root cause (established): the per-pixel ADA temporal integral integrates each Eulerian grid point on its OWN
fixed-t=0 W with NO spatial coupling → it cannot represent a feature TRANSPORTING across cells (a buoyant
plume rising lower→upper cell): a cell whose W(0)≈0 never receives the arriving plume. Buoyancy makes
y-velocity fundamentally Lagrangian (material parcels rise). Fix: predict a per-pixel displacement
d(x,t)=v0·t+½a0·t² (v0,a0 from the encoder, zero-init) and BACKWARD-WARP the IC, base(x,t)=IC(x−d(x,t))
(periodic bilinear) — a genuine Lagrangian transport operator (MOVES features; unlike extrap_residual which
adds a ramp to the value). Single-shot (no recursion), zero-init → identity warp at load. A/B vs optmatch at
1M, fair frames-10-19 metric. (Distinct from the adaptive_mix+gradx run, which addresses the TEMPORAL-basis
axis, not spatial transport.)
Launch (2 GPU): AB_SAVE=phase1_int_ada_152m_6fam_pwb_archbias_warp bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwb_archbias_warp
"""
import os
from .phase1_eff_ada_pwb_archbias_optmatch import Cfg as _Base


class Cfg(_Base):
    warp_head = True             # semi-Lagrangian IC transport (the pdearena Lagrangian-transport fix)
    save_dir = "results/" + os.environ.get("AB_SAVE", "phase1_int_ada_152m_6fam_pwb_archbias_warp")
