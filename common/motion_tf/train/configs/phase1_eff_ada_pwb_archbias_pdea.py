"""Phase-1 ADA — optmatch + two SINGLE-SHOT, philosophy-consistent levers aimed at the lone weak family
(pdearena, the buoyant/turbulent forced NS). Both zero-init → start identical to optmatch, learn from there.
  • adaptive_mix : the Fourier↔Legendre temporal-basis blend is a per-pixel learned map (was ONE global
    scalar mix_logit that converged stuck ~0.58, unable to specialize). Lets turbulent/aperiodic pdearena
    pixels lean Legendre and wave-like pixels lean Fourier. (mix(x)=σ(mix_logit + mix_head_lat(h)).)
  • unified_gradx : a ∂_x horizontal-gradient/shear head, the x-counterpart of the ∂_y buoyancy head
    (Kelvin–Helmholtz shear, horizontal transport).
NO recursion / NO hand-coded extrapolation (extrap_residual rejected as un-architectural). Single-shot speed
kept. A/B vs optmatch at the same 1M-view budget; metric is the FAIR frames-10-19 (inline eval drops the IC).
Launch (2 GPU): bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwb_archbias_pdea
"""
import os
from .phase1_eff_ada_pwb_archbias_optmatch import Cfg as _Base


class Cfg(_Base):
    adaptive_mix = True          # per-pixel Fourier/Legendre mix (latent path) — diagnosed global-mix fix
    unified_gradx = True         # ∂_x horizontal shear/gradient head (mirror of ∂_y buoyancy)
    save_dir = "results/" + os.environ.get("AB_SAVE", "phase1_int_ada_152m_6fam_pwb_archbias_pdea")
