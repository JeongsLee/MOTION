"""Phase-1 ADA — optmatch + ALL THREE orthogonal pdearena levers combined (best-accuracy shot).
Each hits a different axis and composes cleanly (gated, zero-init, single-shot):
  • adaptive_mix  : per-pixel/per-channel Fourier↔Legendre temporal-basis blend (temporal axis).
  • unified_gradx : ∂_x horizontal-shear head, x-counterpart of ∂_y buoyancy (operator axis).
  • warp_head     : semi-Lagrangian IC transport, base(x,t)=IC(x−d(x,t)) (spatial-transport axis — the
                    Eulerian-vs-Lagrangian root cause of the pdearena weakness).
field = warp(IC,d) + decode(adaptive-mixed ∫W[+gradx]). All zero-init → starts == optmatch; single-shot speed
kept. A/B vs optmatch + the isolated runs (warp-only, adaptive_mix+gradx) at 1M, fair frames-10-19 metric.
Launch (2 GPU): AB_SAVE=phase1_int_ada_152m_6fam_pwb_archbias_combo bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwb_archbias_combo
"""
import os
from .phase1_eff_ada_pwb_archbias_optmatch import Cfg as _Base


class Cfg(_Base):
    adaptive_mix = True
    unified_gradx = True
    warp_head = True
    save_dir = "results/" + os.environ.get("AB_SAVE", "phase1_int_ada_152m_6fam_pwb_archbias_combo")
