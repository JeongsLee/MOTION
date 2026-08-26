"""Phase-1 ADA — combo + RECURSION (the accuracy-max variant; paper's 2nd proposed model).
Same as combo (adaptive_mix + unified_gradx + warp_head on optmatch), but DROP the single-shot pw_batched
rollout and use the SEQUENTIAL while_loop with Euler state-feedback (parallel_w=False, pw_batched=False).
Recursion re-evaluates the (learned) heads on the EVOLVING state z(t_i)=z_{i-1}+dt·W → captures the turbulent
cascade / Lagrangian transport that single-shot cannot (the pdearena weakness). This is the mechanism BCAT/
PROSE use (autoregression) — here we add it WITHIN the ADA framework, to quantify the speed↔accuracy
tradeoff: combo (single-shot, ~2.5× faster) vs combo_rec (recursion, recovers pdea toward the AR baselines).
Slower (~2.5 s/step sequential). A/B vs combo at 1M, fair frames-10-19 metric.
Launch (2 GPU): AB_SAVE=phase1_int_ada_152m_6fam_pwb_archbias_combo_rec bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwb_archbias_combo_rec
"""
import os
from .phase1_eff_ada_pwb_archbias_combo import Cfg as _Base


class Cfg(_Base):
    parallel_w = False           # ENABLE recursion: sequential while_loop with Euler state-feedback (z evolves)
    pw_batched = False           # (pw_batched requires parallel_w; off → recursive path)
    save_dir = "results/" + os.environ.get("AB_SAVE", "phase1_int_ada_152m_6fam_pwb_archbias_combo_rec")
