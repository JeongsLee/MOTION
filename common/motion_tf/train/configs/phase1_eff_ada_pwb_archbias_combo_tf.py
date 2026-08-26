"""Phase-1 ADA — combo + TEACHER-FORCED recursion (BCAT-cheap training of the recursive model).
Same model as combo_rec (recursive: parallel_w=False, pw_batched=False, with adaptive_mix+gradx+warp), but
TRAINING uses teacher forcing: feed the GROUND-TRUTH states at the panel times into the learned bank → all
panels' W in ONE parallel batched call (no sequential while_loop) → ADA-integrate → loss. INFERENCE/eval
(no target) falls back to the sequential while_loop (the true recursive model). This is exactly how BCAT/PROSE
(AR transformers) are cheap: parallel teacher-forced training, sequential inference. Net: training ~single-shot
speed (~1 s/step, NOT combo_rec's ~2.5 s) while the model is the recursive one — captures the turbulent cascade
that single-shot cannot. A/B vs combo (single-shot) and combo_rec (sequential-trained recursion) at 1M, fair
frames-10-19 metric. exposure-bias caveat: train on GT states, infer on rolled states (standard AR tradeoff).
Launch (2 GPU): AB_SAVE=phase1_int_ada_152m_6fam_pwb_archbias_combo_tf bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwb_archbias_combo_tf
"""
import os
from .phase1_eff_ada_pwb_archbias_combo_rec import Cfg as _Base


class Cfg(_Base):
    teacher_forced = True        # parallel teacher-forced TRAINING (sequential while_loop inference inherited)
    save_dir = "results/" + os.environ.get("AB_SAVE", "phase1_int_ada_152m_6fam_pwb_archbias_combo_tf")
