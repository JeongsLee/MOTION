"""EVAL-ONLY ANCHOR SWEEP: score the loaded ckpt anchored at frames {0,1,2,4,8} (full-traj over the
remaining frames A+1..Nt-1). Wave showed anchor-0-only evaluation hides an anchor-specialization axis
(Poseidon 19->44 collapse at anchor 1 vs our flat 32); this probes the same axis on the OTHER families,
where the state is fully observed at every anchor — any anchor dependence left is IC-distribution
specialization (+ shrinking horizon), not observability. Run with FT_STEPS=2 and vanishing LR.
Launch (per task):
  FT_TASK=<task> ... FT_LR=1e-9 FT_STEPS=2 FT_CKPT_EVERY=2 FT_EVAL_EVERY=1 FT_TAG=_asweep \
  FT_INIT_PATH=<finished x4 ckpt> bash _run_ivp_eu.sh motion_tf.train.configs.poseidon_finetune_combo_158m_asweep
"""
from .poseidon_finetune_combo_158m import Cfg as _Base


class Cfg(_Base):
    eval_anchor_sweep = (0, 1, 2, 4, 8)
