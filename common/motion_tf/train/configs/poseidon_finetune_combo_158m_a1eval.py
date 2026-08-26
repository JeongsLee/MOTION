"""EVAL-ONLY at anchor frame 1, NO v0 injection: completes our side of the 2x2 (model x info x anchor)
Wave matrix — how badly does OUR finetuned 1IC base degrade when anchored at a state-hidden frame
(the exact probe that took Poseidon 19.3 -> 43.9)? Run with FT_STEPS=2 and a vanishing LR from the
finished base ckpt so the printed evals measure the loaded weights.
Launch:
  FT_TASK=Wave-Layer FT_NSHOT=64 FT_UV_ONLY=0 FT_IC_FRAC=1.0 FT_LR=1e-9 FT_STEPS=2 \
  FT_CKPT_EVERY=2 FT_EVAL_EVERY=1 FT_TAG=_a1eval \
  FT_INIT_PATH=/eu/results/ftcombo_Wave-Layer_N64_162m800000x4/ckpt_12800.npz \
  bash _run_ivp_eu.sh motion_tf.train.configs.poseidon_finetune_combo_158m_a1eval
"""
from .poseidon_finetune_combo_158m import Cfg as _Base


class Cfg(_Base):
    eval_anchor = 1
