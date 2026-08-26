"""PAIR-DIET finetune arm (Poseidon-matched supervision): our single-shot loss supervises ALL valid
frames jointly each step; scOT supervises ONE (t1,t2) pair per sample. If the joint loss forces a
cross-lead compromise (hedged frames), matching the pair diet (one random lead per step, 40
pair-gradients/step = scOT's exact diet) should recover accuracy — first probe on the task where the
budget-matched scOT overtook us (NS-PwC: ours 5.69 vs scOT-x4 5.34). Same forward, same 12800 steps;
only the gradient focus changes.
Launch:
  FT_TASK=NS-PwC FT_NSHOT=64 FT_UV_ONLY=1 FT_IC_FRAC=1.0 FT_LR=5e-5 FT_STEPS=12800 \
  FT_TAG=_162m800000pair FT_INIT_PATH=/eu/results/poseidon_pretrain_ivp_combo_158m_scratch/ckpt_800000.npz \
  bash _run_ivp_eu.sh motion_tf.train.configs.poseidon_finetune_combo_158m_pairloss
"""
from .poseidon_finetune_combo_158m import Cfg as _Base


class Cfg(_Base):
    pair_loss = True
