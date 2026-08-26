"""Finetune config for ADV-BANK pretrained ckpts (v3 lineage): arch must match the pretrain (adv_bank on,
clip=2.0 affects the forward; vstd/tw0 only affect init and are overwritten by the ckpt load). Used for the
matched-step TRANSFER PROBE: FT(v3@100k) vs FT(old@100k) on Wave — the pretrain-integration question is
decided by transfer quality, not pretrain accuracy.
Launch:
  FT_TASK=Wave-Layer FT_NSHOT=64 FT_UV_ONLY=0 FT_IC_FRAC=1.0 FT_LR=5e-5 FT_STEPS=12800 FT_TAG=_v3pre100k \
  FT_INIT_PATH=/eu/results/poseidon_pretrain_ivp_combo_158m_scratch_advb/ckpt_100000.npz \
  bash _run_ivp_eu.sh motion_tf.train.configs.poseidon_finetune_combo_158m_advb
"""
from .poseidon_finetune_combo_158m import Cfg as _Base


class Cfg(_Base):
    adv_bank = True
    adv_bank_M = 4
    adv_bank_K = 4
    adv_bank_vstd = 0.02
    adv_bank_tw0 = True
    adv_bank_clip = 2.0
