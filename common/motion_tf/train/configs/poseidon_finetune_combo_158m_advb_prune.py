"""SELECTIVE PHYSICS-PRUNING arm (advb/v4 base): adv_bank OFF + every equation-mismatched head OFF at
finetune (vars kept — positional v4 load unchanged). Poisson-Gauss set: off = adv_bank, wave, buoyancy,
gradx, warp; kept = reaction, adapter, adaptive_mix. Tests whether the advb transfer deficit (Poisson
9.62 vs combo 6.25) is the ACTIVE task-irrelevant vocabulary (then pruning recovers) or the pretrained
representation (then it does not). A/B partners: ftpois-advb800k-x4 (9.62, all on) and the combo prune arm.
Launch:
  FT_TASK=Poisson-Gauss FT_NSHOT=64 FT_UV_ONLY=0 FT_IC_FRAC=0 FT_LR=5e-5 FT_STEPS=12800 FT_LOSS_CAP=100 \
  FT_TAG=_advb800kx4prune FT_INIT_PATH=/eu/results/poseidon_pretrain_ivp_combo_158m_scratch_advb/ckpt_800000.npz \
  bash _run_ivp_eu.sh motion_tf.train.configs.poseidon_finetune_combo_158m_advb_prune
"""
from .poseidon_finetune_combo_158m_advb import Cfg as _Base


class Cfg(_Base):
    adv_bank_off = True
    unified_heads_off = ("wave", "buoyancy", "gradx")
    warp_off = True
