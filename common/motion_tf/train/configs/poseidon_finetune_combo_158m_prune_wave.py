"""SELECTIVE PHYSICS-PRUNING arm (combo base), Wave-Layer set: the layered wave equation
d2u/dt2 = c(x)^2 Δu has cross-channel gradient coupling (the wave head's design target) and no pointwise
reaction, no buoyancy, no shear forcing, no material transport → off = reaction, buoyancy, gradx, warp;
kept = wave, adapter, adaptive_mix. Vars still created (positional load unchanged); pruned heads are
computed ×0 (built-but-inert). A/B partner: ftwave-800k-x4 (base 31.95, all heads on).
Launch:
  FT_TASK=Wave-Layer FT_NSHOT=64 FT_UV_ONLY=0 FT_IC_FRAC=1.0 FT_LR=5e-5 FT_STEPS=12800 \
  FT_TAG=_162m800000prune FT_INIT_PATH=/eu/results/poseidon_pretrain_ivp_combo_158m_scratch/ckpt_800000.npz \
  bash _run_ivp_eu.sh motion_tf.train.configs.poseidon_finetune_combo_158m_prune_wave
"""
from .poseidon_finetune_combo_158m import Cfg as _Base


class Cfg(_Base):
    unified_heads_off = ("reaction", "buoyancy", "gradx")
    warp_off = True
