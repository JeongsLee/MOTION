"""SELECTIVE PHYSICS-PRUNING arm (combo base), ACE set: Allen-Cahn du/dt = eps*Δu + u − u^3 is pointwise
reaction + diffusion, no transport, no wave coupling, no directional forcing → keep reaction (exactly
R(u)) + adapter (diffusion lives in the trunk); off = wave, buoyancy, gradx, warp. Pruned heads computed
×0 (built-but-inert, positional load unchanged). A/B partner: ftace-800k-x4 (base 0.78, all heads on).
Launch:
  FT_TASK=ACE FT_NSHOT=64 FT_UV_ONLY=0 FT_IC_FRAC=1.0 FT_LR=5e-5 FT_STEPS=12800 \
  FT_TAG=_162m800000prune FT_INIT_PATH=/eu/results/poseidon_pretrain_ivp_combo_158m_scratch/ckpt_800000.npz \
  bash _run_ivp_eu.sh motion_tf.train.configs.poseidon_finetune_combo_158m_prune_ace
"""
from .poseidon_finetune_combo_158m import Cfg as _Base


class Cfg(_Base):
    unified_heads_off = ("wave", "buoyancy", "gradx")
    warp_off = True
