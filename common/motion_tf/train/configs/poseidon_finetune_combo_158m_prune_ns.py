"""SELECTIVE PHYSICS-PRUNING arm (combo base), NS-PwC set: unforced incompressible NS = advection +
diffusion + pressure projection, velocity-only data → keep warp (semi-Lagrangian transport, the dominant
mechanism) + adapter; off = reaction (no source), wave (no cross-channel restoring force in uv-only data),
buoyancy/gradx (unforced — no gravity, no imposed shear). Pruned heads computed ×0 (built-but-inert,
positional load unchanged). A/B partner: ftns-800k-x4 (base 5.69, all heads on).
Launch:
  FT_TASK=NS-PwC FT_NSHOT=64 FT_UV_ONLY=1 FT_IC_FRAC=1.0 FT_LR=5e-5 FT_STEPS=12800 \
  FT_TAG=_162m800000prune FT_INIT_PATH=/eu/results/poseidon_pretrain_ivp_combo_158m_scratch/ckpt_800000.npz \
  bash _run_ivp_eu.sh motion_tf.train.configs.poseidon_finetune_combo_158m_prune_ns
"""
from .poseidon_finetune_combo_158m import Cfg as _Base


class Cfg(_Base):
    unified_heads_off = ("reaction", "wave", "buoyancy", "gradx")
