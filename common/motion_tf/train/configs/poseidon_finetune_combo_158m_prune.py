"""SELECTIVE PHYSICS-PRUNING arm (combo base): finetune with every head that does not match the task's
equation DISABLED in the forward (vars kept — positional load unchanged). For the steady elliptic
Poisson-Gauss: off = wave (cross-channel ∇), buoyancy (∂_y source), gradx (∂_x shear), warp (semi-
Lagrangian transport); kept = reaction (pointwise source, spans f), adapter (generic), adaptive_mix.
NOTE unlike the zero-init banks this is NOT a no-op at load — the pruned heads were live contributors at
pretrain, so the load-time model is amputated and finetune must re-adapt without them. A/B partner:
ftpoisson-800k-v2 (base 6.25, all heads on).
Launch:
  FT_TASK=Poisson-Gauss FT_NSHOT=64 FT_UV_ONLY=0 FT_IC_FRAC=0 FT_LR=5e-5 FT_STEPS=12800 FT_LOSS_CAP=100 \
  FT_TAG=_162m800000prune FT_INIT_PATH=/eu/results/poseidon_pretrain_ivp_combo_158m_scratch/ckpt_800000.npz \
  bash _run_ivp_eu.sh motion_tf.train.configs.poseidon_finetune_combo_158m_prune
"""
from .poseidon_finetune_combo_158m import Cfg as _Base


class Cfg(_Base):
    unified_heads_off = ("wave", "buoyancy", "gradx")
    warp_off = True
