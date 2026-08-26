"""ACE diagnostic arm: 162M combo finetune + TRAINABLE REACTION-DIFFUSION BANK (model.ace_bank). The combo
base's generic pointwise ReactionExpert lacks Allen-Cahn's LONG-TIME structure: the saturating bistable
kinetics closed form u(t)=u0·e^{at}/√(1+s·u0²(e^{2at}−1)), the diffusion semigroup e^{ε²tΔ}, and the
curvature-driven interface motion κ|∇u| — the bank adds all three zero-init on the base path (a/s/δ/γ
learned per-channel, +~40 params). ckpt_800000 loads BY NAME (migrated npz) with ab_* fresh-init.
A/B partner: ftace-800k-x4 (same 12800-step protocol, ace_bank OFF).
Launch:
  FT_TASK=ACE FT_NSHOT=64 FT_UV_ONLY=0 FT_IC_FRAC=1.0 FT_LR=5e-5 FT_STEPS=12800 FT_TAG=_162m800000ab \
  FT_INIT_PATH=/eu/results/migrated/combo158m_800000_named.npz \
  bash _run_ivp_eu.sh motion_tf.train.configs.poseidon_finetune_combo_158m_acebank
"""
from .poseidon_finetune_combo_158m import Cfg as _Base


class Cfg(_Base):
    ace_bank = True
    ace_bank_K = 2
