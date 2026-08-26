"""Wave-Layer diagnostic arm: 162M combo finetune + TRAINABLE WAVE BANK (model.wave_bank). The combo base
has no 2nd-order-in-time structure (warp = transport, ADA = damped integral) — Wave-Layer needs an
oscillatory/conservative propagator with layered-c interface reflection. The bank adds a zero-init Taylor
expansion of u(t)=cos(t√L)u0 + t·sinc(t√L)v0 (L = c̃²Δ, learned c̃² + hidden v0 + ∇c̃²·∇u0 interface term)
on the base path; ckpt_800000 loads BY NAME (migrated npz) with the new wb_* vars fresh-init.
A/B partner: ftwave-162m800k (same seed data/protocol, wave_bank OFF).
Launch:
  FT_TASK=Wave-Layer FT_NSHOT=64 FT_UV_ONLY=0 FT_IC_FRAC=1.0 FT_LR=5e-5 FT_TAG=_162m800000wb \
  FT_INIT_PATH=/eu/results/migrated/combo158m_800000_named.npz \
  bash _run_ivp_eu.sh motion_tf.train.configs.poseidon_finetune_combo_158m_wavebank
"""
from .poseidon_finetune_combo_158m import Cfg as _Base


class Cfg(_Base):
    wave_bank = True
    wave_bank_K = 3
