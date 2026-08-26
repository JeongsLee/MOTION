"""TEMPORAL-PADDING finetune arm (basis-endpoint hypothesis): the data-final frame normally sits at the
ADA basis ENDPOINT (tau=1) — the worst-conditioned point of both temporal bases (Fourier periodic wrap,
Legendre maximal oscillation). Extend the basis span past the data horizon (T_final 1.0 -> 1.1, Nt 21 -> 23
at the SAME 1/20 frame spacing; data frames 0..20 align with model frames 0..20, trailing 2 pad frames
unsupervised) so frame 20 becomes an interior point. Tests whether the @final deficit is basis-boundary
conditioning. A/B partners: ftns-800k-x4 base (5.69, @final 17.0) and ftns-leadw (gradient-balance axis).
Launch:
  FT_TASK=NS-PwC FT_NSHOT=64 FT_UV_ONLY=1 FT_IC_FRAC=1.0 FT_LR=5e-5 FT_STEPS=12800 FT_TAG=_162m800000pad \
  FT_INIT_PATH=/eu/results/poseidon_pretrain_ivp_combo_158m_scratch/ckpt_800000.npz \
  bash _run_ivp_eu.sh motion_tf.train.configs.poseidon_finetune_combo_158m_pad
"""
from .poseidon_finetune_combo_158m import Cfg as _Base


class Cfg(_Base):
    Nt = 23
    T_final = 1.1
    nt_data = 21
