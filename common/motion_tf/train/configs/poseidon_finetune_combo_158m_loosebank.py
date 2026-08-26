"""LOOSE physics-bank arm (FT_TASK-parametrized): physics enters as a FEATURE VOCABULARY (state powers,
Δ/Δ², c̃²-weighted Laplacians, hidden v0, |∇u|/κ|∇u|/∇c̃²·∇u) and the combination is LEARNED — a separable
residual Σ_m w_m(t)·B_m(x) (basis maps from a 1×1-conv MLP, time weights from a Fourier lead-time MLP,
zero-init → no-op at load). Successor to the RIGID wave_bank/ace_bank arms (analytic form + scalar gains):
wave_bank TIED its base (37.34 vs 37.28) and ace_bank trailed its base — consistent with the project
finding that LEARNED bias >> explicit operators; this arm tests the same vocabulary with a free form.
~44k params, name-keyed load (lb_* fresh-init). A/B partners: ft{ace,wave}-800k-x4 (12800-step bases).
Launch (e.g. ACE):
  FT_TASK=ACE FT_NSHOT=64 FT_UV_ONLY=0 FT_IC_FRAC=1.0 FT_LR=5e-5 FT_STEPS=12800 FT_TAG=_162m800000lb \
  FT_INIT_PATH=/eu/results/migrated/combo158m_800000_named.npz \
  bash _run_ivp_eu.sh motion_tf.train.configs.poseidon_finetune_combo_158m_loosebank
"""
from .poseidon_finetune_combo_158m import Cfg as _Base


class Cfg(_Base):
    loose_bank = True
    loose_bank_M = 8
