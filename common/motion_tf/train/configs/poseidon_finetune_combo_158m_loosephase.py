"""LOOSE SPECTRAL-PHASE bank arm (diagnosis-matched, Wave-Layer): the two prior bank arms did NOT match
the phase-drift diagnosis — rigid wave_bank prescribes the propagator but TAYLOR-truncates the phase
(polynomial in t cannot track long-horizon oscillation), and loose_bank's separable residual Σ w_m(t)B_m(x)
cannot represent traveling waves (u(x−ct) is not low-rank in (x,t)). This arm supplies the correct FUNCTION
CLASS as a learnable vocabulary: non-separable phase carriers {sin/cos(c_i κτ), e^{−g_i κτ}, e^{−g_i κ²τ}}
on the radial wavenumber κ=|k| with LEARNED speeds/rates, combined by a zero-init MLP into (κ,τ)-multipliers
(G1,G2) on û0(k) and a learned hidden v̂0(k). The exact d'Alembert propagator lives inside the space
(G1=cos(cκτ), G2∝sinc); dispersion/damping/mode-mixing are free. ~3k params, name-keyed load (pb_* fresh).
A/B partner: ftwave-800k-x4 (12800-step base, 31.95).
Launch:
  FT_TASK=Wave-Layer FT_NSHOT=64 FT_UV_ONLY=0 FT_IC_FRAC=1.0 FT_LR=5e-5 FT_STEPS=12800 FT_TAG=_162m800000pb \
  FT_INIT_PATH=/eu/results/migrated/combo158m_800000_named.npz \
  bash _run_ivp_eu.sh motion_tf.train.configs.poseidon_finetune_combo_158m_loosephase
"""
from .poseidon_finetune_combo_158m import Cfg as _Base


class Cfg(_Base):
    phase_bank = True
    phase_bank_S = 4
