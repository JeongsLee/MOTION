"""CHARACTERISTICS + INTERFACE-REFLECTION bank arm (Wave-Layer diagnosis-matched). The loose_bank residual
is (x,t)-separable and the phase_bank is isotropic in radial κ — neither carries directional traveling
fronts nor interface-localized reflected/transmitted fronts, which is the layered-medium failure mode.
char_bank supplies 4 one-way semi-Lagrangian characteristic transports of the IC at a learned bounded local
speed c̃(x), plus the same carriers gated by a learned interface mask from |∇c̃|; a zero-init 1×1-conv
combiner mixes the stack per (x,τ) (residual ≡ 0 at load, name-keyed cb_* fresh vars). A/B partner:
ftwave-800k-x4 base (31.95) and the lb (32.33) / pb (recovering) arms, all 12800-step from the same ckpt.
Launch:
  FT_TASK=Wave-Layer FT_NSHOT=64 FT_UV_ONLY=0 FT_IC_FRAC=1.0 FT_LR=5e-5 FT_STEPS=12800 \
  FT_TAG=_162m800000cb FT_INIT_PATH=/eu/results/migrated/combo158m_800000_named.npz \
  bash _run_ivp_eu.sh motion_tf.train.configs.poseidon_finetune_combo_158m_charbank
"""
from .poseidon_finetune_combo_158m import Cfg as _Base


class Cfg(_Base):
    char_bank = True
    char_bank_smax = 32.0
