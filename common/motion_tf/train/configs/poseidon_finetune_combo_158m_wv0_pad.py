"""DEFINITIVE WAVE ARM — closed benchmark x endpoint fix x full anchor coverage: (i) v0-slot input
([u,v,c], the SM-defined operator restored), (ii) temporal padding (basis span +2 frames, data-final
interior), (iii) training pairs from i>=0 with v0=0 at i=0 (the true zero-velocity IC) so anchor-0 is
in-distribution. Updates the corrected-benchmark Wave row (wv0: 20.5 anchor-1 full-horizon / 11.5-12.8
swept lead12 with an anchor-0 hole).
Launch:
  FT_TASK=Wave-Layer FT_NSHOT=64 FT_UV_ONLY=0 FT_IC_FRAC=1.0 FT_LR=5e-5 FT_STEPS=12800 \
  FT_TAG=_162m800000wv0pad FT_INIT_PATH=/eu/results/poseidon_pretrain_ivp_combo_158m_scratch/ckpt_800000.npz \
  bash _run_ivp_eu.sh motion_tf.train.configs.poseidon_finetune_combo_158m_wv0_pad
"""
from .poseidon_finetune_combo_158m_wave_v0 import Cfg as _Base


class Cfg(_Base):
    tin2_anchor0 = True
    nt_data = 21
    Nt = 23
    T_final = 1.1
