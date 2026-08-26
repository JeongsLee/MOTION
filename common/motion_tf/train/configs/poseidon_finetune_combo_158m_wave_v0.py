"""WAVE 2-FRAME-IC arm (v0-slot injection): the wave equation's Cauchy data is (u, du/dt) but the IVP
protocol feeds a single u snapshot — the one task in the program whose observed state does not close the
dynamics (the lone Benchmark-II loss, and the failure mode no finetune-attached vocabulary could fix).
This arm injects the OBSERVED time derivative v0 = u_1 − u_0 (slot2 diff) into the unused dummy slot3 of
the same 8-slot input: channel count and positional ckpt load are unchanged; train pairs anchor at i≥1;
eval anchors at frame 1 and scores data frames 2..20 (drops the easiest frame — conservative vs base).
DIAGNOSIS arm, not protocol-fair vs Poseidon (extra input frame) — tests whether the ~32% wave plateau is
the hidden-state ill-posedness. A/B partner: ftwave-800k-x4 (base 31.95, single-frame).
Launch:
  FT_TASK=Wave-Layer FT_NSHOT=64 FT_UV_ONLY=0 FT_IC_FRAC=1.0 FT_LR=5e-5 FT_STEPS=12800 \
  FT_TAG=_162m800000wv0 FT_INIT_PATH=/eu/results/poseidon_pretrain_ivp_combo_158m_scratch/ckpt_800000.npz \
  bash _run_ivp_eu.sh motion_tf.train.configs.poseidon_finetune_combo_158m_wave_v0
"""
from .poseidon_finetune_combo_158m import Cfg as _Base


class Cfg(_Base):
    v0_from_slot = 2       # Wave-Layer QoI slot
    v0_to_slot = 3         # unused dummy slot receives the observed du/dt
