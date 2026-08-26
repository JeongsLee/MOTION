"""ACE reaction-ablation finetune = poseidon_finetune_158m + the pointwise reaction branch ON.

Tests the hypothesis: our operator bank has diffusion (Δ) + advection but NO pointwise reaction R(u) (e.g.
Allen-Cahn's u−u³), which plausibly explains losing ACE to Poseidon-B. Adding a zero-init reaction branch
(experts/unified.py, gated by unified_reaction) and finetuning should close the gap if the hypothesis holds.

init_from MUST be the NAMED-migrated cont base (so the new reaction vars are fresh-init-skipped on load,
not corrupted by positional index shift). Migrate first via motion_tf.utils._migrate_ckpt.

Launch (1 GPU):
  FT_TASK=ACE FT_NSHOT=64 FT_UV_ONLY=0 FT_IC_FRAC=0 FT_TAG=_cont_rx FT_INIT_STEP=60000 \
  FT_INIT_PATH=/eu/results/migrated/cont260k_lr5e6_ckpt60000_named.npz \
  bash /code-vol/code/runners/_run_ivp_eu.sh motion_tf.train.configs.poseidon_finetune_158m_reaction
"""
import os
from .poseidon_finetune_158m import Cfg as _Base


class Cfg(_Base):
    unified_reaction = True
    unified_reaction_hidden = int(os.environ.get("FT_RX_HIDDEN", "32"))
