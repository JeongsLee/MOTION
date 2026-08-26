"""Wave-Layer ablation finetune = poseidon_finetune_158m + the cross-channel ∇-coupling (wave) branch ON.

Tests: our late-frame wave-propagation weakness (t20 ~72% vs Poseidon ~40%) — is it the bank lacking an
explicit wave operator? The bank has Δ + per-channel ∇ but only IMPLICITLY mixes them via the deep conv;
the WaveExpert-style branch adds an explicit linear cross-channel ∇ coupling (the restoring force). zero-init
→ cont base loads identically (by NAME); finetune learns the coupling. If late frames improve, hypothesis holds.

init_from MUST be the NAMED-migrated cont base (new wave vars fresh-init-skipped on load).
Launch (1 GPU):
  FT_TASK=Wave-Layer FT_NSHOT=64 FT_UV_ONLY=0 FT_IC_FRAC=0 FT_TAG=_cont_wv FT_INIT_STEP=60000 \
  FT_INIT_PATH=/eu/results/migrated/cont260k_lr5e6_ckpt60000_named.npz \
  bash /code-vol/code/runners/_run_ivp_eu.sh motion_tf.train.configs.poseidon_finetune_158m_wave
"""
import os
from .poseidon_finetune_158m import Cfg as _Base


class Cfg(_Base):
    unified_wave = True
    unified_wave_hidden = int(os.environ.get("FT_WV_HIDDEN", "32"))
