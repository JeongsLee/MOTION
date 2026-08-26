"""Transfer-favorable architecture (option A) = poseidon_finetune_158m + a GENERIC learnable local-operator
adapter ON (NOT a hardcoded physics operator). Tests whether a fresh, low-param, fluid-prior-decoupled
adapter improves OOD transfer (ACE reaction-diffusion, Wave hyperbolic) — generalizing the reaction result.

Same FT_TASK-parametrized base; works for any task (FT_TASK=ACE | Wave-Layer | ...). init_from MUST be the
NAMED-migrated cont base (new adapter vars fresh-init-skipped on load).
Launch (1 GPU):
  FT_TASK=ACE        FT_NSHOT=64 FT_UV_ONLY=0 FT_IC_FRAC=0 FT_TAG=_cont_ad FT_INIT_STEP=60000 \
  FT_INIT_PATH=/eu/results/migrated/cont260k_lr5e6_ckpt60000_named.npz \
  bash /code-vol/code/runners/_run_ivp_eu.sh motion_tf.train.configs.poseidon_finetune_158m_adapter
"""
import os
from .poseidon_finetune_158m import Cfg as _Base


class Cfg(_Base):
    unified_adapter = True
    unified_adapter_hidden = int(os.environ.get("FT_AD_HIDDEN", "48"))
