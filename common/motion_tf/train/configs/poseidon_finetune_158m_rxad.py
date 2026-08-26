"""ACE transfer with reaction + adapter BOTH on, Poseidon-style discriminative LR: the FRESH operator
branches (u_rx reaction, u_ad adapter) + output heads get a HIGH lr (like Poseidon's re-initialized recovery
head), the pretrained backbone gets a LOW lr (gently adapt, preserve fluid knowledge). Mirrors Poseidon-B's
finetune (lr_embedding_recovery=5e-4 / backbone 5e-5).

init_from = NAMED-migrated cont base (u_rx + u_ad fresh-init-skipped on load).
Launch (1 GPU):
  FT_TASK=ACE FT_NSHOT=64 FT_UV_ONLY=0 FT_IC_FRAC=0 FT_TAG=_rxad_disc FT_INIT_STEP=60000 \
  FT_LR=1e-5 FT_LR_DEC=5e-4 FT_INIT_PATH=/eu/results/migrated/cont260k_lr5e6_ckpt60000_named.npz \
  bash /code-vol/code/runners/_run_ivp_eu.sh motion_tf.train.configs.poseidon_finetune_158m_rxad
"""
import os
from .poseidon_finetune_158m import Cfg as _Base


class Cfg(_Base):
    unified_reaction = True
    unified_adapter = True
    lr = float(os.environ.get("FT_LR", "1e-5"))            # backbone (low)
    lr_decoder = float(os.environ.get("FT_LR_DEC", "5e-4"))  # fresh op branches + heads (high, Poseidon recovery-head style)
    decoder_keys = ("u_rx", "u_ad", "u_out", "decode_h", "decode_out")  # what gets the high LR
