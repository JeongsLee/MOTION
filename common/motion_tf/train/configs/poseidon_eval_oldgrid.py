"""EVAL-ONLY one ckpt of the OLD 800k scratch run (steps=1 < ckpt -> resume -> eval only).
save_dir is a staging dir holding a single copied ckpt; set via OLDEVAL_DIR env."""
import os
from .poseidon_pretrain_ivp_combo_158m_scratch import Cfg as _B


class Cfg(_B):
    steps = 1
    ckpt_every = 0
    eval_every = 0
    save_dir = os.environ.get("OLDEVAL_DIR", "/eu/results/_oldgrid_eval")
