"""Phase-2 pretrain, IMG-REFINE variant: ADA + per-snapshot 2D high-freq corrector at the output."""
from .poseidon_pretrain_ivp import Cfg as _Base


class Cfg(_Base):
    img_refine = True
    refine_dim = 48
    save_dir = "results/poseidon_pretrain_ivp_imgref"
