"""Phase-1 Task-1, BOTH high-freq mechanisms: learned-upsample (coarse->full W) + img-refine (per-snapshot
2D high-freq corrector at output). Tests if they stack. Separate save_dir; baseline untouched."""
from .prose_150M_fluid5_6x import Cfg as _Base


class Cfg(_Base):
    learned_upsample = True
    img_refine = True
    refine_dim = 48
    save_dir = "results/prose_150M_fluid5_6x_lupref"
