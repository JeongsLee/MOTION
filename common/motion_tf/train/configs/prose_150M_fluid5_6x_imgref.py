"""Phase-1 Task-1, IMG-REFINE variant: ADA + a per-snapshot 2D image-to-image high-freq corrector at the
output (temporal spectral stays with the ADA basis). Isolates the refiner (learned_upsample stays False).
Separate save_dir; original Task-1 untouched."""
from .prose_150M_fluid5_6x import Cfg as _Base


class Cfg(_Base):
    img_refine = True
    refine_dim = 48
    save_dir = "results/prose_150M_fluid5_6x_imgref"
