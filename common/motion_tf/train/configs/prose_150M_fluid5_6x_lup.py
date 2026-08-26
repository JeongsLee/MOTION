"""Phase-1 Task-1, LEARNED-UPSAMPLE variant: identical to prose_150M_fluid5_6x EXCEPT the coarse->full
per-panel W upsample uses bilinear + a learned zero-init high-freq residual (sub-pixel conv) instead of
pure bilinear. Tests whether learned upsampling recovers velocity high-freq (NTO-ADA uses Conv2DTranspose;
our baseline uses bilinear = low-pass). Separate save_dir; the original Task-1 run is untouched."""
from .prose_150M_fluid5_6x import Cfg as _Base


class Cfg(_Base):
    learned_upsample = True
    save_dir = "results/prose_150M_fluid5_6x_lup"
