"""Phase-2 pretrain, LEARNED-UPSAMPLE variant: identical to poseidon_pretrain_ivp EXCEPT learned high-freq
residual on the coarse->full W upsample (vs bilinear). Tests velocity high-freq recovery. Separate save_dir."""
from .poseidon_pretrain_ivp import Cfg as _Base


class Cfg(_Base):
    learned_upsample = True
    save_dir = "results/poseidon_pretrain_ivp_lup"
