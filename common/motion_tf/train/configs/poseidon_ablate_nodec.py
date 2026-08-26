"""ABLATION: test whether the latent DECODER causes velocity-amplitude smoothing. Identical to
poseidon_pretrain_ivp EXCEPT latent_decoder=False (experts->basis emit the 8 physical channels directly,
no latent-16 bottleneck + no-bias decode MLP). Short run (6000 steps) to compare NS velocity pred-std vs
the main run at the same step. Separate save_dir so the main pretraining is untouched."""
from .poseidon_pretrain_ivp import Cfg as _Base


class Cfg(_Base):
    latent_decoder = False         # <-- the only change: direct per-channel basis output, no decoder
    steps = 6000
    ckpt_every = 2000
    save_dir = "results/poseidon_ablate_nodec"
