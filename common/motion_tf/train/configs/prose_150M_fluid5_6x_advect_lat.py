"""Phase-1 Task-1 advect, LATENT KEPT ON — clean 1-change ablation of the reported ADA (fluid5_6x_lup):
add structured self-advection without turning latent_decoder off. ic_enc learns the latent advection
velocity (channels 0,1); keeps latent's generalization benefit + the advective inductive bias. Upwind
for stability. Compare per-family physical rel-L2 vs the lup baseline (and vs the latent-OFF advect arm)."""
from .prose_150M_fluid5_6x_lup import Cfg as _Base


class Cfg(_Base):
    conv_self_advect = True
    conv_upwind = True
    save_dir = "results/prose_150M_fluid5_6x_advect_lat"
