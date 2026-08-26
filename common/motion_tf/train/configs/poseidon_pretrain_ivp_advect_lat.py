"""Phase-2 advect, LATENT KEPT ON — the clean 1-change ablation: add structured self-advection to the
lup baseline WITHOUT turning latent_decoder off. Rationale (user): self-advection −(z0·∂x z + z1·∂y z)
does NOT require physical velocity — ic_enc can LEARN latent channels 0,1 to be the advection-relevant
transport field, so we keep latent_decoder's generalization benefit AND get the advective inductive bias.
Upwind for stability + the NaN-skip guard in train_ivp (skip rare overflow batches instead of diverging)."""
from .poseidon_pretrain_ivp_lup import Cfg as _Base


class Cfg(_Base):
    # latent_decoder stays True (inherited) — NN learns the latent advection velocity in channels 0,1
    conv_self_advect = True
    conv_upwind = True
    save_dir = "results/poseidon_pretrain_ivp_advect_lat"
