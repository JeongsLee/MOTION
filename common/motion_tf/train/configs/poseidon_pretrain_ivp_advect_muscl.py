"""Phase-2 advect, latent-ON + MUSCL (minmod) self-advection — 2nd-order TVD advection (low numerical
diffusion) instead of 1st-order upwind. In Phase-1 the MUSCL arm overtook the upwind arm by step 2000
(incom 7.86 vs 8.38), so test the same scheme in the IVP/Poseidon pretrain. Same as ..._advect_lat
except minmod instead of upwind. NaN-skip guard in train_ivp covers rare overflow batches."""
from .poseidon_pretrain_ivp_lup import Cfg as _Base


class Cfg(_Base):
    # latent_decoder stays True (inherited) — ic_enc learns the latent advection velocity
    conv_self_advect = True
    conv_muscl = True              # minmod 2nd-order TVD (precedence over upwind)
    save_dir = "results/poseidon_pretrain_ivp_advect_muscl"
