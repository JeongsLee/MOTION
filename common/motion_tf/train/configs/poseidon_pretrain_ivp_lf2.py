"""Phase-2 IVP, latent_factor=2 (dynamics at 64^2 = 2x finer than lf=4's 32^2). Tests whether finer-grid
dynamics reduces the velocity-amplitude smoothing, WITHOUT OOM and WITHOUT changing N_p (=64). lf=2 rollout
is 4x lf=4's; batch 4->1 offsets it (~56GB) and keeps the temporal resolution identical for a clean
comparison vs the lf=4 main run. Separate save_dir; main run untouched."""
from .poseidon_pretrain_ivp import Cfg as _Base


class Cfg(_Base):
    latent_factor = 2              # 64^2 dynamics (vs lf=4 32^2); the test
    batch = 1                      # offsets the 4x rollout memory of lf=2 (keeps N_p=64)
    swap_memory = True             # safety margin
    grad_checkpoint = False        # (recompute_grad breaks with route_film in the while_loop)
    steps = 10000
    ckpt_every = 2000
    save_dir = "results/poseidon_pretrain_ivp_lf2"
