"""Phase-2 IVP, latent_factor=1 (NO spatial compression — per-panel expert dynamics at full 128² instead of
32²). Tests whether the lf=4 coarse-grid->bilinear-upsample is what smooths velocity (dec/nodec ablation
showed the decoder is NOT the cause; lf=4 is the prime suspect). Watch NS-Sines learning vs the lf=4 main
run. lf=1 blows up the O(N_p) rollout activation ~16x -> use grad_checkpoint + swap_memory + batch=2 to fit
80GB. Separate save_dir; the main lf=4 run is untouched."""
from .poseidon_pretrain_ivp import Cfg as _Base


class Cfg(_Base):
    latent_factor = 1              # <- the test: full-res dynamics, no 32^2 compression
    # grad_checkpoint=True(recompute_grad) breaks inside the while_loop with route_film's captured var
    # ("internal_captures but not in internal_capture_to_output"). Use swap_memory instead: offloads the
    # N_p panel activations to HOST RAM during the while_loop -> fits 80GB at N_p=64 without recompute.
    grad_checkpoint = False
    swap_memory = True
    batch = 2                      # smaller batch (lf=1 is ~16x the rollout memory/compute)
    steps = 10000                  # enough evals (2k/4k/6k/8k/10k) to see the NS-Sines trend
    ckpt_every = 2000
    save_dir = "results/poseidon_pretrain_ivp_lf1"
