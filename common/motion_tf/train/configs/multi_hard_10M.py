"""~10M architecture scale-up test on hard multi-family (heat·ADR·NS·Euler).
Validates the architecture scales to ~10M and trains stably. N_p=64/n_modes=32 (settled).
grad_checkpoint ON to fit the big config on local 12GB (recompute_grad → VRAM ∝ 1 panel)."""


class Cfg:
    Nx = 64
    Nt = 11
    T_final = 1.0
    domain_L = 1.0
    N_p = 64                     # settled
    n_modes = 32                 # = N_p/2 (Nyquist)
    max_order = 32
    d_geom = 320                 # scaled (was 40) → ~10M
    geo_layers = 5
    expert_hidden = 256
    n_channels = 4
    desc_dim = 4
    experts = ("convective", "diffusive", "reaction", "elliptic", "shock")
    lambda_gate = 1e-3
    n_per_train = 64
    n_per_test = 16
    steps = 600
    batch = 4
    lr = 5e-4                    # smaller lr for bigger model (warm-restart lore)
    grad_checkpoint = True       # fit ~10M·N_p=64·C4·64² on 12GB
    jit_compile = False          # recompute_grad path
    seed = 0
