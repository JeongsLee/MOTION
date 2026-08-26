"""PROSE-FD SWE ~10M VESSL run (H100). 128², N_p=64, intermediate ckpt every 2000 steps."""


class Cfg:
    Nx = 128
    T_in = 10
    Nt = 11
    T_final = 1.0
    domain_L = 1.0
    N_p = 64                 # settled
    n_modes = 32
    max_order = 32
    d_geom = 288             # ~10M (run prints actual count)
    geo_layers = 5
    expert_hidden = 224
    n_channels = 6
    desc_dim = 6
    experts = ("convective", "diffusive", "reaction", "elliptic", "shock")
    lambda_gate = 1e-3
    # data
    dataset = "shallow_water"
    n_total = 900
    t_num = 20
    t_step = 5
    # optim
    steps = 6000             # first VESSL validation run; extend after confirming
    batch = 8
    lr = 3e-4
    grad_checkpoint = True   # fit 128²·N_p64·C6·10M on H100
    jit_compile = False      # XLA compile of N_p=64 unrolled rollout is prohibitive → graph mode
    # checkpointing
    save_dir = "results/prose_swe_10M"
    ckpt_every = 2000
    seed = 0
