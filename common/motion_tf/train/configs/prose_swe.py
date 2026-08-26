"""PROSE-FD SWE local smoke (small) — validate the real-data pipeline (6-slot, T_in=10)."""


class Cfg:
    Nx = 128
    T_in = 10
    Nt = 11                  # IC(frame9) + 10 future
    T_final = 1.0
    domain_L = 1.0
    N_p = 16                 # reduced for local 12GB smoke (VESSL uses 64)
    n_modes = 8
    max_order = 8
    d_geom = 48
    geo_layers = 2
    n_channels = 6
    desc_dim = 6
    experts = ("convective", "diffusive", "reaction", "elliptic", "shock")
    lambda_gate = 1e-3
    # data
    dataset = "shallow_water"
    n_total = 400
    t_num = 20               # 10 input + 10 output window
    t_step = 5
    # optim
    steps = 300
    batch = 2                # reduced for local 12GB smoke
    lr = 1e-3
    jit_compile = False      # checkpoint path (recompute_grad + XLA finicky)
    grad_checkpoint = True   # rollout activations recomputed → fits 12GB
    seed = 0
