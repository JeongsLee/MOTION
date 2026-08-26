"""ADR specialization config — CPU-runnable; validates that the family gate recovers the
active operators (DESIGN.md §6.4, §7)."""


class Cfg:
    # grid / time
    Nx = 32
    Nt = 16
    T_final = 1.0
    # ADA basis
    N_p = 16
    n_modes = 8
    max_order = 8
    # model
    d_geom = 24
    experts = ("convective", "diffusive", "reaction")
    # data
    n_train = 128
    n_val = 64
    seed = 0
    # optim
    steps = 600
    batch = 16
    lr = 1e-3
    lambda_gate = 1e-3          # gate-sparsity (L1 on α) → specialization
    save_dir = "results/adr_smoke"
