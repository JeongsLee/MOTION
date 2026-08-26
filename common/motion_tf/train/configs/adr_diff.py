"""ADR with a STRONG-diffusion regime — tests whether the diffusive gate specializes once
diffusion is actually visible in the data (diagnosis of the weak-ν miss). DESIGN.md §6.4."""


class Cfg:
    Nx = 32
    Nt = 16
    T_final = 1.0
    N_p = 16
    n_modes = 8
    max_order = 8
    d_geom = 24
    experts = ("convective", "diffusive", "reaction")
    n_train = 128
    n_val = 64
    seed = 0
    steps = 600
    batch = 16
    lr = 1e-3
    lambda_gate = 1e-3
    jit_compile = True
    # strong diffusion so ν·k²·T is O(1) → diffusion clearly smooths the field
    a_range = (-1.0, 1.0)
    nu_range = (0.05, 0.25)
    r_range = (0.0, 2.0)
    save_dir = "results/adr_diff"
