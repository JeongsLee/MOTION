"""Quick local 2-D compressible Euler shock validation (4-channel ρ,ρu,ρv,E)."""


class Cfg:
    Nx = 64
    Nt = 11
    T_final = 1.0
    domain_L = 1.0
    N_p = 12
    n_modes = 6
    max_order = 6
    d_geom = 32
    geo_layers = 2
    n_channels = 4                 # ρ, ρu, ρv, E
    experts = ("shock", "diffusive")
    lambda_gate = 1e-3
    n_train = 128
    n_test = 32
    steps = 2500
    batch = 8
    lr = 1e-3
    jit_compile = True
    seed = 0
