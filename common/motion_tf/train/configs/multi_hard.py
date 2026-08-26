"""Hard-physics multi-family foundation: heat·ADR·NS·Euler in ONE model (common 4-channel
layout). Full operator basis; gate must discriminate diffusive vs conv/react vs elliptic vs shock."""


class Cfg:
    Nx = 64
    Nt = 11
    T_final = 1.0
    domain_L = 1.0
    N_p = 12
    n_modes = 6
    max_order = 6
    d_geom = 40
    geo_layers = 2
    n_channels = 4
    desc_dim = 4                 # family one-hot (4 families)
    experts = ("convective", "diffusive", "reaction", "elliptic", "shock")
    lambda_gate = 1e-3
    n_per_train = 96
    n_per_test = 24
    steps = 4000
    batch = 8
    lr = 1e-3
    grad_checkpoint = False      # local: model fits → prefer jit speed (checkpointing validated
                                 # at 2.7GB, reserved for H100 scale where memory binds)
    jit_compile = True
    seed = 0
