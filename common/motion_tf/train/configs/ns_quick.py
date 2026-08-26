"""Quick local NS Elliptic-expert validation — reduced N_p / modes / orders / samples."""


class Cfg:
    Nx = 32
    Nt = 16              # subsample from 33 frames
    T_final = 1.0
    domain_L = 1.0
    N_p = 12             # reduced (NTO-ADA NS used 64)
    n_modes = 6          # reduced
    max_order = 6        # reduced
    d_geom = 32
    geo_layers = 2
    experts = ("convective", "diffusive", "elliptic")
    n_channels = 1
    lambda_gate = 1e-3
    n_train = 256
    n_test = 128
    nu_range = (2.5e-3, 5e-3)   # narrow ν band (high-ν, more diffusive/smoother → quick)
    steps = 2500
    batch = 16
    lr = 1e-3
    jit_compile = True
