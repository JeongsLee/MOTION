"""A-rematch step 2: scaled-up model (~0.5M params) trained to convergence.
Closer to FNO's 2.37M (5× fewer vs the previous 100× gap). batch reduced for 12GB VRAM
(sequential N_p rollout holds all-panel activations — the local memory bottleneck)."""


class Cfg:
    Nx = 128
    Nt = 20
    T_final = 1.0
    domain_L = 1.0
    N_p = 16
    n_modes = 8
    max_order = 8
    d_geom = 128
    geo_layers = 4
    expert_hidden = 96
    experts = ("convective", "diffusive", "reaction")
    lambda_gate = 1e-3
    fno_width = 32
    fno_modes = 12
    fno_layers = 4
    S = 128
    n_val = 64
    n_test = 240
    max_steps = 10000
    eval_every = 250
    batch = 6
    lr = 1e-3
    jit_compile = True
    results_json = "/tmp/ace_compare_results.json"
