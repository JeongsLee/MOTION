"""A-rematch step 1: SAME 23k architecture, trained to convergence (10k steps).
Isolates whether the FNO gap was undertraining. FNO result reused from ace_compare json."""


class Cfg:
    Nx = 128
    Nt = 20
    T_final = 1.0
    domain_L = 1.0
    N_p = 16
    n_modes = 8
    max_order = 8
    d_geom = 48
    geo_layers = 2
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
    batch = 16
    lr = 1e-3
    jit_compile = True
    results_json = "/tmp/ace_compare_results.json"   # reuse stored FNO (9.0e-5)
