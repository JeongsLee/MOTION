"""ACE S=128 convergence-based comparison config (Poseidon AG convention).
Same task/metric/protocol for our model and the FNO reference."""


class Cfg:
    Nx = 128
    Nt = 20
    T_final = 1.0
    domain_L = 1.0
    # our model
    N_p = 16
    n_modes = 8
    max_order = 8
    d_geom = 48
    experts = ("convective", "diffusive", "reaction")
    lambda_gate = 1e-3
    # FNO baseline
    fno_width = 32
    fno_modes = 12
    fno_layers = 4
    # protocol (Poseidon AG uses S=128 for time-dependent PDEs)
    S = 128
    n_val = 64
    n_test = 240
    max_steps = 3000
    eval_every = 100
    batch = 16
    lr = 1e-3
    jit_compile = True
    results_json = "/tmp/ace_compare_results.json"
