"""Phase-1 ACE (Allen-Cahn) few-shot config — 128², Poseidon downstream task.
Sample-efficiency: train our small model on N trajectories, eval on the official 240 test."""


class Cfg:
    Nx = 128
    Nt = 20                  # ACE has 20 time steps
    T_final = 1.0            # normalized segment (physical T absorbed into learned W)
    domain_L = 1.0           # ACE unit square
    N_p = 16
    n_modes = 8
    max_order = 8
    d_geom = 48
    experts = ("convective", "diffusive", "reaction")
    # few-shot sweep: train-set sizes
    N_list = (16, 64, 256)
    n_test = 240
    seed = 0
    steps = 800
    batch = 16
    lr = 1e-3
    lambda_gate = 1e-3
    jit_compile = True
    save_dir = "results/ace_fewshot"
