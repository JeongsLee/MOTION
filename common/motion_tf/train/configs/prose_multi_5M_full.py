"""5M on FULL data with the PROVEN-GOOD config (matches the original run that hit NS 22% @900):
N_p=64, n_modes=32, fp32, eff batch 16 (batch=2 x 8 GPU, NO grad_accum), lr 3e-4, shared-W.
ONLY change vs that baseline = FULL data (pdearena conditioned 3712 vs 900) → isolates the data
effect. stream=True (from_generator avoids the 45GB from_tensor_slices GPU constant; N_p=64 compute
alone is ~74.5GB so the data must NOT also sit on the GPU)."""


class Cfg:
    Nx = 128
    T_in = 10
    Nt = 11
    T_final = 1.0
    domain_L = 1.0
    N_p = 64                 # restored — needed for integration fidelity (N_p=16 gave 76% garbage)
    n_modes = 32
    max_order = 32
    d_geom = 224
    geo_layers = 5
    expert_hidden = 192
    n_channels = 6
    desc_dim = 6
    experts = ("convective", "diffusive", "reaction", "elliptic", "shock")
    split_basis = False      # match the proven shared-W baseline (clean gate); split_basis tested separately
    lambda_gate = 1e-3
    datasets = ("shallow_water", "pdearena_ns", "diff_react")
    n_per = 100000           # ALL (SWE 1000 + pdearena conditioned 3712 + diff_react 1000)
    t_num = 20
    stream = True            # no 45GB GPU constant (N_p=64 compute ~74.5GB already)
    steps = 8000
    batch = 2                # per-replica; eff = 2 x 8 = 16 (old standard), grad_accum=1
    lr = 3e-4
    lr_decay = True
    clipnorm = 1.0
    grad_accum = 1
    bf16 = False             # fp32 (SciML standard; bf16 diverged)
    grad_checkpoint = False
    jit_compile = True
    save_dir = "results/prose_multi_5M_full"
    ckpt_every = 1000
    seed = 0
