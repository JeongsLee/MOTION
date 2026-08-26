"""~150M PURE-ROUTER scale-up (descriptor removed — label-free field routing for both forecasting
and transfer axes). — the PROSE-comparable budget. fp32 (SciML standard) made to fit at N_p=64 via
tf.while_loop(swap_memory=True): the rollout's O(N_p) forward activations are offloaded to HOST RAM and
brought back in the backward pass — fp32-exact, no precision change. Requires jit_compile=False (XLA
ignores swap_memory). descriptor (given) gate — accuracy-first, in-dist mode.

First use = a FIT/STABILITY TEST on the small n900 5-family set (fast cache): confirm (1) params ≈150M,
(2) fits with swap_memory (no OOM), (3) trains finite. Then scale data + steps for the real run."""


class Cfg:
    Nx = 128
    T_in = 10
    Nt = 11
    T_final = 1.0
    domain_L = 1.0
    N_p = 64
    n_modes = 32
    max_order = 32
    d_geom = 1024            # 832(92M) -> 1024
    geo_layers = 9           # 8 -> 9
    expert_hidden = 960      # 768 -> 960   (target ~150M; verify param print)
    n_channels = 6
    desc_dim = 6
    experts = ("convective", "diffusive", "reaction", "elliptic", "shock")
    split_basis = False
    field_gate = False
    input_router = True       # PURE field router — descriptor REMOVED (label-free, both axes native)
    lambda_gate = 1e-4        # light — router (no sparsity starve)
    datasets = ("shallow_water", "pdearena_ns", "diff_react", "com_ns", "incom_ns")
    n_per = 100000           # FULL data
    t_num = 20
    stream = True
    steps = 15000
    batch = 1               # per-replica
    lr = 1e-4
    lr_decay = True
    warmup = 1500
    instance_norm = False
    clipnorm = 1.0
    grad_accum = 4           # eff-batch = 1 x 8GPU x 4 = 32 (~$210 budget)
    bf16 = False            # fp32
    grad_checkpoint = False
    jit_compile = False     # jit XLA temp pushed 8-GPU LF=2 over 80GB → off (slower, fits)
    swap_memory = False
    latent_factor = 2       # 64² latent — recover turbulent-NS fine scale (vs LF=4 32²)
    save_dir = "results/prose_150M_LF2_full"
    ckpt_every = 1500
    seed = 0
