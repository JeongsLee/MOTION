"""SMALL architecture-iteration config: 5 families x 900 samples, 1 GPU, fast cache. NOT for SOTA —
for quick gate/expert/architecture comparisons (does it diverge? does the gate specialize? rough
per-family rel-L2 + α behavior). Clone this and flip field_gate / input_router for ablations.

Run: resourcespec-ch100x1 (1 GPU) + _run_prose_small.sh. eff-batch = 2 x 1 x grad_accum(2) = 4.
Cache = the tiny n900 5-family prebuilt (copies in ~1min vs ~30min for the full set)."""


class Cfg:
    Nx = 128
    T_in = 10
    Nt = 11
    T_final = 1.0
    domain_L = 1.0
    N_p = 64
    n_modes = 32
    max_order = 32
    d_geom = 224
    geo_layers = 5
    expert_hidden = 192
    n_channels = 6
    desc_dim = 6
    experts = ("convective", "diffusive", "reaction", "elliptic", "shock")
    split_basis = False
    field_gate = True
    gate_desc_dropout = 0.2
    input_router = False
    lambda_gate = 1e-3
    datasets = ("shallow_water", "pdearena_ns", "diff_react", "com_ns", "incom_ns")
    n_per = 900                  # small fixed set per family → tiny cache, fast iteration
    t_num = 20
    stream = True
    steps = 3000                 # enough to read gate behavior + rough per-family
    batch = 2                    # per-replica; 1 GPU → eff = 2 x 1 x accum(2) = 4
    lr = 2e-4
    lr_decay = True
    warmup = 300
    instance_norm = True
    clipnorm = 1.0
    grad_accum = 2
    bf16 = False
    grad_checkpoint = False
    jit_compile = True
    save_dir = "results/prose_small5_field_norm"
    ckpt_every = 500
    seed = 0
