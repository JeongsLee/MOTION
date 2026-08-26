"""Mid-scale (~20M) fits-now scale-up of the proven 5M config. 100M is BLOCKED (recompute_grad +
tf.while_loop incompatible; bf16 diverges; fp32-N_p16 garbage). The fits-now lever = micro-batch=1
+ grad_accum so step memory = batch-1 cost (not batch-2), freeing headroom to widen the model while
KEEPING N_p=64 fp32 (the accuracy-critical knobs). Everything else identical to prose_multi_5M_full.

Width chosen to ride under the 80GB H100 at N_p=64 fp32 batch=1 (5M was ~37GB/sample). Step-1 mem
print + OOM-guard confirm; this config is the first empirical probe — widen further if it fits with
headroom, shrink if it OOMs."""


class Cfg:
    Nx = 128
    T_in = 10
    Nt = 11
    T_final = 1.0
    domain_L = 1.0
    N_p = 64                 # KEEP — accuracy-critical (N_p=16 gave 76% garbage)
    n_modes = 32
    max_order = 32
    d_geom = 416             # 224 -> 416 (~1.85x width)
    geo_layers = 6           # 5 -> 6
    expert_hidden = 352      # 192 -> 352
    n_channels = 6
    desc_dim = 6
    experts = ("convective", "diffusive", "reaction", "elliptic", "shock")
    split_basis = False
    lambda_gate = 1e-3
    datasets = ("shallow_water", "pdearena_ns", "diff_react", "com_ns")
    n_per = 100000           # ALL data
    t_num = 20
    stream = True
    steps = 8000
    batch = 1                # per-replica=1; eff = 1 x 8 x grad_accum(2) = 16 (old standard)
    lr = 2e-4                # slightly lower for the larger model (warm-restart LR-scale lesson)
    lr_decay = True
    clipnorm = 1.0
    grad_accum = 2           # eff-batch 16 with batch=1 x 8 GPU
    bf16 = False
    grad_checkpoint = False  # NOT used (recompute_grad+while_loop broken); rely on batch=1 memory
    jit_compile = True
    save_dir = "results/prose_multi_mid"
    ckpt_every = 1000
    seed = 0
