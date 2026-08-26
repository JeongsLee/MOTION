"""100M — SAME proven config as 5M (prose_multi_5M_full): N_p=64, n_modes=32, fp32, eff batch 16
(batch=2 x 8 GPU), shared-W, lr3e-4, FULL data. ONLY model size differs (d_geom 224->832,
expert_hidden 192->768, geo_layers 5->8 ~ 90M). gradient_checkpoint=True is the memory enabler
(N_p=64 fp32 at this width is ~300GB without it); recompute_grad ⊥ XLA so jit OFF. while_loop keeps
the graph O(1) (fast build). Slower per-step (no XLA + recompute) but fits + accurate + stable.
Clean 100M-vs-5M capacity comparison: same everything except size."""


class Cfg:
    Nx = 128
    T_in = 10
    Nt = 11
    T_final = 1.0
    domain_L = 1.0
    N_p = 64
    n_modes = 32
    max_order = 32
    d_geom = 832
    geo_layers = 8
    expert_hidden = 768
    n_channels = 6
    desc_dim = 6
    experts = ("convective", "diffusive", "reaction", "elliptic", "shock")
    split_basis = False
    lambda_gate = 1e-3
    datasets = ("shallow_water", "pdearena_ns", "diff_react")
    n_per = 100000
    t_num = 20
    stream = True
    steps = 8000
    batch = 2                # eff 16 (match 5M); checkpoint frees memory so batch=2 fits
    lr = 3e-4
    lr_decay = True
    clipnorm = 1.0
    grad_accum = 1
    bf16 = False             # fp32 (bf16 diverged)
    grad_checkpoint = True   # memory enabler (recompute N_p panels in backward)
    jit_compile = False      # recompute_grad ⊥ XLA; while_loop keeps build O(1)
    save_dir = "results/prose_multi_100M_v2"   # fresh (no stale ckpt)
    ckpt_every = 1000
    seed = 0
