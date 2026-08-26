"""SMALL architecture-iteration config: LOSS ablation — clamped normalized relative-L2.

Identical to prose_small5_attn_film_norm_fo (future-only / IC=last, instance_norm, 6 experts +
attn_router + route_film) EXCEPT the LOSS: instead of MSE, use the per-frame normalized rel-L2 with a
ratio CLAMP (loss_relclamp=4.0). Rationale: our EVAL metric IS rel-L2, so training on (a stabilized)
rel-L2 directly optimizes it + keeps per-frame scale-invariance (small-amplitude frames/families not
drowned by high-amplitude bulk). The clamp caps the per-frame ratio at 4.0 (≈200% rel-L2) so a frame
whose normalized target ≈ its own mean (den→0) can't blow up the loss (the 180-init-spike pathology
that drove us to MSE). Comparison is uncontrolled (goal = beat PROSE's rel-L2, not match its recipe) →
loss is a free hyperparameter.

Run on deneb-kr. Head-to-head vs prose_small5_attn_film_norm_fo (MSE, 42.9% T=10 @3000)."""


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
    experts = ("convective", "diffusive", "reaction", "elliptic", "shock", "helmholtz")
    split_basis = False
    field_gate = False
    input_router = False
    attn_router = True
    route_context = False
    route_film = True
    loss_mse = False             # <- use rel-L2 loss ...
    loss_relclamp = 4.0          # <- ... clamped at ratio 4.0 (≈200% rel-L2) to kill the small-norm blow-up
    router_dim = 192
    router_depth = 2
    router_heads = 4
    router_patch = 8
    lambda_gate = 1e-4
    datasets = ("shallow_water", "pdearena_ns", "diff_react", "com_ns", "incom_ns")
    n_per = 900
    t_num = 20
    stream = True
    steps = 3000
    batch = 1
    lr = 2e-4
    lr_decay = True
    warmup = 300
    instance_norm = True
    clipnorm = 1.0
    grad_accum = 4
    bf16 = False
    grad_checkpoint = False
    jit_compile = True
    save_dir = "results/prose_small5_attn_film_norm_fo_relc"
    ckpt_every = 500
    seed = 0
