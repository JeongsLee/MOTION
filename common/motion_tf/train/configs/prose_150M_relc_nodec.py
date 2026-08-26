"""150M FULL-DATA, DECODER-FREE ablation — identical to prose_150M_relc_dec EXCEPT latent_decoder=False.
Isolates the latent-decoder's effect at 150M scale: here the experts output per-pixel PHYSICAL W (6 ch),
the shared Fourier+Legendre basis integrates with ic=IC (hard-IC anchored in the basis), and the output
IS the field directly (decoder-free, like the original / NTO-ADA NTO-ADA base). Same clamped rel-L2 loss,
balanced sampling, instance_norm, future-only, bf16, latent_factor=4, full data. Run on deneb-kr ×2.
Head-to-head vs prose_150M_relc_dec (latent decoder)."""


class Cfg:
    Nx = 128
    T_in = 10
    Nt = 11
    T_final = 1.0
    domain_L = 1.0
    N_p = 64
    n_modes = 32
    max_order = 32
    d_geom = 1024
    geo_layers = 9
    expert_hidden = 960
    n_channels = 6
    desc_dim = 6
    experts = ("convective", "diffusive", "reaction", "elliptic", "shock", "helmholtz")
    split_basis = False
    field_gate = False
    input_router = False
    attn_router = True
    route_context = False
    route_film = True
    latent_decoder = False       # <- DECODER-FREE (the ablation): per-pixel physical W → shared basis → field
    loss_mse = False
    loss_relclamp = 4.0
    router_dim = 256
    router_depth = 2
    router_heads = 4
    router_patch = 2
    lambda_gate = 1e-4
    latent_factor = 4
    datasets = ("shallow_water", "pdearena_ns", "diff_react", "com_ns", "incom_ns")
    n_per = 100000
    t_num = 20
    stream = True
    steps = 15000
    batch = 1
    lr = 2e-4
    lr_decay = True
    warmup = 500
    instance_norm = True
    clipnorm = 1.0
    grad_accum = 4
    bf16 = True
    grad_checkpoint = False
    jit_compile = True
    swap_memory = False
    save_dir = "results/prose_150M_relc_nodec"
    ckpt_every = 1000
    seed = 0
