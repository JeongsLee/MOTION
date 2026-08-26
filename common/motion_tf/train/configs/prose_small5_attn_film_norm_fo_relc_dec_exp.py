"""relc base + LATENT DECODER + EXPERT FIXES — the combined "everything" bet. = relc (clamped rel-L2)
with latent_decoder=True (n_latent=16, NTO-ADA HiLatent: latent-channel dynamics + no-bias MLP decode
+ residual hard-IC) AND the two operator fixes: conv_self_advect=True (self-advection u·∇z) + the
"wave" cross-channel-∇ expert. NOTE: in latent mode the experts act on the 16-ch LATENT state, so
self-advection / wave become learned nonlinear couplings in latent space (not physically grounded, but
still expressive operators). Run on deneb-kr. Compare vs relc, relc+dec, relc+exp."""


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
    experts = ("convective", "diffusive", "reaction", "elliptic", "shock", "helmholtz", "wave")
    conv_self_advect = True
    latent_decoder = True
    n_latent = 16
    decode_hidden = 32
    split_basis = False
    field_gate = False
    input_router = False
    attn_router = True
    route_context = False
    route_film = True
    loss_mse = False
    loss_relclamp = 4.0
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
    save_dir = "results/prose_small5_attn_film_norm_fo_relc_dec_exp"
    ckpt_every = 500
    seed = 0
