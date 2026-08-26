"""relc base + LATENT DECODER (NTO-ADA HiLatent). = fo baseline with clamped normalized rel-L2 loss
AND latent_decoder=True: experts output per-pixel W in n_latent=16 LATENT channels, the dual basis
integrates the latent W with ic=0 (residual), and a per-pixel NO-BIAS MLP (16→32→6) decodes the latent
trajectory to the 6 physical channels; field = IC + δ (hard-IC exact via no-bias decode(0)=0). The
rollout starts from a learned 1×1 encode of the IC. Decouples integration dim from physical channels +
(proven in NTO-ADA NS) sharply improves OOD generalization (ood_k 36.7→17.5%). Run on deneb-kr."""


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
    latent_decoder = True        # <- NTO-ADA HiLatent: latent-channel dynamics + no-bias MLP decode + residual
    n_latent = 16                # latent channels (NTO-ADA's value)
    decode_hidden = 32           # decoder MLP hidden width
    loss_mse = False             # relc base: rel-L2 loss ...
    loss_relclamp = 4.0          # ... clamped at ratio 4.0
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
    save_dir = "results/prose_small5_attn_film_norm_fo_relc_dec"
    ckpt_every = 500
    seed = 0
