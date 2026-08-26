"""dec base + PCGrad (family-conflict gradient surgery). = the winning relc+dec config (clamped rel-L2
loss, latent decoder n_latent=16, future-only, instance_norm, attn_router + route_film) with
pcgrad=True: each of the K=4 grad-accum micro-batches (= 1 random-family sample) is treated as a
separate task gradient; conflicting pairs (g_k·g_j<0) are projected off each other before summing →
removes destructive inter-family interference. Also logs the mean pairwise grad cosine = a direct
MEASURE of family conflict. Targets the NS-family gap (pde/com/incom 57-60%). Run on deneb-kr."""


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
    latent_decoder = True        # dec base (the winner)
    n_latent = 16
    decode_hidden = 32
    loss_mse = False             # relc base: clamped rel-L2
    loss_relclamp = 4.0
    pcgrad = True                # <- PCGrad family-conflict gradient surgery (+ logs grad cosine)
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
    grad_accum = 4               # K=4 micro-batches = 4 task gradients for PCGrad
    bf16 = False
    grad_checkpoint = False
    jit_compile = True
    save_dir = "results/prose_small5_attn_film_norm_fo_relc_dec_pcg"
    ckpt_every = 500
    seed = 0
