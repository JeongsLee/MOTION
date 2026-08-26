"""fluid5 with the 6-EXPERT set (the all-8 ablation showed wave+forcing don't help at matched samples →
revert to the proven 6). PROSE/BCAT-matched 5 fluid families (SWE/CNS/INS/NS-cond/CFDBench, no diff_react).
This is OUR model for the data-efficiency curve (rel-L2 vs samples-seen), 1-GPU, evals every 4k samples."""


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
    latent_decoder = True
    n_latent = 16
    decode_hidden = 32
    loss_mse = False
    loss_relclamp = 4.0
    router_dim = 256
    router_depth = 2
    router_heads = 4
    router_patch = 2
    lambda_gate = 1e-4
    latent_factor = 4
    datasets = ("shallow_water", "com_ns", "incom_ns", "pdearena_ns", "cfdbench")
    n_per = 100000
    t_num = 20
    stream = True
    steps = 15000
    batch = 1
    lr = 1e-4
    lr_decay = True
    warmup = 1000
    instance_norm = True
    clipnorm = 1.0
    grad_accum = 4
    bf16 = True
    grad_checkpoint = False
    jit_compile = True
    swap_memory = False
    save_dir = "results/prose_150M_fluid5_6x"
    ckpt_every = 1000
    seed = 0
