"""All-experts variant of prose_150M_relc_dec: enables EVERY expert we built — adds `wave` (cross-channel
gradient coupling, previously unused) and `forcing` (input-inferred external body-force, NEW) on top of the
6 in the base config. Ablation: does the full expert set (esp. forcing → forced families NS-cond/incom)
beat the 6-expert baseline (60k = 11.18% overall)? Trained 15000 steps (60k samples) for a matched compare.
Fresh save_dir (8-expert arch ≠ baseline ckpt). lr=1e-4/warmup=1000 (stable on dt=1)."""


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
    experts = ("convective", "diffusive", "reaction", "elliptic", "shock", "helmholtz", "wave", "forcing")
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
    datasets = ("shallow_water", "pdearena_ns", "diff_react", "com_ns", "incom_ns")
    n_per = 100000
    t_num = 20
    stream = True
    steps = 15000               # 60k samples — matched to the 6-expert baseline (results/prose_150M_relc_dec_dt1)
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
    save_dir = "results/prose_150M_relc_dec_allx"
    ckpt_every = 1000
    seed = 0
