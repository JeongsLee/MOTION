"""PHASE-1 TASK 2 (best-vs-best, train to convergence): RESUME from the Task-1 fluid5_6x checkpoint
(same save_dir → runner loads cks[-1] = ckpt_15000) and continue to a SOTA-level budget. Warm-start from
the 60k-trained weights → much faster convergence than from scratch. data-only (descriptor OFF — same info
as SOTA BCAT). Uses the EXPANDED 5-fluid data (incom 274 + com Turb) once those finish. Intended for 4-GPU
(MirroredStrategy, batch=2/replica) on deneb-kr when an H100x4 frees up; falls back to x1/x2."""


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
    attn_router = True            # input-based routing → descriptor effectively OFF (data-only, BCAT-comparable)
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
    steps = 200000               # ~800k–3.2M samples (4·batch·steps); SOTA-level budget. Resumes ckpt_15000.
    batch = 2                    # per-replica; on 4 GPU → global 8 × accum4 = 32 effective
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
    save_dir = "results/prose_150M_fluid5_6x"   # SAME as Task-1 → resume from its ckpt_15000
    ckpt_every = 2000
    seed = 0
