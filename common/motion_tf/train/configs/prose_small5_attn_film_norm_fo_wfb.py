"""SMALL architecture-iteration config: cross-expert W-FEEDBACK ablation.

Identical to prose_small5_attn_film_norm_fo (future-only / IC=last, instance_norm, MSE loss, 6 experts
+ attn_router + route_film) EXCEPT `w_feedback=True`: at each rollout panel every expert additionally
sees an embedding of ALL experts' PREVIOUS-panel contributions {W_k^{i-1}}, not just the summed/
integrated state z. Operator-splitting WITH coupling (Chorin projection: the elliptic/pressure expert
reacts to the convective+diffusive predicted tendency → the structurally-correct form for incompressible
NS, our worst family). zero-init conv → identity at start; learns coupling only if it helps.

Run on deneb-kr (cluster-e25pumfmj51e). Compare head-to-head vs prose_small5_attn_film_norm_fo."""


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
    w_feedback = True            # <- cross-expert W-feedback: experts see all experts' previous-panel W
    loss_mse = True
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
    save_dir = "results/prose_small5_attn_film_norm_fo_wfb_v2"
    ckpt_every = 500
    seed = 0
