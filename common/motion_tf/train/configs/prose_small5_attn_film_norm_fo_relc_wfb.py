"""relc base + cross-expert W-FEEDBACK. = fo baseline with clamped normalized rel-L2 loss
(loss_mse=False, loss_relclamp=4.0) AND w_feedback=True (each panel's experts see all experts'
previous-panel W, stop_gradient+tanh stabilized — Chorin projection for NS). Re-tests the feedback
idea on the better loss (it took an early hit under MSE). Run on deneb-kr. Compare vs relc-alone."""


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
    w_feedback = True            # cross-expert W-feedback (stopgrad+tanh stabilized)
    loss_mse = False             # rel-L2 loss ...
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
    save_dir = "results/prose_small5_attn_film_norm_fo_relc_wfb"
    ckpt_every = 500
    seed = 0
