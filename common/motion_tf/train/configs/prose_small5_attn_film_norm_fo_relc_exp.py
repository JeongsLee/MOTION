"""relc base + EXPERT FIXES (self-advection + wave expert). = the winning relc config (clamped rel-L2
loss, future-only, instance_norm, attn_router + route_film) with two operator fixes:
  (1) conv_self_advect=True — ConvectiveExpert now does NONLINEAR SELF-ADVECTION u·∇z from the state
      velocity (channels 0,1), reviving the transport term that was DEAD (coeffs ≡ 0 → 0) for every
      fluid family (pdearena/com/incom NS). [SWE is height-only → still 0 there, correctly.]
  (2) + "wave" expert — cross-channel ∇ coupling (gravity/acoustic restoring: g·h·∇h, −∇p) that no
      single-channel operator could form; targets com_ns (acoustic) and the fluid pressure coupling.
7 experts total. Run on deneb-kr. Compare vs relc (44.8% @1500)."""


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
    conv_self_advect = True      # (1) nonlinear self-advection u·∇z from state velocity (ch 0,1)
    split_basis = False
    field_gate = False
    input_router = False
    attn_router = True
    route_context = False
    route_film = True
    loss_mse = False             # relc base: clamped rel-L2 loss
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
    save_dir = "results/prose_small5_attn_film_norm_fo_relc_exp"
    ckpt_every = 500
    seed = 0
