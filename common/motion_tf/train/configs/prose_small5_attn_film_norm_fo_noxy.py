"""SMALL architecture-iteration config: NO-COORDS ablation (drop absolute x,y from the encoder input).

Identical to prose_small5_attn_film_norm_fo (future-only / IC=last, instance_norm, MSE, 6 experts +
attn_router + route_film) EXCEPT `no_coords=True`: the absolute (x,y) coordinate channels are removed
from the encoder input. On our uniform PERIODIC grids the coord grid is a fixed spatial pattern that is
identical across every sample → carries no sample-discriminative info and BREAKS the translation
equivariance that periodic-conv experts would otherwise have. Dropping it makes the encoder fully
translation-equivariant, which matches the translation-invariant periodic physics and should improve
generalization (esp. the NS families, where pattern — not absolute position — is what matters).

Run on deneb-kr. Head-to-head vs prose_small5_attn_film_norm_fo (with-coords, 42.9% T=10 @3000)."""


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
    no_coords = True             # <- drop absolute (x,y) from the encoder input → translation-equivariant
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
    save_dir = "results/prose_small5_attn_film_norm_fo_noxy"
    ckpt_every = 500
    seed = 0
