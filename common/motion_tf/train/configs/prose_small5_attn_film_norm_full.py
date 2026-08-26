"""SMALL architecture-iteration config: TRANSFORMER router ablation. Identical to prose_small5_hz
(6 experts incl. helmholtz, 5 families x 900) EXCEPT the router: a small transformer (AttnRouter)
replaces the 1x1-conv sigmoid router. Tests the thesis (DESIGN.md §router): the data->weights mapping
needs a verified architecture (transformers — PROSE-FD/Poseidon), so the ROUTER is transformer-based
while the physics-aware EXPERTS are unchanged. softmax-over-experts bounds the routed sum (cures the
sigmoid-router divergence + elliptic collapse).

Run on deneb-kr (cluster-e25pumfmj51e — co-located with the Seoul data) via _run_prose_small.sh."""


class Cfg:
    Nx = 128
    T_in = 10
    Nt = 20
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
    input_router = False         # <- off; the attn router below is the input-dispatch router instead
    attn_router = True           # TRANSFORMER router (AttnRouter): patch-attn -> per-expert softmax
    route_context = False
    route_film = True            # FiLM(γ,β) route-context
    ic_first = True              # FULL-TRAJECTORY supervision: IC=first frame, target=u[0:Nt]
    loss_mse = True              # PROSE training loss = MSE in normalized space (rel-L2 only for EVAL)
    router_dim = 192             # router token width
    router_depth = 2             # transformer blocks
    router_heads = 4
    router_patch = 8             # 128/8 = 16 -> 16x16 = 256 tokens (LF=1, full-res h_geom)
    lambda_gate = 1e-4
    datasets = ("shallow_water", "pdearena_ns", "diff_react", "com_ns", "incom_ns")
    n_per = 900
    t_num = 20
    stream = True
    steps = 3000
    batch = 1                    # eff = 1 x 1 x accum(4) = 4 (6-expert N_p64 needs batch=1)
    lr = 2e-4
    lr_decay = True
    warmup = 300
    instance_norm = True
    clipnorm = 1.0
    grad_accum = 4
    bf16 = False
    grad_checkpoint = False
    jit_compile = True
    save_dir = "results/prose_small5_attn_film_norm_full"
    ckpt_every = 500
    seed = 0
