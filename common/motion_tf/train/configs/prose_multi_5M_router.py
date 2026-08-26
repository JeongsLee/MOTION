"""ABLATION: input-ROUTER (per-pixel per-expert sigmoid dispatch) vs output-gating. IDENTICAL to
prose_multi_5M_full except input_router=True: instead of a global scalar α_k weighting expert
OUTPUTS, a per-pixel r_k(x)=σ(Conv(h_geom)) decides FROM THE INPUT (once, up front) how much of each
expert's contribution flows in at each location. Tests:
  (1) does per-pixel routing tie expert learning-signal to relevance (vs α starving low-gated experts)?
  (2) does it beat output-gating / field-gate at matched steps?
  (3) interpretability: per-expert spatial routing maps (where each operator governs).
lambda_gate=0 — no sparsity pressure on the route (would starve experts); add load-balance if one
expert dominates. independent sigmoid (not softmax) so experts can be jointly active per pixel."""


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
    experts = ("convective", "diffusive", "reaction", "elliptic", "shock")
    split_basis = False
    field_gate = False           # router replaces the gate entirely
    input_router = True          # <-- the ablation: per-pixel per-expert sigmoid input dispatch
    lambda_gate = 1e-4           # light sparsity on pooled route → discourages all-experts-saturate
    datasets = ("shallow_water", "pdearena_ns", "diff_react")
    n_per = 100000
    t_num = 20
    stream = True
    steps = 8000
    batch = 2
    lr = 2e-4                    # slightly lower — router run diverged @700 at 3e-4
    lr_decay = True
    clipnorm = 1.0
    grad_accum = 1
    bf16 = False
    grad_checkpoint = False
    jit_compile = True
    save_dir = "results/prose_multi_5M_router_v2"  # v2 = neg-bias router init + light reg + lr2e-4 (v1 NaN@700)
    ckpt_every = 1000
    seed = 0
