"""ABLATION: field-conditioned gate vs the label/descriptor gate. IDENTICAL to prose_multi_5M_full
(same data, experts, N_p, fp32, eff-batch 16, lr) — the ONLY change is field_gate=True, so α_k is
routed from the pooled field features h_geom instead of the per-family-constant descriptor. Tests:
  (1) does α now VARY within a family (std>0) → regime-awareness (high-Re vs low-Re look different)?
  (2) does the NS↔diff_react conflict (shared diffusive expert) EASE vs the descriptor-gate 5M?
descriptor kept as a dropout-prior (gate_desc_dropout=0.5) so it stays usable yet field-primary
(→ transfers when the label is unknown). Compare against results/prose_multi_5M_full at matched steps."""


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
    field_gate = True            # <-- the ablation: α from field regime-stats, not the label
    gate_desc_dropout = 0.2      # descriptor kept as a weak prior, randomly hidden → field-primary
    lambda_gate = 1e-3
    datasets = ("shallow_water", "pdearena_ns", "diff_react")
    n_per = 100000
    t_num = 20
    stream = True
    steps = 8000
    batch = 2                    # eff = 2 x 8 = 16 (same as the descriptor-gate 5M baseline)
    lr = 3e-4
    lr_decay = True
    clipnorm = 1.0
    grad_accum = 1
    bf16 = False
    grad_checkpoint = False
    jit_compile = True
    save_dir = "results/prose_multi_5M_fieldgate_v2"  # v2 = regime-stats gate input (v1 pooled-h_geom collapsed)
    ckpt_every = 1000
    seed = 0
