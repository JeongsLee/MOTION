"""Synthetic pretrain→transfer sanity (Tier 0). Pretrain on heat·advection·reacdiff,
transfer to held-out ADR (conv+diff+react = a composition never seen jointly in pretrain).
Goal: confirm pretraining helps at low N in OUR architecture (foundation pipeline works)."""


class Cfg:
    Nx = 64
    Nt = 16
    T_final = 1.0
    N_p = 16
    n_modes = 8
    max_order = 8
    d_geom = 48
    geo_layers = 2
    experts = ("convective", "diffusive", "reaction")
    lambda_gate = 1e-3
    # pretrain
    pretrain_families = ("heat", "advection", "reacdiff")
    n_per_pretrain = 200
    pretrain_steps = 4000
    # transfer to held-out family
    heldout_family = "adr"
    N_list = (0, 1, 2, 4, 16, 64)   # incl. zero-shot (N=0) + truly-scarce low N
    n_pool = 128                 # held-out train pool
    n_test = 128
    finetune_steps = 4000        # fair: match scratch (was 2000)
    scratch_steps = 4000
    batch = 16
    lr = 1e-3                    # scratch lr
    finetune_lr = 2e-4           # lower lr for finetuning (avoid disrupting pretrained features)
    # FNO-scratch baseline
    fno_width = 32
    fno_modes = 12
    fno_layers = 4
    jit_compile = True
    ckpt = "/tmp/foundation_pretrain.npz"
    reuse_ckpt = True            # skip pretrain if ckpt exists (same pretrain config)
