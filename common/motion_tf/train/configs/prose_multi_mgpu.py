"""PROSE-FD MULTI-FAMILY mGPU run. Per-sample c_mask/desc; FamilyGate routes per family.
Start with the 2 fully-downloaded families (SWE + PDEArena NS); append comp/incomp NS +
diffusion-reaction to `datasets` as their downloads finish. batch raised to 2/replica."""


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
    lambda_gate = 1e-3
    # data — multi-family (build_multi). Add "com_ns","incom_ns","diff_react" when downloaded.
    datasets = ("shallow_water", "pdearena_ns", "diff_react")   # +"com_ns","incom_ns" when downloaded
    n_per = 900               # samples per family
    t_num = 20                # families resampled to common length
    # optim — SWE single-family was still dropping at 6000 steps (undertrained); train long + decay
    steps = 25000
    batch = 2                 # per-replica; eff 8×2=16. MEASURED: batch 3&4 OOM on 80GB
    #                           (per-sample ~21GB: b4 tried 70GB, b3 tried 64.5GB). b2≈42GB fits.
    lr = 3e-4
    lr_decay = True           # cosine to 5% of peak over `steps`
    grad_checkpoint = False
    jit_compile = True
    # checkpointing — eval (per-family test rel-L2) runs at each ckpt to track convergence
    save_dir = "results/prose_multi_mgpu"
    ckpt_every = 2000
    seed = 0
