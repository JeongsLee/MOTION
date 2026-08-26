"""PHASE-2 pretraining config — IVP mode (single IC snapshot, all2all, lead-time = output-frame index).
Pretrain OUR PhysicsOperatorMixture on Poseidon's EXACT pretraining set (CE-RP/KH/CRP/Gauss + NS-Sines/Gauss),
data-only (descriptor OFF), to then transfer to the 15 unseen downstream tasks. SEPARATE from all Phase-1
configs/entries — run with train_ivp.py so the Phase-1 pipeline is untouched and both keep running.

IVP design: T_in=1 (single snapshot), Nt=21 + T_final=1 → the model's fixed output grid (spacing 1/20)
aligns with PDEgym's 21 uniform snapshots, so output-frame index k = lead time. all2all is done in the train
loop: sample an IC frame i per trajectory, supervise the model's output frames against data frames i..20
(later frames masked). Autonomous PDEs → time-translation invariant, so predicting from any frame is valid;
forcing/BC/geometry enter as INPUT CHANNELS (slots 6/7) for downstream tasks (absent in CE/NS pretraining)."""


class Cfg:
    Nx = 128
    T_in = 1                       # single IC snapshot (IVP)
    Nt = 21                        # PDEgym snapshot count → output frame index = lead time
    T_final = 1.0                  # unit horizon (matches PDEgym T=1) → frame spacing 1/20
    domain_L = 1.0
    N_p = 64
    n_modes = 32
    max_order = 32
    d_geom = 1024
    geo_layers = 9
    expert_hidden = 960
    n_channels = 8                 # Phase-2 8-slot layout (vx,vy,tracer,density,pressure,energy,geom*,forcing*)
    desc_dim = 8
    experts = ("convective", "diffusive", "reaction", "elliptic", "shock", "helmholtz")
    split_basis = False
    field_gate = False
    input_router = False
    attn_router = True             # input-based routing → descriptor OFF (data-only, Poseidon-comparable)
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

    # ---- IVP / all2all (read by train_ivp.py) ----
    ivp_mode = True
    all2all = True                 # sample IC frame i ∈ [0, max_ic], supervise frames i..20 (masked)
    max_ic_frac = 0.5              # sample IC frame up to 50% into the trajectory (≥11 supervised frames)

    # ---- Poseidon data ----
    datasets = ("CE-RP", "CE-KH", "CE-CRP", "CE-Gauss", "NS-Sines", "NS-Gauss")
    stream = True
    test_per_fam = 64

    steps = 120000
    batch = 4
    lr = 1e-4
    lr_decay = True
    warmup = 1000
    instance_norm = True           # (poseidon loader uses per-dataset global stats; flag kept for parity)
    clipnorm = 1.0
    grad_accum = 1
    bf16 = True
    grad_checkpoint = False
    jit_compile = True
    swap_memory = False
    save_dir = "results/poseidon_pretrain_ivp"
    ckpt_every = 2000
    seed = 1               # changed from 0 after a step-44300 NaN (rare bad-batch) → new data order on resume
