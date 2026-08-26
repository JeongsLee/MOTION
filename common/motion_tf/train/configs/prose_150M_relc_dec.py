"""150M FULL-DATA scale-up of the WINNING small recipe (relc+dec). Combines the proven small-model
architecture/training with the 150M-scale dims + latent-grid that make fp32 fit:

  WINNER recipe (from the small sweep, best = relc+dec @42.3%):
    • clamped normalized rel-L2 loss (loss_mse=False, loss_relclamp=4.0)
    • instance_norm=True, future-only / IC=last (Nt=11 = 1 IC + 10 future)
    • latent_decoder=True (n_latent=16, NTO-ADA HiLatent: latent dynamics + no-bias MLP decode + residual)
    • 6 physics experts + attn_router + route_film
  150M SCALE dims (from prose_150M_film, fp32 fits via latent_factor=4 → experts at 32²):
    • d_geom=1024, geo_layers=9, expert_hidden=960, latent_factor=4
    • attn_router router_dim=256, router_patch=2 (32² latent grid → 16×16=256 tokens)
  FULL data: n_per=100000 (all 5 families), streamed from the volume (no copy).

Run on deneb-kr (cluster-e25pumfmj51e) via _run_prose_stream.sh. ckpt_every=1000 → early evals +
resume-safe if credits run low. Verify the param print is ~150M and step-1 fits / is finite."""


class Cfg:
    Nx = 128
    T_in = 10
    Nt = 11
    T_final = 1.0
    domain_L = 1.0
    N_p = 64
    n_modes = 32
    max_order = 32
    d_geom = 1024
    geo_layers = 9
    expert_hidden = 960
    n_channels = 6
    desc_dim = 6
    experts = ("convective", "diffusive", "reaction", "elliptic", "shock", "helmholtz")
    split_basis = False
    field_gate = False
    input_router = False
    attn_router = True
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
    router_patch = 2             # latent grid 32² (LF=4) → patch2 → 16×16=256 tokens
    lambda_gate = 1e-4
    latent_factor = 4            # experts at 32² → ~16× less rollout activation → 150M fp32 fits
    datasets = ("shallow_water", "pdearena_ns", "diff_react", "com_ns", "incom_ns")
    n_per = 100000               # FULL data
    t_num = 20
    stream = True
    steps = 40000               # extended from 15000 (60k samples) → 160k samples to probe MAX performance;
    batch = 1                   #   resumes from ckpt_15000 (60k results preserved in logs/ckpts/memory)
    lr = 1e-4                    # 2e-4 collapsed @step350 on dt=1 data (W→0 dead-expert); halved for stability
    lr_decay = True
    warmup = 1000                # longer ramp (was 500) — smoother dt=1 dynamics need gentler early lr
    instance_norm = True
    clipnorm = 1.0
    grad_accum = 4
    bf16 = True                  # fp32 OOMs (46.8GB single alloc); bf16 fits ~23GB — cast gaps in route_film/latent fixed inline
    grad_checkpoint = False      # tf.recompute_grad drops grads here (none=37) → OFF; bf16 alone fits
    jit_compile = True
    swap_memory = False
    save_dir = "results/prose_150M_relc_dec_dt1"   # fresh dir for the dt=1 run (avoid stale-arch ckpt)
    ckpt_every = 1000
    seed = 0
