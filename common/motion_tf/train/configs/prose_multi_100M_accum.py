"""PROSE-FD MULTI-FAMILY ~100M frontier run.
- split_basis: Fourier(cos/sin) and Legendre integrate SEPARATE W; a learned per-pixel/per-channel
  map decides where to trust each (no shared W).
- ~92M params: d_geom=832, expert_hidden=768, geo_layers=8 (elliptic ~22M, 15× the 5M run → the
  capacity the NS laggard needs; geo ~41M shared feature lift).
- grad_checkpoint=True (recompute the N_p panel rollout in backward) is the memory enabler at this
  width → jit OFF (recompute_grad is incompatible with XLA). batch=2/replica; raise after the
  step-1 [mem] print shows headroom. clipnorm=1.0 (NaN guard). bf16 = next lever if too slow.
"""


class Cfg:
    Nx = 128
    T_in = 10
    Nt = 11
    T_final = 1.0
    domain_L = 1.0
    N_p = 16                 # bf16 ALONE didn't fit N_p=64 at 92M (OOM 71GB temp); bf16+N_p32 fits
    #                          + keeps jit. N_p=32 is ample for an 11-frame trajectory (n_modes 16).
    n_modes = 8
    max_order = 8
    d_geom = 832
    geo_layers = 8
    expert_hidden = 768
    n_channels = 6
    desc_dim = 6
    experts = ("convective", "diffusive", "reaction", "elliptic", "shock")
    split_basis = True
    lambda_gate = 1e-3
    # data — reuses the 7GB prebuilt cache (3 families); add com_ns/incom_ns when downloaded
    datasets = ("shallow_water", "pdearena_ns", "diff_react")
    n_per = 100000           # ALL available (loaders cap at dataset size): SWE~1k, DR~1k, pdearena~8.7k
    t_num = 20
    # optim — while_loop rollout (O(1) graph) + jit + bf16 is the scaling recipe (Python-unroll
    # exploded the graph; recompute_grad+no-jit was unusably slow at 92M).
    steps = 3000              # frontier validation; extend after confirming NS improves at scale
    batch = 1                 # per-replica (global 8); raise per step-1 [mem] print
    lr = 1.5e-4
    lr_decay = True
    clipnorm = 0.5
    grad_accum = 8           # eff.batch = batch×#GPU×K = 1×8×8 = 64 (vs 8) — bigger batch, no extra mem
    bf16 = False
    stream = True           # data small now (3712 NS, 45GB 6-slot fits RAM); load-all from local copy feeds GPU full-speed. (volume-streaming starved the GPU; local-streaming TODO for 6-family scale)            # tf.data streaming: no cache copy, no full-RAM load, scales unbounded
               # Section-1 (given-coeff, PROSE-FD arena) trains in bf16 like PROSE;
    #                           halves memory so 92M·N_p64 fits 80GB. derivatives/state managed at
    #                           fp32-cast boundaries; ADA basis + loss fp32.
    grad_checkpoint = False   # while_loop (O(1) graph) + jit + bf16 fits without recompute
    jit_compile = True        # tf.while_loop IS XLA-compatible (unlike recompute_grad)
    # checkpointing
    save_dir = "results/prose_multi_100M_accum"
    ckpt_every = 500
    seed = 0
