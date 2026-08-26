"""PHASE-2 IVP pretraining — COMBO from SCRATCH, budget-matched to Poseidon-B (2026-06-29).

The 120k combo base was ~100x under Poseidon-B's sample budget (eff-batch 8 x 120k = 0.96M samples vs
Poseidon ~199M pairs) — THAT, not the arch, is the NS weakness. This run fixes it:

  * all2all via IC-REUSE (ic_reuse=True): the reader yields a FULL trajectory; we reuse that one read for
    ALL ICs 0..19 (shuffled) as consecutive optimizer updates. This covers ALL 210 (i,j) pairs per
    trajectory AND cuts reader load ~20x. (Naive "one random IC per read" re-read each trajectory 20x; the
    single serial HDF5 reader — ~18 traj/s, un-threadable since HDF5 is not thread-safe — then capped
    throughput at ~18 samples/s on ANY GPU count = ~12 days. IC-reuse makes it compute-bound -> 4-GPU scales.)
    max_ic_frac=0.95 -> ICs 0..19 (IC=20 excluded: no future frame).
  * 12 epochs over the 6-family ~80k trajectories (~= Poseidon-B's 77,840 traj):
      80k traj x 20 IC x 12 ep / eff-batch 48 = 400k steps (optimizer updates).
    Each (i,j) pair seen 12x (Poseidon ~12x) -> 201.6M pair-gradients ~= Poseidon's 199M (parity).
  * eff-batch 48 = batch 12/replica x 4 H100 (ch100x4), grad_accum=1 (fast jitted dist_step; NO accum —
    grad-accum on multi-GPU is un-jittable due to the outer-jit-on-strategy.run device bug). batch=12
    peaked at 58GB/80GB (safe). reader_threads=1 (single serial reader; fine now that IC-reuse cut its load).
    Compute-bound -> ~0.6s/step -> ~2.8 days.
  * ckpt_every 5000 (80 ckpts ~36G, disk-safe in 139G) but eval_every 2500 (decoupled -> watch the NS curve).

Run on resourcespec-ch100x4 (4x H100, cluster-capella = same cluster as /eu). Scratch (no init_from).
"""
from .poseidon_pretrain_ivp_combo import Cfg as _Combo


class Cfg(_Combo):
    # ---- all-IC all2all via read-amortizing IC-REUSE (one read batch -> all ICs) ----
    all2all = True
    ic_reuse = True                # reuse each read batch for all ICs 0..max_ic (shuffled) -> ~20x fewer reads
    max_ic_frac = 0.95             # max_ic = round(0.95 * 20) = 19 -> ICs 0..19 (IC=20 has no future -> excluded)
    reader_threads = 1             # single serial reader (HDF5 not thread-safe); IC-reuse removes the I/O bottleneck

    # ---- budget-matched schedule (eff-batch 48 = batch 12 x 4 GPU, NO grad_accum -> fast jitted path) ----
    batch = 12
    grad_accum = 1
    steps = 400000                 # 12 epochs x 80k traj x 20 IC / 48
    lr = 1e-4
    lr_decay = True
    warmup = 2000

    # ---- checkpoint/eval cadence (disk-safe: 80 ckpts ~36G in 139G free) ----
    ckpt_every = 5000
    eval_every = 2500

    seed = 1
    save_dir = "results/poseidon_pretrain_ivp_combo_scratch"
