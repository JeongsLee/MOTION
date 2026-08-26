"""PHASE-2 IVP unified — CONTINUATION on capella EU 2xH100, warm-started from the 120k checkpoint, with
the training budget MATCHED to Poseidon-B's pretraining (the fair same-budget comparison).

Why a warm-RESTART (init_from) and not an in-place resume: the 120k run's cosine LR had decayed to its
floor (~5e-6) and the eval flat-lined (116k=15.31% -> 120k=15.47%). Resuming in place would replay that
floor LR -> no further learning. So we load the 120k weights as an INIT and run a FRESH cosine over the
remaining budget at a re-warmed peak LR.

Budget math (match Poseidon-B = 39 epochs over the full 77,840 pretraining trajectories):
  - global_batch = batch(4) x 2 GPU = 8  ->  steps/epoch = 77840/8 = 9730.
  - 120k done @ global_batch 4 = 480,000 trajectory-presentations = 6.17 epochs already.
  - target 39 epochs total -> 32.83 NEW epochs -> 32.83 * 9730 ~= 319,500 NEW steps.  (steps below.)
Each step samples ONE IC frame per trajectory and supervises the masked future frames (our all2all),
so "epoch" here = one pass over the trajectory set. Poseidon-B: 39 ep, eff-batch 320, cosine + 1ep warmup.

Data on EU: /eu/data/poseidon/_assembled (488GB, full 6-operator set, capella-local cephfs -> fast).
Run with _run_ivp_eu.sh (mounts the EU cluster volume, sets POSE_ASSEMBLED, adds the mGPU thread caps)."""
from .poseidon_pretrain_ivp_unified import Cfg as _Base


class Cfg(_Base):
    # ---- warm-restart source (absolute path inside the EU container; run script guarantees it exists) ----
    init_from = "/eu/results/poseidon_pretrain_ivp_unified/ckpt_120000.npz"

    save_dir = "results/poseidon_pretrain_ivp_unified_eu2"  # NEW dir -> init_from re-fires, no stale ckpts

    # ---- budget: 2 GPU (global_batch = 4*2 = 8), fresh cosine to match Poseidon-B 39-epoch total ----
    batch = 4                      # per-GPU; MirroredStrategy makes global_batch = 8 on 2 GPUs
    steps = 320000                 # ~32.8 new epochs -> 39 total (with the 6.17 already done)
    # LR FIX (v2): the original 1g run RAN ITS COSINE TO THE FLOOR (~5e-6) and converged. Re-warming to the
    # original PEAK 1e-4 kicked the converged weights OUT of the basin -> eval REGRESSED 15.47% -> 17.73%
    # @44k (NS-Sines 49->56%). So warm-restart at a MODERATE peak: ~4x the late-stage LR, not 20x. Gentle
    # enough that the converged CE families hold while NS keeps learning; cosine -> floor over the budget.
    lr = 2e-5
    lr_decay = True
    warmup = 300                   # short ramp; we are continuing converged weights, not training cold
    ckpt_every = 2000
    seed = 2                       # new data order vs the 120k run (was seed=1)
