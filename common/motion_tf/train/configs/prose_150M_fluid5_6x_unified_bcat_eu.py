"""PHASE-1 Task-2 (BCAT-budget unified) — CONTINUATION on capella EU 2xH100, warm-started from the deneb
run's ckpt_54000, finishing the remaining budget. Migrated off deneb-kr ahead of the Korea maintenance.

The deneb run (prose_150M_fluid5_6x_unified_bcat) is a single cosine over 160k steps (warmup 4000, peak
lr 1e-4, alpha=0.05). It reached ~step 54k. train_prose_mgpu resets the optimizer on load, so a plain
resume would REPLAY the warmup and re-inject the peak LR mid-run (its own comment notes "resume restarts
the schedule from step 0"). To avoid that spike we warm-RESTART (init_ckpt, fresh save_dir) with a fresh
cosine over ONLY the remaining steps, whose PEAK is set to the LR the original cosine was at by step 54k,
so the LR continues smoothly downward to the floor instead of jumping back up.

LR at original step 54000:  p=(54000-4000)/156000=0.3205 -> decay=0.05+0.95*0.5*(1+cos(pi*p))=0.7781
  -> lr(54k)=1e-4*0.7781 = 7.78e-5  (used as the continuation peak; cosine -> 0.05*peak floor).
Remaining budget: 160000 - 54000 = 106000 steps (eff-batch 32 unchanged -> ~3.4M more sample-views).
Data streams from the capella-local mirror /eu/data/prose/prebuilt (cephfs -> fast, no cross-region FUSE).
Run with _run_prose_stream_eu.sh."""
from .prose_150M_fluid5_6x_unified_bcat import Cfg as _Base


class Cfg(_Base):
    init_ckpt = "/eu/results/prose_150M_fluid5_6x_unified_bcat/ckpt_54000.npz"  # staged from /code-vol if absent
    save_dir = "results/prose_150M_fluid5_6x_unified_bcat_eu"

    steps = 106000        # remaining of the 160k plan (54k already done on deneb)
    lr = 7.78e-5          # = original cosine's LR at step 54k -> smooth continuation, no peak re-injection
    warmup = 200          # brief ramp to settle the freshly-reset optimizer moments (not a full re-warm)
    # grad_accum=16, batch=1, lr_decay=True, ckpt_every=2000 all inherited (eff-batch 32 preserved)
