"""Phase-1 INTEGRATED (Task-1 efficacy + Task-2 full-train) — ADA.

ADA = best Phase-1 recipe: 152M unified-lup MASKING (geo_layers=13, single UnifiedExpert + transformer
encoder, NO routing → data-only), 6 families incl pdearena_uncond.

Protocol (3 models ADA/PROSE-FD/BCAT, all from scratch, eff-batch 32 = BCAT/PROSE match, 1M sample-views):
  - eff-batch 32 = batch(1) × grad_accum(32) on 1 GPU.  steps = 31,250 → 1,000,000 views.
  - x-axis of the curve = #views seen = 32 × step.  Early part = data-efficacy, endpoint = full-train.
  - eval_every=250 steps (=8,000 views) → ~125-point dense training-trajectory curve, logged (the runner
    tees stdout to a /eu file so the curve survives vessl log truncation).
  - ckpt_milestones = LOG-SCALE step list (= views 4k,8k,…,1M) → bounded disk + post-hoc re-eval backup.
  - METRIC: the inline [EVAL] now logs the PROSE/BCAT-exact metric (per-frame arithmetic-mean physical
    rel-L2 norm, class-average over 6 families) — IDENTICAL to eval_prose_exact and to PROSE/BCAT native
    eval, so the logged curve is directly comparable across all 3 models.
  - data-only (lambda_gate=0), test = last 10% per family (same trajectories as PROSE/BCAT).
"""
from .prose_150M_fluid5_6x_unified_155m_mask import Cfg as _Base


class Cfg(_Base):
    datasets = ("shallow_water", "com_ns", "incom_ns", "pdearena_ns", "cfdbench", "pdearena_uncond")
    batch = 1
    grad_accum = 16                     # eff-batch = batch(1) × nrep(2 GPU) × grad_accum(16) = 32 (BCAT/PROSE match)
    steps = 31250                       # 31,250 × 32 = 1,000,000 views
    eval_every = 250                    # 250 × 32 = 8,000 views per eval point (dense curve)
    ckpt_milestones = [125, 250, 500, 1000, 2000, 4000, 8000, 16000, 31250]  # views 4k,8k,…,1M (log-scale)
    ckpt_every = 0                      # disable fixed-interval ckpt; use the log-scale milestones above
    save_dir = "results/phase1_int_ada_152m_6fam"
