"""SCRATCH (cold) BCAT-budget run — independent of the buggy r1/r2 weights.

Why a fresh run: r1/r2 trained pdearena_uncond WITHOUT the temporal mask (the deployed prose.py SPEC was
missing valid_t=14 → _build_tmasks returned all-ones → the clip-repeat padded frames 14-19 polluted the
loss with a wrong "freeze after frame 13" signal). This run starts from scratch with valid_t=14 now in
the SPEC, so uncond is supervised only on its 4 real future frames (PROSE's data_mask_len behaviour).

BCAT budget match (arXiv 2501.18972, src/configs/main.yaml: batch_size=32, 40 epochs x 4000 steps):
  eff-batch 32 (2 GPU x batch 1 x grad_accum 16) x 160k steps = 5.12M sample-views = BCAT's exact budget.
Single cosine over 160k from cold (warmup 4000 -> peak lr -> alpha=0.05 floor) — NO warm-restart, so no
mid-run LR re-injection. ~4.5 days on 2x H100 deneb-kr (~13.3 views/s measured), ~$510.
"""
from .prose_150M_fluid5_6x_unified_task2 import Cfg as _Base


class Cfg(_Base):
    grad_accum = 16                # 2 GPU x batch 1 x 16 = eff-batch 32 (= BCAT batch_size=32)
    steps = 160000                 # 40 epochs x 4000 = BCAT; 160k x 32 = 5.12M views
    warmup = 4000                  # single cosine over 160k (cold start -> no schedule restart)
    save_dir = "results/prose_150M_fluid5_6x_unified_bcat"
    ckpt_every = 2000              # crash-safe / resumable
    # NO init_ckpt: cold start, independent of the buggy-mask r1/r2 weights.
