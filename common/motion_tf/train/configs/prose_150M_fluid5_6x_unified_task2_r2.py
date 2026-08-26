"""Phase-1 Task-2, SEGMENT 2: CONTINUE the warm-started 6-family run from r1's ckpts to 640k total steps
(= 160 epochs at PROSE's 4000-step/epoch convention). Same save_dir as r1 → the trainer auto-resumes from
r1's latest ckpt (ckpt_40000) and continues the step counter. Goal: push toward the BCAT/PROSE paper
targets with much more data exposure (640k×eff-batch-8 ≈ 5.1M views ≈ ~1/5.5 of PROSE's 40-epoch budget).
Needs a credit topup (~$850+ on ×2). Crash-safe (ckpt every 2000); resumable if interrupted."""
from .prose_150M_fluid5_6x_unified_task2_r1 import Cfg as _Base


class Cfg(_Base):
    steps = 640000                 # 160 epochs x 4000 steps; resumes from r1's ckpt_40000 in the shared save_dir
    # save_dir inherited from r1 (results/prose_150M_fluid5_6x_unified_task2_r1) → auto-resume continuation
