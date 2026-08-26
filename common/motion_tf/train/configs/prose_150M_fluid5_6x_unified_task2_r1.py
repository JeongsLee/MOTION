"""Phase-1 Task-2, SEGMENT 1: warm-start from Task-1 unified weights (ckpt_15000) and train the full
6-family set for 10 more epochs (= 10 x 4000 = 40000 steps) on 2 GPUs. Crash-safe (ckpt every 2000) and
RESUMABLE — a later segment just points its save_dir here / or continues from this run's ckpts.

Why warm-start: Task-1 already converged the 5 fluid families to overall 10.43% phys rel-L2 (60k samples);
starting Task-2 from those weights (instead of scratch) means the only new thing to learn is the 6th family
(pdearena_uncond, temporally masked) + the 2x-larger effective batch — far cheaper than a cold 40-epoch run.
"""
from .prose_150M_fluid5_6x_unified_task2 import Cfg as _Base


class Cfg(_Base):
    init_ckpt = "/code-vol/results/prose_150M_fluid5_6x_unified/ckpt_15000.npz"   # Task-1 weights (warm start)
    steps = 40000                  # 10 epochs x 4000 steps/epoch (PROSE epoch convention)
    warmup = 500                   # short ramp — weights are already trained, don't disrupt them
    ckpt_every = 2000
    save_dir = "results/prose_150M_fluid5_6x_unified_task2_r1"
