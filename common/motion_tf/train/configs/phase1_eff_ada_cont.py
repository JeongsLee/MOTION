"""Phase-1 ADA CONTINUATION — warm-start from the finished 1M-view ckpt (ckpt_31250) and train 10M MORE
views to measure how much further OUR model alone improves. Standard (previous) all2all/stream scheme of
train_prose_mgpu (no det-ic). lr lowered to 1/5 of the original peak (1e-4 -> 2e-5) with a gentle warmup
to avoid the warm-restart spike. Frequent early eval verifies the warm-start landed (~4.43%); if it spiked,
relaunch with a lower CONT_LR.
"""
import os
from .phase1_eff_ada import Cfg as _Base


class Cfg(_Base):
    init_ckpt = os.environ.get("CONT_INIT", "/eu/results/phase1_int_ada_152m_6fam/ckpt_31250.npz")
    lr = float(os.environ.get("CONT_LR", "2e-5"))            # 1e-4 / 5
    warmup = int(os.environ.get("CONT_WARMUP", "300"))       # gentle ramp (avoid re-warm bump from a good min)
    steps = int(os.environ.get("CONT_STEPS", "312500"))      # 10M more views (eff-batch 32)
    eval_every = int(os.environ.get("CONT_EVAL_EVERY", "250"))  # first eval @250 = warm-start check
    ckpt_milestones = [250, 1000, 4000, 16000, 64000, 156250, 312500]  # log-scale (bounded disk)
    ckpt_every = 0
    save_dir = "results/" + os.environ.get("CONT_SAVE", "phase1_int_ada_152m_6fam_cont10M")
