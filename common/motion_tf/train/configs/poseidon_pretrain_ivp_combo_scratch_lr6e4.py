"""Hypothesis B: same combo scratch protocol (eff-batch 48, all-IC ic_reuse, 400k steps, 4x H100) but
LR scaled 1e-4 -> 6e-4. The old explicit base trained at eff-batch 8 / lr 1e-4; combo scratch runs eff-batch
48 (6x larger) at the SAME lr 1e-4, so by the linear-scaling rule its per-sample step is ~6x too small.
This run tests whether restoring the effective step size (6x -> ~6e-4) lets the single-shot combo learn the
6 fluid families to a level closer to the recursive explicit base. Fresh save_dir; runs in parallel with the
lr=1e-4 run for a clean A/B. Warmup kept at 2000 (cosine ramps to the 6e-4 peak); NaN-skip guards divergence.
"""
from .poseidon_pretrain_ivp_combo_scratch import Cfg as _Base


class Cfg(_Base):
    lr = 6e-4                       # 6x the 1e-4 baseline (linear scaling for eff-batch 8 -> 48)
    warmup = 2000
    save_dir = "results/poseidon_pretrain_ivp_combo_scratch_lr6e4"
