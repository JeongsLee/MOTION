"""Warm-restart continuation test from the lr1e-4 scratch ckpt (~step 305k), on 2x H100, LR re-warmed to
6e-4, to see whether a higher effective step size accelerates convergence past the ~16% plateau. train_ivp
warm-restart: init_from loads weights with a FRESH step counter + FRESH cosine over cfg.steps. eff-batch 24
(batch12 x 2 GPU; batch24 would OOM). ic_reuse/all-IC/ckpt/eval cadence inherited. Fresh save_dir.
"""
from .poseidon_pretrain_ivp_combo_scratch import Cfg as _Base


class Cfg(_Base):
    init_from = "/eu/results/poseidon_pretrain_ivp_combo_scratch/ckpt_305000.npz"
    lr = 6e-4
    warmup = 1000
    steps = 60000                  # continuation test horizon (fresh cosine at 6e-4)
    save_dir = "results/poseidon_pretrain_ivp_combo_cont305k_lr6e4"
