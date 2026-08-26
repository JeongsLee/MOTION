"""Warm-restart continuation from the scratch ckpt (~305k), testing the hypothesis that the eff-batch-48
PLATEAU (~16%) is batch-driven: drop to eff-batch 8 (batch4 x 2 GPU) — the SAME regime the old explicit base
used to reach 7.45% — with a MODEST re-warmed LR (1e-4 peak, warmup 2000; NOT 6e-4, which diverged: 6e-4 on
eff-batch 24 was ~12x the sane per-sample step and blew up). Frequent eval (every 1000) to dial in conditions.
train_ivp warm-restart: init_from loads weights, fresh step counter + fresh cosine over cfg.steps.
"""
from .poseidon_pretrain_ivp_combo_scratch import Cfg as _Base


class Cfg(_Base):
    init_from = "/eu/results/poseidon_pretrain_ivp_combo_scratch/ckpt_305000.npz"
    batch = 4                      # eff-batch 8 on 2 GPU (== old explicit base regime)
    lr = 1e-4                      # old-base peak; warm-restart with warmup (6e-4 diverged)
    warmup = 2000
    steps = 60000
    ckpt_every = 5000
    eval_every = 1000              # frequent eval to dial in training conditions
    save_dir = "results/poseidon_pretrain_ivp_combo_cont305k_b8"
