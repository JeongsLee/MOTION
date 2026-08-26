"""Gentle low-LR continuation from the scratch ckpt (~305k) — do NOT shake the well-trained weights.
The previous warm-restart re-warmed the LR to a peak (warmup ramp) and that perturbation spiked the eval at
the start. Here: warmup=0 so training STARTS directly at a LOW lr (2e-5, ~ the main run's decayed LR at 305k
=~1.8e-5), i.e. no re-warm shock, just continued descent at eff-batch 8 (batch4 x 2 GPU) for 200k more steps.
Frequent eval (every 1000). train_ivp loads init_from with a fresh step counter; with warmup=0 the fresh
cosine begins at `lr` (no ramp-up).
"""
from .poseidon_pretrain_ivp_combo_scratch import Cfg as _Base


class Cfg(_Base):
    init_from = "/eu/results/poseidon_pretrain_ivp_combo_scratch/ckpt_305000.npz"
    batch = 4                      # eff-batch 8 on 2 GPU
    lr = 2e-5                      # LOW start (~= 305k state LR) -> weights not shaken
    warmup = 0                     # NO re-warm ramp (the peak ramp is what perturbed the good weights)
    lr_decay = True
    steps = 200000
    ckpt_every = 5000
    eval_every = 1000
    save_dir = "results/poseidon_pretrain_ivp_combo_cont305k_lowlr"
