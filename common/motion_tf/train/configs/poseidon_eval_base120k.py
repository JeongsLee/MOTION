"""EVAL-ONLY: re-measure the 120k combo IVP base with the CURRENT SOL-SLOT inline metric, so it is
apples-to-apples vs the scratch run (same test set, same de-diluted solution-slot metric). Inherits the
combo arch; save_dir is pre-seeded (in the launch cmd) with the base ckpt_120000.npz so train_ivp's resume
loads it; steps=120000 -> the training loop is empty (start=120001) -> only log_eval("final") runs.
"""
from .poseidon_pretrain_ivp_combo import Cfg as _Combo


class Cfg(_Combo):
    steps = 120000                 # == existing ckpt step -> resume sets start=120001 -> loop skips -> eval only
    ckpt_every = 0
    eval_every = 0
    reader_threads = 1
    ic_reuse = False
    save_dir = "results/_eval_base120k"
