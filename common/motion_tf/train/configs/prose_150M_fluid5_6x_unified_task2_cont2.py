"""Matched-budget CONTROL for the adv_bank arm: same warm-start (named task2_r1@56k), same 20k-step fresh
cosine, same data/batch — no bank. The A/B isolates the transport vocabulary's contribution on pdearena.
Launch:  bash _run_prose_stream_eu.sh motion_tf.train.configs.prose_150M_fluid5_6x_unified_task2_cont2
"""
from .prose_150M_fluid5_6x_unified_task2 import Cfg as _Base


class Cfg(_Base):
    steps = 20000
    warmup = 500
    init_ckpt = "/eu/results/migrated/task2r1_56000_named.npz"
    save_dir = "/eu/results/prose_150M_fluid5_6x_unified_task2_cont2"
    ckpt_every = 2000
