"""CLEAN RERUN of the adv_bank from-scratch arm (run 1 died at ~118k on disk quota; a resume breaks the
data-efficiency protocol — optimizer restart => the curve is no longer comparable). Changes vs run 1:
(a) fresh save_dir, (b) RELATIVE save_dir so the runner's LOCAL/REMOTE staging works as designed
(train.log lands in the run dir, not /eu/eu/...), (c) ckpt_every=10000 (the inline-eval curve lives in
train.log; sparse ckpts are just backup -> 5x less disk, no quota risk). Training math identical to run 1.
Launch:  bash _run_prose_stream_eu.sh motion_tf.train.configs.prose_150M_fluid5_6x_unified_task2_advb2
"""
from .prose_150M_fluid5_6x_unified_task2 import Cfg as _Base


class Cfg(_Base):
    adv_bank = True
    adv_bank_M = 4
    adv_bank_K = 4
    save_dir = "results/prose_150M_fluid5_6x_unified_task2_advb2"
    ckpt_every = 10000
