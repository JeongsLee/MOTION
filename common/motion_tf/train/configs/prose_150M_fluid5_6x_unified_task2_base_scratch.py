"""FROM-SCRATCH control for the adv_bank arm: the exact task2 joint protocol, no bank, EU paths.
Launch:  bash _run_prose_stream_eu.sh motion_tf.train.configs.prose_150M_fluid5_6x_unified_task2_base_scratch
"""
from .prose_150M_fluid5_6x_unified_task2 import Cfg as _Base


class Cfg(_Base):
    save_dir = "/eu/results/prose_150M_fluid5_6x_unified_task2_base_scratch"
    ckpt_every = 2000
