"""FROM-SCRATCH adv_bank arm (pdearena transport diagnosis): identical to task2 joint training but with the
loose transport vocabulary (adv_bank) on from step 0. A/B partner: ..._task2_base_scratch (same seed-free
protocol, no bank). Compare the full per-family eval HISTORY — the pdearena_ns/uncond curves are the
diagnosis-matched question; the other families check the bank does no harm.
Launch:  bash _run_prose_stream_eu.sh motion_tf.train.configs.prose_150M_fluid5_6x_unified_task2_advb_scratch
"""
from .prose_150M_fluid5_6x_unified_task2 import Cfg as _Base


class Cfg(_Base):
    adv_bank = True
    adv_bank_M = 4
    adv_bank_K = 4
    save_dir = "/eu/results/prose_150M_fluid5_6x_unified_task2_advb_scratch"
    ckpt_every = 2000
