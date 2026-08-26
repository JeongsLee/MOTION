"""SMOKE of the Task-2 unified 6-family config: verify the 6-family stream (incl pdearena_uncond) +
temporal-mask threading + model forward + eval all RUN on 1 GPU, and that the added uncond family shows
up in the per-family EVAL log. ~40 steps, tiny test set. NOT a real run."""
from .prose_150M_fluid5_6x_unified_task2 import Cfg as _Base


class Cfg(_Base):
    steps = 40
    warmup = 5
    ckpt_every = 20                # eval at step 20, 40, final
    test_max_per_fam = 16          # 6 x 16 = 96 eval samples (fast materialize)
    save_dir = "results/_smoke_unified_task2"
