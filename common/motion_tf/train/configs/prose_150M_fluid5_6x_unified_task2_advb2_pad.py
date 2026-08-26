"""BENCHMARK-I PAD ARM: the advb2 recipe (paper basis, 3.89 @1M) with the temporally padded output grid —
the ADA basis span extends two frames past the data horizon at the same 1/10 frame spacing (Nt 11 -> 13,
T 1.0 -> 1.2, data frames 0..10 aligned; trailing pad frames zero-target + tmask-masked, the same
mechanism as pdearena_uncond's valid_t). Motivated by the Benchmark-II finding that the data-final frame
sits at the basis endpoint where both temporal bases are worst-conditioned; downstream this cut @final by
3-4 points on every task at no full-traj cost. Identical corpus/budget/eval to advb2 (matched-1M read at
step 125k); eval_every=2000 for a dense early curve (BCAT/PROSE log from ~8k views; advb2's first point
at 80k views made the efficacy plot start late).
Launch (EU capella 2xH100):
  bash _run_prose_stream_eu.sh motion_tf.train.configs.prose_150M_fluid5_6x_unified_task2_advb2_pad
"""
from .prose_150M_fluid5_6x_unified_task2_advb2 import Cfg as _Base


class Cfg(_Base):
    nt_data = 11
    Nt = 13
    T_final = 1.2
    save_dir = "results/prose_150M_fluid5_6x_unified_task2_advb2_pad"
    ckpt_every = 10000
    ckpt_keep = 1
    eval_every = 2000
