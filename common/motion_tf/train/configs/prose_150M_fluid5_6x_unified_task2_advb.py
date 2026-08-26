"""LOOSE TRANSPORT bank arm (pdearena diagnosis-matched, Benchmark-I joint setting): the spectral-coherence
analysis shows the task2 model's pdearena gap is transport-phase misplacement — E(k) closer to GT than
BCAT's, but phase coherence collapsing in the filament band (k~15-35). This arm attaches the adv_bank
(time-modulated semi-Lagrangian velocity vocabulary, av_* fresh-init, gate zero-init) to the task2_r1 base
and CONTINUES joint training at matched budget vs the _cont2 control. Warm-start needs the NAMED ckpt
(positional would corrupt on the new vars) — migrate task2_r1/ckpt_56000 first.
Launch:  bash _run_prose_stream_eu.sh motion_tf.train.configs.prose_150M_fluid5_6x_unified_task2_advb
"""
from .prose_150M_fluid5_6x_unified_task2 import Cfg as _Base


class Cfg(_Base):
    adv_bank = True
    adv_bank_M = 4
    adv_bank_K = 4
    steps = 20000
    warmup = 500
    init_ckpt = "/eu/results/migrated/task2r1_56000_named.npz"
    save_dir = "/eu/results/prose_150M_fluid5_6x_unified_task2_advb"
    ckpt_every = 2000
