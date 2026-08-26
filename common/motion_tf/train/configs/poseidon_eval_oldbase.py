"""EVAL-ONLY: re-measure the OLD explicit-operator 158M base (cont ckpt_60000, the 8.769% @final run) with
the CURRENT SOL-SLOT inline metric, so it is apples-to-apples vs the combo scratch run (same test set, same
de-diluted solution-slot metric, both @final and full-traj). Inherits the 158M explicit arch; save_dir is
pre-seeded (in the launch cmd) with the ckpt named ckpt_60000.npz -> train_ivp resume loads it; steps=60000
-> loop empty (start=60001) -> only log_eval("final") runs.
"""
from .poseidon_pretrain_ivp_unified_158m import Cfg as _Base


class Cfg(_Base):
    steps = 60000                  # == ckpt step -> resume sets start=60001 -> loop skips -> eval only
    ckpt_every = 0
    eval_every = 0
    reader_threads = 1
    save_dir = "results/_eval_oldbase"
