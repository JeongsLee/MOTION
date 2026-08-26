"""EVAL-ONLY re-measure of the RUNNING loose-phase Wave arm's latest ckpt (steps=1 < ckpt -> resume ->
loop skips -> final eval only). Reads the live training dir (read-only: ckpt_every=0)."""
from .poseidon_finetune_combo_158m_loosephase import Cfg as _B


class Cfg(_B):
    steps = 1
    ckpt_every = 0
    save_dir = "/eu/results/ftcombo_Wave-Layer_N64_162m800000pb"
