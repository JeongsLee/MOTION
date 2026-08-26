"""AUTOREGRESSIVE ANCHOR SWEEP (hop lead 2): from each anchor {0,1,2,4,8}, roll out by re-feeding the
model's own lead-2 output; score at the hop grid A+2, A+4, ... <= 20 (same targets for scOT's AR — its
full-T training grid is even leads, so hop 2 is in-distribution for both models). Diagnoses whether our
single-shot long-lead deficit is integration accumulation (AR helps: short hops avoid the far-lead
panels) or hedge-floor (AR compounds: each re-feed feeds a smoothed state). EVAL-ONLY."""
from .poseidon_finetune_combo_158m import Cfg as _Base


class Cfg(_Base):
    eval_anchor_sweep = (0, 1, 2, 4, 8)
    eval_ar_step = 2
