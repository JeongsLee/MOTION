"""LEAD-MATCHED ANCHOR SWEEP (the corrected-benchmark metric): anchors {0,1,2,4,8}, every anchor scored
on the SAME lead set 1..12 (anchor 8 + lead 12 = frame 20 stays inside the data), full-traj = mean over
those 12 leads, median over the 128 held-out trajectories. Closes the protocol hole (anchor-0/IC-
distribution specialization) while keeping horizons comparable across anchors; the information hole
(hidden state) is closed separately by the v-channel. EVAL-ONLY (FT_STEPS=2, vanishing LR)."""
from .poseidon_finetune_combo_158m import Cfg as _Base


class Cfg(_Base):
    eval_anchor_sweep = (0, 1, 2, 4, 8)
    eval_lead_cap = 12
