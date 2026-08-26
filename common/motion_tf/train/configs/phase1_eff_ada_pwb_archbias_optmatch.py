"""Phase-1 ADA — archbias with the optimizer/LR schedule MATCHED to the PROSE/BCAT baselines.
Isolates whether ADA's small accuracy edge is helped/hurt by its optimizer differing from the baselines.
Baselines (felix-lyx, both identical): AdamW, lr 1e-4, weight_decay 1e-4, beta_2 0.95, cosine, warmup 3125
(=10% of 31250). archbias default was: plain Adam, beta_2 0.999, no weight_decay, warmup 1000 (~3%).
This config flips ADA to the exact baseline optimizer recipe; everything else = archbias.
Launch (2 GPU): bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwb_archbias_optmatch
"""
import os
from .phase1_eff_ada_pwb_archbias import Cfg as _Base


class Cfg(_Base):
    optimizer = "adamw"          # was plain Adam
    weight_decay = 1e-4          # was 0 (plain Adam)
    beta_2 = 0.95                # was 0.999 (TF default)
    warmup = 3125                # was 1000 (~3%) -> 3125 (~10%), matching PROSE/BCAT
    # lr=1e-4, lr_decay=True (cosine, alpha=0.05), clipnorm=1.0, steps=31250 already inherited & matched.
    save_dir = "results/" + os.environ.get("AB_SAVE", "phase1_int_ada_152m_6fam_pwb_archbias_optmatch")
