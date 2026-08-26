"""combo-IVP pretraining, FROM SCRATCH to 320k (single clean cosine) — the fair base matched to the old
explicit-operator base's 320k budget. Replaces the warm-restart continuation (which re-warmed the LR to
peak and bumped eval 16.47→17.5 — a two-schedule artifact). One cosine over 320k = cleaner experimental base.
320k = matched to the old base (its eval was @320000) ≈ Poseidon training budget. Single-shot ~0.22s/step
→ ~19.5h. Tests if combo (learned heads) matches/beats the old base per-family at equal budget, esp. NS.
Launch: bash /code-vol/code/runners/_run_ivp_eu.sh motion_tf.train.configs.poseidon_pretrain_ivp_combo_320k
"""
import os
from .poseidon_pretrain_ivp_combo import Cfg as _Base


class Cfg(_Base):
    steps = 320000                 # single cosine over the full matched budget (was 120000)
    warmup = 2000
    save_dir = "results/poseidon_pretrain_ivp_combo_320k"
