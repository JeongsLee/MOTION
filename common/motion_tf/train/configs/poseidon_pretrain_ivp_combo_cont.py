"""combo-IVP pretraining CONTINUATION → 320k total (matched to the old base / Poseidon-B budget).
The first run (poseidon_pretrain_ivp_combo) stopped at 120k; the old explicit-operator base trained ~320k,
so the 120k-vs-320k comparison was unfair (combo 2.7x under-trained, esp. NS). Warm-restart from ckpt_120000
with a FRESH cosine over 200k more steps → 320k total. Tests whether NS (combo NS-Sines 84% @120k) recovers
with matched budget (step-deficit) or stays weak (arch limit of single-shot T_in=1 on forced NS).
Launch: bash /code-vol/code/runners/_run_ivp_eu.sh motion_tf.train.configs.poseidon_pretrain_ivp_combo_cont
"""
import os
from .poseidon_pretrain_ivp_combo import Cfg as _Base


class Cfg(_Base):
    init_from = "/eu/results/poseidon_pretrain_ivp_combo/ckpt_120000.npz"   # warm-restart (fresh step counter + re-warmed LR)
    steps = 200000                 # +200k → 320k total trained
    warmup = 2000                  # re-warm the LR after restart
    save_dir = "results/poseidon_pretrain_ivp_combo_cont"
