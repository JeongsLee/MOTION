"""Finetune the ~162M capacity-scaled combo pretrain (geo_layers 14 / router_depth 4 / n_latent 24). Same
FT_TASK-parametrized N-shot protocol as poseidon_finetune_combo; only the capacity knobs are raised so the
model builds at the 162M arch that matches the 158m-scratch ckpt (positional load, same var set — no migration).
"""
from .poseidon_finetune_combo import Cfg as _Base


class Cfg(_Base):
    geo_layers = 14
    router_depth = 4
    n_latent = 24
