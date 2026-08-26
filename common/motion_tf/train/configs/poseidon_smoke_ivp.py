"""Phase-2 IVP SMOKE config — tiny run to validate the loader/model/loss/eval end-to-end before the full
pretraining. Same architecture as poseidon_pretrain_ivp but steps=40, frequent ckpt/eval, small test set."""
from .poseidon_pretrain_ivp import Cfg as _Base


class Cfg(_Base):
    steps = 40
    ckpt_every = 20
    test_per_fam = 4
    save_dir = "results/poseidon_smoke_ivp"
