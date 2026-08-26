"""SMOKE of the Phase-2 IVP unified config: verify the confirmed UnifiedExpert arch builds + forwards in
the IVP/8-channel/all2all setting on 1 GPU, runs steps, evals, and checkpoints. ~40 steps. NOT a real run."""
from .poseidon_pretrain_ivp_unified import Cfg as _Base


class Cfg(_Base):
    steps = 40
    warmup = 5
    ckpt_every = 20
    test_per_fam = 16
    save_dir = "results/_smoke_ivp_unified"
