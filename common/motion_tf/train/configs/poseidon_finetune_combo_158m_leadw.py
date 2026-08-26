"""LEAD-WEIGHTED finetune arm (panel-gradient equalization): frame-k loss reweighted by k^1 so the
late ADA panels (supervised only by long-lead pairs) receive a balanced expected gradient. Direct test of
the panel-starvation hypothesis for the long-lead deficit (Poseidon wins only leads 13-20 on NS).
Launch: FT_TASK=NS-PwC FT_UV_ONLY=1 FT_IC_FRAC=1.0 FT_LR=5e-5 FT_STEPS=12800 FT_TAG=_162m800000leadw ..."""
from .poseidon_finetune_combo_158m import Cfg as _Base


class Cfg(_Base):
    lead_weight_pow = 1.0
