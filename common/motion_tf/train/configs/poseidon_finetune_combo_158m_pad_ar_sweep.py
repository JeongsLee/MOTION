"""pad+AR combination probe (eval-only): the padded NS ckpt evaluated in AR mode (hop-2) over anchors —
the 3.30% AR record was measured on the UN-padded ckpt; padding lowers per-hop error, so composition may
compound the gain."""
from .poseidon_finetune_combo_158m_pad import Cfg as _Base


class Cfg(_Base):
    eval_anchor_sweep = (0, 1, 2, 4, 8)
    eval_ar_step = 2
