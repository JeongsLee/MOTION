"""Anchor sweep of the DEFINITIVE wave arm (wv0pad: 2IC + pad + i>=0): checks the anchor-0 hole is
closed and gives the swept closed-benchmark row. EVAL-ONLY."""
from .poseidon_finetune_combo_158m_wv0_pad import Cfg as _Base


class Cfg(_Base):
    eval_anchor_sweep = (0, 1, 2, 4, 8)
    eval_lead_cap = 12
