"""CORRECTED WAVE BENCHMARK sweep (2IC track): the wv0 ckpt evaluated with the v0-slot input at anchors
{0,1,2,4,8} (v0 = u_A − u_{A−1}; anchor 0 uses the true zero-velocity IC), lead-matched 1..12. This is the
closed-problem Wave benchmark ([u,v,c] per the Poseidon SM's own operator definition) — 1IC tracks are
either special-solution recall (anchor 0, v=0) or hedge-floor measurement (anchor>0). EVAL-ONLY."""
from .poseidon_finetune_combo_158m_wave_v0 import Cfg as _Base


class Cfg(_Base):
    eval_anchor_sweep = (0, 1, 2, 4, 8)
    eval_lead_cap = 12
