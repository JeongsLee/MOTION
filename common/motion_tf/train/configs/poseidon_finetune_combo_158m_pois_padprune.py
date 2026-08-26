"""Poisson pad+prune combination: the steady task's solution frame also sits at the temporal-basis
endpoint (2-frame IVP framing, solution at tau=1) — pad it interior (Nt 2->4, span x3, data frames
aligned) ON TOP of the unsteady-head pruning that set the 5.65% record. Both levers are orthogonal
(basis geometry vs head interference).
Launch: FT_TASK=Poisson-Gauss FT_UV_ONLY=0 FT_IC_FRAC=0 FT_LOSS_CAP=100 ... FT_TAG=_162m800000padprune"""
from .poseidon_finetune_combo_158m_prune import Cfg as _Base

_NTD = int(_Base.Nt)


class Cfg(_Base):
    nt_data = _NTD
    Nt = _NTD + 2
    T_final = float(_Base.T_final) * (_NTD + 1) / max(_NTD - 1, 1)
