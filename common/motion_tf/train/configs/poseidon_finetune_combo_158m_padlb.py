"""ACE pad+loose-bank combination: endpoint padding (basis geometry) x reaction vocabulary (loose bank)
are orthogonal mechanisms — pad alone 0.592, lb alone 0.648 (un-padded base 0.775). Name-keyed init
(lb_* fresh) from the migrated named ckpt, as in the lb arm.
Launch: FT_TASK=ACE FT_IC_FRAC=1.0 FT_TAG=_162m800000padlb FT_INIT_PATH=.../migrated/combo158m_800000_named.npz"""
from .poseidon_finetune_combo_158m_loosebank import Cfg as _Base

_NTD = int(_Base.Nt)


class Cfg(_Base):
    nt_data = _NTD
    Nt = _NTD + 2
    T_final = float(_Base.T_final) * (_NTD + 1) / max(_NTD - 1, 1)
