"""TEMPORAL-PADDING arm, task-generic: extend the ADA basis span 2 frames past the data horizon at the
SAME frame spacing, so the data-final frame is an interior point of the temporal basis instead of the
endpoint (Fourier wrap / Legendre max-oscillation). NS validated first (@final 17.0 -> ~13.2 immediately,
full-traj crosses the un-padded final by step 2400); this config adapts the pad to the task's frame count
(ACE 20 -> 22 model frames, others 21 -> 23; steady Poisson excluded — no time axis).
Launch (ACE):  FT_TASK=ACE ... FT_TAG=_162m800000pad  (Wave): FT_TASK=Wave-Layer ...
"""
import os
from .poseidon_finetune_combo_158m import Cfg as _Base

_NTD = int(_Base.Nt)                     # task's data frames (20 for ACE, 21 otherwise)


class Cfg(_Base):
    nt_data = _NTD
    Nt = _NTD + 2
    T_final = float(_Base.T_final) * (_NTD + 1) / (_NTD - 1)
