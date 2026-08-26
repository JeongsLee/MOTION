"""Phase-1 ADA — combo (single-shot, all-levers) with N_p=24 instead of 64.
The output is only Nt=11 frames, so N_p=64 (n_modes=32) is ~6× over-parameterized in time. The recursive
np24 run confirmed N_p=24 (n_modes/max_order=12) MATCHES N_p=64 accuracy (even slightly better on pdea).
For the SINGLE-SHOT pw_batched path, N_p is folded into the batch dim, so dropping 64→24 should make the
bank ~2.7× cheaper → faster main model at the same accuracy (unlike the recursive path, where the sequential
while_loop serialization dominates and N_p barely matters). A/B vs combo (N_p=64, 4.78) at 1M: confirm
accuracy held + measure the single-shot speedup.
Launch (2 GPU): AB_SAVE=phase1_int_ada_152m_6fam_pwb_archbias_combo_np24 bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwb_archbias_combo_np24
"""
import os
from .phase1_eff_ada_pwb_archbias_combo import Cfg as _Base


class Cfg(_Base):
    N_p = 24                     # was 64 — single-shot folds N_p into batch → ~2.7× cheaper bank
    n_modes = 12                 # Fourier modes = N_p//2 (Nyquist)
    max_order = 12               # Legendre order, matched ratio
    save_dir = "results/" + os.environ.get("AB_SAVE", "phase1_int_ada_152m_6fam_pwb_archbias_combo_np24")
