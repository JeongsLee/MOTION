"""Phase-1 ADA — combo_rec (recursive, all-levers) but with N_p=24 instead of 64.
Motivation: the output is only Nt=11 frames, so N_p=64 panels (n_modes=32) is ~6× over-parameterized in time
AND the sequential recursion costs N_p Euler steps → 64 is why combo_rec is slow (~3-4 s/step). Dropping to
N_p=24 (n_modes/max_order=12, keeping the N_p//2 ratio) should make the SAME recursion ~2.7× cheaper
(24 vs 64 sequential steps) → near single-shot speed, while the REAL recursion (self-rollout, train=infer
consistent) avoids the teacher-forcing divergence. Tests: (a) does N_p=24 keep accuracy (esp. turbulent
pdearena temporal content)? (b) how much faster is the recursion? A/B vs combo_rec (N_p=64) at 1M.
Launch (2 GPU): AB_SAVE=phase1_int_ada_152m_6fam_pwb_archbias_combo_rec_np24 bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwb_archbias_combo_rec_np24
"""
import os
from .phase1_eff_ada_pwb_archbias_combo_rec import Cfg as _Base


class Cfg(_Base):
    N_p = 24                     # was 64 — fewer panels → fewer sequential recursion steps → cheaper
    n_modes = 12                 # Fourier modes ≤ N_p//2 (Nyquist); keep N_p//2 ratio
    max_order = 12               # Legendre order, matched ratio
    save_dir = "results/" + os.environ.get("AB_SAVE", "phase1_int_ada_152m_6fam_pwb_archbias_combo_rec_np24")
