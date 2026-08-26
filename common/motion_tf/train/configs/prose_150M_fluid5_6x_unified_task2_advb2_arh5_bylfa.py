"""arh5_bylfa — byl STANDARD + FULL AR (10-segment, per-frame re-anchoring, = BCAT's per-step feedback).
byl5s(5-seg) showed AR gain is turbulence-localized (pdea -0.88/unc -0.38 vs byl, incom +0.26). Full-AR
pushes re-anchoring to every frame: Nt=2 (1 future/seg) x 10 segments = 10 futures. n_modes=4 (a 1-future
arc needs ~2 modes). N_p=8. Judged vs byl (3.175) and byl5s. Divergence risk higher (more re-anchor
boundaries) — LR-aligned resume patch available if it NaNs.
  bash runners/_run_prose_stream_eu.sh motion_tf.train.configs.prose_150M_fluid5_6x_unified_task2_advb2_arh5_bylfa
"""
from .prose_150M_fluid5_6x_unified_task2_advb2_arh5_byl import Cfg as _Base


class Cfg(_Base):
    Nt = 2
    T_final = 0.1
    ar_seg = 10
    N_p = 8
    n_modes = 4
    save_dir = "results/prose_150M_fluid5_6x_unified_task2_advb2_arh5_bylfa"
