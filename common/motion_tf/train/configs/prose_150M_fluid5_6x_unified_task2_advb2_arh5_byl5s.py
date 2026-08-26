"""arh5_byl5s — byl STANDARD + 5-segment AR feedback (the campaign's proven lever, finer).
arh5's single big win was AR-t5 (2-seg: 3.89->3.50); per-frame analysis shows our deficit vs BCAT is
SHORT-LEAD (its per-step feedback zone). 5 segments x 2 futures = 5x finer re-anchoring. N_p8/n_modes8
(a 3-frame arc needs ~3 modes; params UNCHANGED - quadrature not weights, benchmarked neutral).
Capacity unchanged (114M) -> byl5s - byl isolates the feedback lever on the standard.
  bash runners/_run_prose_stream_eu.sh motion_tf.train.configs.prose_150M_fluid5_6x_unified_task2_advb2_arh5_byl5s
"""
from .prose_150M_fluid5_6x_unified_task2_advb2_arh5_byl import Cfg as _Base


class Cfg(_Base):
    Nt = 3
    T_final = 0.2
    ar_seg = 5
    N_p = 8
    n_modes = 8
    save_dir = "results/prose_150M_fluid5_6x_unified_task2_advb2_arh5_byl5s"
