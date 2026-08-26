"""arh5_big — stronger AR + BCAT-level capacity + FASTER, all at once.
Key: arh5 ran N_p=64/n_modes=32 which is ~10x oversampled for a short AR segment (a 3-frame arc needs
~3 modes). Cutting to N_p=8/n_modes=8 (params UNCHANGED — quadrature resolution, not weights) frees the
expert-bank compute (bank cost ∝ B·N_p) to fund BOTH 5-segment AR and a bigger encoder.
Benchmarked (H100, fwd+bwd, accum=4): N_p8/modes8/seg5/geo_layers=13 = 757 ms/step = 0.72x the current
arh5 (2-seg/114M/N_p64 = 1052 ms). So: 5x finer AR re-anchoring + 152M (≈ BCAT 155.6M) + 28% faster.
Targets pdearena turbulence — manuscript diagnosis is transport-PHASE misplacement; finer AR re-anchoring
holds phase (BCAT-like) and the added encoder capacity closes the params gap to BCAT. From-scratch.
  bash runners/_run_prose_stream_eu.sh motion_tf.train.configs.prose_150M_fluid5_6x_unified_task2_advb2_arh5_big
"""
from .prose_150M_fluid5_6x_unified_task2_advb2_arh5 import Cfg as _Base


class Cfg(_Base):
    Nt = 3                # MODEL ADA horizon: 3 frames = t=0.2 (2 futures/segment)
    T_final = 0.2         # ADA basis spans [0, 0.2]
    ar_seg = 5            # 5-segment autoregressive (5 × 2 = 10 futures; data_nt=11 inherited)
    N_p = 8               # was 64 — 10x oversampled for a 3-frame segment; params unchanged
    n_modes = 8           # was 32 — 3-frame arc needs ~3 modes
    geo_layers = 13       # 9 -> 13 : ~152M ≈ BCAT (155.6M) — closes the capacity gap
    save_dir = "results/prose_150M_fluid5_6x_unified_task2_advb2_arh5_big"
    ckpt_every = 10000
    ckpt_keep = 1
    eval_every = 2000
