"""AR-t5 ARM — advb2 recipe with a SHORTENED ADA horizon + AUTOREGRESSIVE training. The ADA basis predicts
only t=0.5 (Nt=6 → 5 futures) per call instead of the full t=1.0 (10 futures); two segments are CHAINED
(free-running, stop-grad at the boundary) to cover the full 10-future benchmark horizon (data_nt=11).
Motivation: chaotic NS error ~ e^(λT); a t=10 single-shot integration blows up over the full horizon, while
two t=5 segments halve the per-segment exponent (1 re-anchor → minimal accumulation). Training FOR AR removes
the single-shot→AR distribution shift that made AR-EVAL of the single-shot model diverge (ext 3.09→10.07).
Makes the model BCAT-like (AR) where BCAT's turbulence edge lives, while keeping the ADA basis per segment.
Eval is 2-segment AR too → still the 10-future benchmark metric (comparable to pad/advb2/PROSE/BCAT).
From-scratch, corpus/budget identical to advb2. Launch (EU capella 2xH100):
  bash runners/_run_prose_stream_eu.sh motion_tf.train.configs.prose_150M_fluid5_6x_unified_task2_advb2_arh5
"""
from .prose_150M_fluid5_6x_unified_task2_advb2 import Cfg as _Base


class Cfg(_Base):
    Nt = 6                # MODEL ADA horizon: 6 frames = t=0.5 (5 futures/segment)
    T_final = 0.5         # ADA basis spans [0, 0.5]
    data_nt = 11          # DATA/loss horizon: IC + 10 futures (full benchmark), decoupled from model Nt
    ar_seg = 2            # 2-segment autoregressive train + eval (2 × 5 = 10 futures)
    save_dir = "results/prose_150M_fluid5_6x_unified_task2_advb2_arh5"
    ckpt_every = 10000
    ckpt_keep = 1
    eval_every = 2000
