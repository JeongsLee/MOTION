"""AR 5-SEGMENT ARM — even shorter ADA horizon than arh5. Each ADA call predicts only t=0.2 (Nt=3 → 2
futures); FIVE segments are chained (free-running, stop-grad boundaries) to cover the 10-future benchmark.
Pairs with advb2_arh5 (2 segments × 5 futures, t=0.5) to probe the horizon-length axis directly: does an
even shorter per-segment horizon help chaotic NS more (smaller e^(λT) per segment), or does the error
accumulated over MORE re-anchors (4 boundaries vs 1) start to hurt? (10 futures ÷ 4 isn't integer, so 5
segments of 2 is the clean "more/shorter" point; 10 = 5×2.)
Same recipe/budget as advb2/arh5, from-scratch, 10-future eval (comparable). Launch (EU capella 2xH100):
  bash runners/_run_prose_stream_eu.sh motion_tf.train.configs.prose_150M_fluid5_6x_unified_task2_advb2_ar5seg
"""
from .prose_150M_fluid5_6x_unified_task2_advb2 import Cfg as _Base


class Cfg(_Base):
    Nt = 3                # MODEL ADA horizon: 3 frames = t=0.2 (2 futures/segment)
    T_final = 0.2
    data_nt = 11          # data/loss horizon: IC + 10 futures
    ar_seg = 5            # 5-segment autoregressive (5 × 2 = 10 futures)
    save_dir = "results/prose_150M_fluid5_6x_unified_task2_advb2_ar5seg"
    ckpt_every = 10000
    ckpt_keep = 1
    eval_every = 2000
