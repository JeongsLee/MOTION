"""ALL-3 stacked arm — AR-t5 + APC + hsplit combined. Tests whether the two feature/panel levers
(panel_advect = advective W-panel coupling; unified_hsplit = per-group latent differentiation) COMPOUND
on top of the dominant AR-t5 lever (short ADA horizon Nt=6 + 2-segment autoregressive). All three are
orthogonal (temporal rollout × panel-coupling × latent-subspace) and zero-init/warm-safe. Decomposition
set: arh5 (AR only) / apchs (APC+hsplit only) / all3 (this) → attributes each lever's marginal gain.
From-scratch, advb2 protocol, 2-seg AR eval (10-future, comparable). Launch (EU capella 2xH100):
  bash runners/_run_prose_stream_eu.sh motion_tf.train.configs.prose_150M_fluid5_6x_unified_task2_advb2_all3
"""
from .prose_150M_fluid5_6x_unified_task2_advb2 import Cfg as _Base


class Cfg(_Base):
    # AR-t5
    Nt = 6
    T_final = 0.5
    data_nt = 11
    ar_seg = 2
    # APC + hsplit
    panel_advect = True
    unified_hsplit = 3
    unified_hsplit_hidden = 8
    save_dir = "results/prose_150M_fluid5_6x_unified_task2_advb2_all3"
    ckpt_every = 10000
    ckpt_keep = 1
    eval_every = 2000
