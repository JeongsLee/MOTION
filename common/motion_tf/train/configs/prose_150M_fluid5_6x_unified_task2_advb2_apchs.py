"""APC + HSPLIT stacked arm — combine the two orthogonal turbulence levers in ONE model to test whether
they COMPOUND or are REDUNDANT. hsplit (per-group latent differentiation, feature/subspace axis, +840
params) and panel_advect (advective W-panel coupling, time/separability axis, +273 params) act on
different parts of the pipeline, both zero-init (warm-safe, identical to advb2 at step 0). Individually
each edged pad on turbulence (pdea/uncond) but only in the noisy tie-band; stacking tests additivity.
From-scratch, single-shot (t=10), advb2 protocol. Launch (EU capella 2xH100):
  bash runners/_run_prose_stream_eu.sh motion_tf.train.configs.prose_150M_fluid5_6x_unified_task2_advb2_apchs
"""
from .prose_150M_fluid5_6x_unified_task2_advb2 import Cfg as _Base


class Cfg(_Base):
    unified_hsplit = 3
    unified_hsplit_hidden = 8
    panel_advect = True
    save_dir = "results/prose_150M_fluid5_6x_unified_task2_advb2_apchs"
    ckpt_every = 10000
    ckpt_keep = 1
    eval_every = 2000
