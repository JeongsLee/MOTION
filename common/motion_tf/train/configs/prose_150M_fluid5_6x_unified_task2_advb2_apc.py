"""APC ARM - advb2 recipe + ADVECTIVE W-panel coupling (panel_advect). Panel k tendency receives the
IC-velocity-advected previous panel (semi-Lagrangian) -> injects the advective space-time correlation the
SEPARABLE ADA basis lacks (pdearena turbulence). zero-init (+273 params) -> identical to advb2 at step 0.
From-scratch, protocol identical to advb2/hsplit3 (matched-1M @125k). Unlike AR: no feedback -> no accumulation.
Launch: bash runners/_run_prose_stream_eu.sh motion_tf.train.configs.prose_150M_fluid5_6x_unified_task2_advb2_apc
"""
from .prose_150M_fluid5_6x_unified_task2_advb2 import Cfg as _Base


class Cfg(_Base):
    panel_advect = True
    save_dir = "results/prose_150M_fluid5_6x_unified_task2_advb2_apc"
    ckpt_every = 10000
    ckpt_keep = 1
    eval_every = 2000
