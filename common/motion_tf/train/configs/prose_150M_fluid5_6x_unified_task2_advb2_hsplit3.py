"""H-SPLIT ARM — advb2 recipe + per-group LATENT differentiation (3 groups). Instead of every physics
operator reading the SAME 16-dim latent z, each operator GROUP gets its own zero-init bottleneck-adapted
latent  z_k = z + Up_k(gelu(Down_k(z))):
    transport {advection, vorticity, divergence + velocity}  |  diffusive {Δ, ∂x, ∂y}  |  reaction {raw z}.
Zero-init Up → z_k == z at step 0 → IDENTICAL to advb2 at load; the groups differentiate through training.
Hypothesis (measured from the shared-latent baseline): the nonlinear transport operators can't be linearly
absorbed by the downstream head, so a transport-specific subspace should help the pdearena turbulence
laggards (pdearena_ns ~11.5%, uncond ~7.8%) without hurting the already-strong families.
From-scratch, corpus/budget/eval identical to advb2 (matched-1M read at step 125k). Paired with the
_ctrl arm (param-matched single shared adapter) to separate subspace-split from raw capacity.
Launch (EU capella 2xH100):
  bash runners/_run_prose_stream_eu.sh motion_tf.train.configs.prose_150M_fluid5_6x_unified_task2_advb2_hsplit3
"""
from .prose_150M_fluid5_6x_unified_task2_advb2 import Cfg as _Base


class Cfg(_Base):
    unified_hsplit = 3
    unified_hsplit_hidden = 8
    save_dir = "results/prose_150M_fluid5_6x_unified_task2_advb2_hsplit3"
    ckpt_every = 10000
    ckpt_keep = 1
    eval_every = 2000
