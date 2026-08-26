"""PHASE-1 Task-1 ABLATION — ENCODER-side fix for the NS/turbulence over-smoothing.

Root cause (code): the state is avg-pooled 128²→32² (latent_factor=4) before the experts — avg-pool is a
hard LOW-PASS that destroys turbulent high-frequency content; bilinear upsample can't recover it. This run
replaces that lossy downsample with a DEPTHWISE-EXPANDING one (enc_expand): keep the avg-pooled state
(slots 0,1 intact → ω/div/u·∇ operators unchanged) AND concat a depthwise strided conv (depth_multiplier
=enc_expand_mult) that packs the LF×LF detail into extra feature channels → the coarse-grid expert input
is OVERCOMPLETE (32²×(16+16·M) ≫ 128²), so high-freq SURVIVES the coarse dynamics. General, no equation
key, no PDE-specific structure (pure re-gridding) → applies to all families incl. compressible.

ONLY the encoder downsample changes (loss/normalization = baseline instance-norm + clamped rel-L2, NOT
the physloss change) → clean isolation vs the Task-1 baseline curve (overall 11.60% @15k; SWE 0.62 /
com 1.17 / incom 5.98 / pdearena_ns 25.19 / cfd 0.58). Target: pdearena_ns + incom_ns drop."""
from .prose_150M_fluid5_6x import Cfg as _Base


class Cfg(_Base):
    enc_expand = True
    enc_expand_mult = 4        # 16 state ch → +64 hf ch = 80 ch at 32² (32²·80 ≈ 5×128², overcomplete)
    save_dir = "results/prose_150M_fluid5_6x_encexp"
