"""PHASE-1 Task-1 (UNIFIED baseline) — combine physics-ops ↔ transformer by CROSS-ATTENTION (not concat).

The unified head currently CONCATS [physics-op bank, transformer/encoder features] and lets a dilated-conv
learn the combination. This run replaces the concat with CROSS-ATTENTION: the 7 physics operators
(Δz, ∂x z, ∂y z, −u·∇z, ω, δ, z) are per-pixel TOKENS, the transformer feature is the QUERY, and a per-pixel
softmax over the operator tokens decides how much of each operator to apply WHERE (soft, data-driven,
per-pixel operator selection — unlike the hard softmax router that collapsed). Attention output → the same
dilated-conv head + ADA basis. Non-spectral, general, no equation key. Only this one change vs the unified
baseline → clean isolation. Compare to the unified Task-1 train curve (pdef-fluid5-unified2, RMS/T10:
@15k overall 10.43 / pdearena_ns 22.10 / incom 5.33 / com 3.24). Target: better op-combination → pdearena/
incom drop without regressing com_ns."""
from .prose_150M_fluid5_6x_unified import Cfg as _Base


class Cfg(_Base):
    unified_xattn = True
    unified_xattn_dim = 256
    save_dir = "results/prose_150M_fluid5_6x_unified_xattn"
