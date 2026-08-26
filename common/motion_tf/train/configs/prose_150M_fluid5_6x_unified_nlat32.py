"""PHASE-1 Task-1 (UNIFIED) — widen the FEATURE bottleneck: n_latent 16→32. The only lever that ever worked
(encexp) widened the most-compressed representation; n_latent is the tightest one — ALL dynamics + the ADA
basis flow through this latent feature dim (per (pixel,channel) temporal trajectory). Decoupled from
trajectory smoothness (that's n_modes/N_p). Config-only; basis/decode/expert-out auto-adapt. vs unified
Task-1 baseline (pdf-fluid5-unified2, @15k overall 10.43 / pdearena 22.10 / com 3.24)."""
from .prose_150M_fluid5_6x_unified import Cfg as _Base


class Cfg(_Base):
    n_latent = 32
    save_dir = "results/prose_150M_fluid5_6x_unified_nlat32"
