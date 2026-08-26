"""Phase-1 Task-1, UNIFIED single-expert redesign. The K-expert mixture + softmax router COLLAPSED to
diffusive-only (router_probe: every family → diffusive 1.0, zero per-family differentiation). Instead of
fixing the router, REMOVE routing: one UnifiedExpert computes ALL physics operators (Δ, ∇, −u·∇ advection,
ω, δ + dilated-conv global coupling) as a feature bank and a single head combines them; the transformer
becomes a feature ENCODER (encode_dim=d_geom, no expert softmax). ADA basis + latent + hard-IC unchanged
(the structure that beat SOTA). Goal: keep the SOTA-winning solver while letting operators beyond diffusive
(advection/vorticity/...) actually be used. Compare per-family rel-L2 vs baseline-lup at matched steps."""
from .prose_150M_fluid5_6x_lup import Cfg as _Base


class Cfg(_Base):
    experts = ("unified",)         # single physics expert (no routing → no collapse)
    unified_expert = True          # transformer = feature ENCODER; expert sees all operators + encoder feats
    lambda_gate = 0.0              # NO routing → kill the load-balance reg (it ran on encoder features =
                                   # unbounded/negative → drove the loss to −∞: the step-750 NaN divergence)
    save_dir = "results/prose_150M_fluid5_6x_unified"
