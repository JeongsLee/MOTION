"""ABLATION of the unified expert: identical to prose_150M_fluid5_6x_unified EXCEPT the hardcoded physics
operator bank (Δ, ∇, −u·∇, ω, δ) is REMOVED — the single expert sees only [state z, transformer-encoder
features] → same dilated-conv head + ADA basis + latent + hard-IC. Tests whether the physics decomposition
is actually used, or the transformer-encoder + ADA basis alone produce the unified result. If this matches
unified → ops unused (it's the encoder+basis); if unified beats this → the physics operators contribute."""
from .prose_150M_fluid5_6x_unified import Cfg as _Base


class Cfg(_Base):
    unified_ops = False            # drop the physics-operator bank (encoder + conv head + ADA only)
    save_dir = "results/prose_150M_fluid5_6x_unified_noops"
