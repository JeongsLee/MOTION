"""PHASE-2 IVP pretrain with the CONFIRMED architecture = the lean single UnifiedExpert (no routing).
Phase-1 settled on it: the K-expert softmax router COLLAPSED to diffusive-only (zero per-family
differentiation); replacing it with ONE UnifiedExpert (full physics-operator bank → one head) + the
transformer as a feature ENCODER beat the 187M baseline with 39% fewer params (class-avg 6.31 vs 6.44).
This brings that exact arch into Phase-2 (IVP: T_in=1, all2all, Poseidon pretraining set) for the
Poseidon transfer comparison — same engine (ADA basis + latent + hard-IC), just the unified head.

Mirrors `prose_150M_fluid5_6x_unified` over the IVP base: experts=("unified",), unified_expert=True,
lambda_gate=0 (the routing load-balance reg ran on encoder features → drove loss to −∞; only fix needed),
learned_upsample=True (lup recovers velocity high-freq), latent_decoder stays ON (the standard)."""
from .poseidon_pretrain_ivp import Cfg as _Base


class Cfg(_Base):
    experts = ("unified",)         # single physics expert (no routing → no collapse)
    unified_expert = True          # transformer = feature ENCODER; expert sees all operators + encoder feats
    lambda_gate = 0.0              # NO routing → kill load-balance reg (ran on encoder feats → −∞ divergence)
    learned_upsample = True        # bilinear + learned zero-init high-freq residual (velocity high-freq)
    save_dir = "results/poseidon_pretrain_ivp_unified"
