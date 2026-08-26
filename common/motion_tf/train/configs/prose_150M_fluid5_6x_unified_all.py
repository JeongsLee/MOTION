"""Unified-ALL: every physics expert (convective+diffusive+reaction+elliptic+shock+helmholtz+wave+forcing)
ALWAYS-ON with full transformer-encoder context, summed, NO routing (no collapse). Adds the operators the
lean UnifiedExpert lacked — shock (conservation-form ∇·F̂ shock-capturing), explicit reaction, wave
(cross-channel restoring/acoustic), forcing (external source) — via their proven module implementations.
Convective uses MUSCL self-advection. ADA basis + latent + hard-IC + lambda_gate=0 (no routing reg)
inherited. Compare vs the lean unified (114M, ops=Δ/∇/adv/ω/δ) and baseline-lup at matched steps —
does the full operator set (esp shock for compressible) help further?"""
from .prose_150M_fluid5_6x_unified import Cfg as _Base


class Cfg(_Base):
    experts = ("convective", "diffusive", "reaction", "elliptic", "shock", "helmholtz", "wave", "forcing")
    unified_expert = True          # transformer = encoder; all experts get full context, summed (no routing)
    conv_self_advect = True        # convective: real self-advection −u·∇z
    conv_muscl = True              # MUSCL (2nd-order TVD) advection
    # lambda_gate=0 inherited (no routing → no load-balance reg)
    save_dir = "results/prose_150M_fluid5_6x_unified_all"
