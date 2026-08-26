"""Phase-2 pretrain, CROSS-CHANNEL ADVECTION variant — tests the hypothesis that the chronic NS weakness
is the MISSING structured self-advection −u·∇z (the NS nonlinearity). In all prior runs the convective
expert was DEAD: self_advect=False + coeffs≡0 → no transport at all; and latent_decoder=True meant the
physics experts ran on LATENT channels (slot 0,1 ≠ physical velocity), so even helmholtz's ω/δ were
mis-applied. Here:
  • latent_decoder=False  → experts run on PHYSICAL channels (slot 0=vx, 1=vy) so velocity is real.
  • conv_self_advect=True  → restore −u·∇z (advects velocity, tracer, density, … by the state velocity).
  • conv_upwind=True       → first-order UPWIND (stable); the earlier central-diff self-advection blew up
                             (incom 311% @ step 500, 2026-06-11). Upwind is the standard stable scheme.
Compare its per-family scOT metric (esp. NS-Sines uv/tracer) vs the latent baselines. NOTE: this also turns
latent_decoder OFF (required so velocity is physical), so it is a distinct arch arm, not a 1-flag ablation.
"""
from .poseidon_pretrain_ivp import Cfg as _Base


class Cfg(_Base):
    latent_decoder = False         # physical channels → slot 0,1 = real velocity (REQUIRED for self-advection)
    conv_self_advect = True        # restore the NS nonlinearity −u·∇z (was effectively absent in all runs)
    conv_upwind = True             # stable upwind (central-diff self-advection diverged: incom 311%)
    save_dir = "results/poseidon_pretrain_ivp_advect"
