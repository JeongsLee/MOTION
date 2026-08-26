"""Phase-1 Task-1, CROSS-CHANNEL ADVECTION variant of the reported ADA (prose_150M_fluid5_6x_lup).
Tests whether the chronic NS weakness is the MISSING structured self-advection −u·∇z. In all prior runs
the convective expert was DEAD (self_advect=False + coeffs≡0 → no transport) and latent_decoder=True ran
the physics experts on LATENT channels (slot 0,1 ≠ physical velocity). Here, same two changes as the
Phase-2 advect arm:
  • latent_decoder=False  → experts on PHYSICAL channels (slot 0=vx, 1=vy real); SWE (height-only) has
                            no velocity → transport=0 there (correctly inactive).
  • conv_self_advect=True + conv_upwind=True → restore −u·∇z with the STABLE upwind scheme (central-diff
                            self-advection blew up: incom 311% @ step 500, 2026-06-11).
Compare per-family physical rel-L2 (esp. incom_ns / pdearena_ns velocity) vs the lup baseline.
NOTE: distinct arch arm (latent off required for physical velocity), not a 1-flag ablation.
"""
from .prose_150M_fluid5_6x_lup import Cfg as _Base


class Cfg(_Base):
    latent_decoder = False
    conv_self_advect = True
    conv_upwind = True
    save_dir = "results/prose_150M_fluid5_6x_advect"
