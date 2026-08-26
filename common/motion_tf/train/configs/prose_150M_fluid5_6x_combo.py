"""PHASE-1 Task-1 — COMBO: capacity (enc_expand) + structured global coupling (multigrid elliptic).

Findings so far (vs Task-1 baseline curve, pdearena_ns 40.2→32.8 @1k→4k):
  - enc_expand (depthwise-expand encoder, +41M): clear win — pdearena 41.8→30.4, but capacity-confounded.
  - elliptic multigrid (lean 384, +14M): only MARGINAL (pdearena 45.2→31.6 @4k, ~baseline, < enc_expand).
Hypothesis: capacity (high-freq preservation) is the strong lever; a properly-sized multigrid global
coupling should ADD on top of it (complementary — encoder keeps small scales, multigrid does the non-local
elliptic pressure coupling). This run turns BOTH on, and gives the V-cycle real width (768, not the lean
384) so it isn't under-powered. If combo > enc_expand-alone, the multigrid structure adds value beyond
capacity. Non-spectral throughout (orthogonal to ADA time-Fourier). loss/norm = baseline."""
from .prose_150M_fluid5_6x import Cfg as _Base


class Cfg(_Base):
    # capacity / high-freq preservation (the proven lever)
    enc_expand = True
    enc_expand_mult = 4
    # structured non-local global coupling (the elliptic pressure operator as a real V-cycle), now wider
    elliptic_multigrid = True
    elliptic_mg_levels = 3
    elliptic_mg_hidden = 768
    save_dir = "results/prose_150M_fluid5_6x_combo"
