"""PHASE-1 Task-1 ABLATION — PHYSICAL-scale loss (option ii) vs the per-sample-normalized baseline.

Baseline `prose_150M_fluid5_6x` trains with PER-SAMPLE instance norm + clamped rel-L2. Both of those are
per-sample relative normalizations: instance-norm divides each sample by its OWN std, and rel-L2 divides
the error by ‖tgt‖ — together they scale every sample to ~unit energy, so a high-energy TURBULENT sample
contributes the same loss as a calm one. That systematically UNDER-WEIGHTS turbulence (the NS/pdearena
bottleneck). Poseidon/PROSE avoid this.

This run switches ONLY the loss/normalization (architecture identical → clean ablation vs the Task-1 curve):
  - global_norm=True  : Poseidon-style FIXED per-family per-channel normalization (not per-sample). A fixed
                        std PRESERVES each sample's physical-energy ratio (turbulent samples keep their
                        weight) while channels stay balanced (each ÷ its own family std → no density
                        domination). General, family-agnostic, NO equation key.
  - instance_norm=False (per-sample OFF; global_norm takes precedence in the stream)
  - loss_mse=True     : plain MSE, NOT rel-L2 — rel-L2's ÷‖tgt‖ is itself a per-sample relative norm that
                        would re-introduce the down-weighting even in global-normalized space.
Everything else (5 families, 15k steps, ckpt_every=1000, 6-expert set, lr=1e-4) inherited → the [EVAL]
physical rel-L2 curve is DIRECTLY comparable to the baseline (SWE 0.62 / com 1.17 / incom 5.98 /
pdearena_ns 25.19 / cfd 0.58, overall 11.60% @15k). Target: pdearena_ns + incom_ns improve."""
from .prose_150M_fluid5_6x import Cfg as _Base


class Cfg(_Base):
    instance_norm = False
    global_norm = True
    loss_mse = True
    save_dir = "results/prose_150M_fluid5_6x_physloss"
