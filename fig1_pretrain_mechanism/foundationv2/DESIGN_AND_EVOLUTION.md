# foundationv2 — Design Principles & Evolution Log

A general PDE foundation model: **one weight set** over 2D + 3D + mesh/point-cloud, used
Poseidon-style via finetuning, built on the ADA (anti-derivative) synthesis. This document
records the *invariants we hold* and *how the model has grown* — so every future change can be
checked against the principles and the warm-start chain stays unbroken.

---

## Part 1 — Design Principles (invariants)

1. **ADA core (anti-derivative synthesis).** The field is synthesized from panel coefficients
   through closed-form anti-derivatives. Time uses the Legendre anti-derivative with `G(0)=0`
   → **hard IC is structural**. Space (as of the dual-basis work) can use an anti-derivative
   basis vanishing at the boundary → **hard BC is structural**. Closed-form derivatives/integrals
   are the point: they give (a) super-resolution in space & time, (b) exact PDE operators for the
   physics-residual endgame.

2. **One weight set, K-agnostic.** The encoder, LatentBoxCore (axis-factorized 1D ops), tendency
   banks, and decoder are **dimension-agnostic** — the SAME trainable weights serve K=2 and K=3.
   The only per-K objects are the synthesis tables (constants) + tiny per-axis separable gains.
   Adding a dimension = adding one axis to the tensor-product synthesis, not a new model.

3. **Warm-start expansion — never drop weights.** Every capacity/structure change is applied on
   top of the previous checkpoint via:
   - **name-keyed partial load** (new components resume from init),
   - **zero-init gated branches** (new module contributes 0 at load → output unchanged),
   - **n_p pad-load** (grow ADA time-panels; new panels 0 → trajectory unchanged),
   - **size-agnostic dims** (grow the latent box; weights unchanged, per-axis gains reset to
     identity).
   The original run13 weights have carried through every run since, unbroken.

4. **Gate every GPU launch behind a $0.30 CPU check.** Keras-2 (TF 2.15) compat + the warm-start
   load report (`loaded / padded / new-at-init / mismatch`) + a few grad steps. Catches basis/
   shape/autodiff bugs cheaply. **Validate new math in numpy locally first** (the gate cannot
   catch bugs that are invisible at the warm-start init point — e.g. a transpose that only bites
   once α,β≠0).

5. **Steady = t→∞ ADA relaxation.** Steady problems are solved as a 2-frame IVP through the SAME
   ADA operator: QoI(0)=0 + source/geometry forcing → QoI(1) = steady solution. Mirrors pseudo-
   transient continuation (the real steady-RANS method) and unifies steady+unsteady under one
   operator. (Validated originally on Poisson; the SteadyField head is the fallback `field` mode.)

6. **Balanced, never-abandon training.** The family interleave CYCLES each family forever
   (reshuffle per epoch) — a family is never dropped when its units run out. (The abandonment bug
   caused data-poor mesh families to be trained early then forgotten → divergence.) Keep the
   **best-eval checkpoint** separately + newest-3 (late training can overfit/degrade minority
   families; the last ckpt is not the best one).

7. **Geometry conditioning is universal but effective where geometry exists.** A zero-init geom
   branch takes SDF / boundary-layer mask / surface normals / elliptic source (0 where absent).
   Mesh & walled grids benefit; periodic families feed zeros and are inert. Geometry ≠ turbulence
   lever — turbulence needs capacity/recursion, not geometry.

8. **Dual basis, encoder-selected.** Fourier-ADA (periodic-optimal) and Jacobi-ADA (bounded/wall,
   Sturm-Liouville, trainable α,β) run in PARALLEL with separate coefficients; a zero-init gate
   from the encoder latent selects per family. This keeps periodic families safe (Fourier
   preserved) AND is warm-start output-preserving (Jacobi off at load). Jacobi = the trainable
   generalization of Legendre (α=β=0); its shape params control boundary-layer emphasis. A
   *linear* trainable basis is redundant (absorbed into coefficients) — expressiveness must come
   from **nonlinear** params (α,β / knots / frequencies) kept **closed-form integrable**.

9. **Honest, per-group bars — foundation ≠ specialist.** Success is not one class-avg number.
   Per group: smooth 2D <3–5; 3D physical ≤ DPOT 22.6 (beaten); turbulent/shock the laggard
   (target <15–20); mesh zero-shot = a *finetunable* init (specialist parity is a *finetune*
   claim). Ultimate proof = finetune-transfer matching Poseidon-B. Report physical-scale rel-L2
   for 3D/mesh (matches the literature); watch train-vs-test to separate overfitting from capacity.

10. **Space-time coupling.** The tensor product is in the BASIS (`φ_m(x)·G_p(t)`); coupling lives
    in the JOINT coefficient `A_{m,p}` (space-mode × time-panel) + the banks' spatial derivatives
    (∇, Δ so a cell's tendency depends on neighbors) + recursion (re-evaluate banks on the evolved
    state). Single-shot modal coefficients + fixed analytic time basis = efficient for smooth
    evolution, limited for chaotic turbulence (→ recursion / capacity levers).

---

## Part 2 — Current architecture (one line each)

- **Encoder**: GINO cell-binning (`BinEncoder`, O(N), scatter-mean into the latent box) + a
  zero-init universal geometry branch. Scales to enc_n 16k+ and fine meshes.
- **Core**: `LatentBoxCore` — depth × (axial 1D conv + axial attention + pointwise MLP),
  axis-factorized, K-agnostic (RankFreeLN for Keras-2 rank-agnostic norm).
- **Dynamics**: `TendencyBanks` (small-init base + 5 zero-init gated mechanisms
  reaction/diffusion/buoyancy/shear/wave, pruned by per-family operator metadata) → panel tensor
  W. `SteadyField` head is the `field`-mode steady fallback.
- **Synthesis**: `KAxisTPADA` — per spatial axis a basis projection (Fourier and, in progress,
  Jacobi-ADA in parallel) + the time-axis Legendre anti-derivative (hard IC). Recursion dial.
- **Decoder**: bias-free pointwise MLP on the latent trajectory z(x,t) (hard IC: MLP(0)=0) +
  Fourier-feature FiLM on query coords (zero-init) for sharp gradients.

---

## Part 3 — Evolution log (the warm-start chain)

Each run inherits the previous weights (mechanism in brackets). Metric = eval class-avg rel-L2
(normalized) unless noted; physical rel-L2 for 3D/mesh where stated.

| run | dir | change | warm-start | outcome |
|---|---|---|---|---|
| run12 | v2 | first stable full-stack (graph-mode + XLA + std-norm) | — | ~34% @2.5k; steady families unstuck by SteadyField |
| run13 | v3 | bin encoder + recursion(2) + t_pad 1.1 | fresh | ~31% @26k; airfrans STUCK ~100% (fatal) |
| run14 | v4 | **airfrans geometry levers** (SDF norm + boundary-layer mask + near-wall collocation) | init v3 (arch unchanged) | airfrans 100→~54; class-avg ~24% |
| run15 | v5 | +shapenet_car into pretrain | init v4 | shapenet 79→46, healthy |
| run16 | v6 | 26-family attempt (crashed on Poisson `_nc_load` ndim2) → 25-family | init v5 | ndim2 fix; ACE/Wave/CE-RM added |
| run17 | v7 | 26-family (Poisson source-conditioning) | init v5 | full corpus (field-mode steady) |
| run18 | v8 | **steady-through-ADA** (all steady via relaxation) | init v7 | Poisson 41→11 (win), drivaernet win; BUT geometry-mesh families DIVERGED late |
| — | — | **post-mortem**: train≈test (not overfit) → **`_interleave` abandonment bug** (data-poor families exhausted ~step 10k, then untrained) | — | root cause found |
| run19 | v9 | **interleave cycles forever + best-eval ckpt** | init v8 (diverged) | all families RECOVERED; class-avg → 21.2% |
| run20 | v10 | **dims3 16→24** (3D capacity) + **Fourier-FiLM decoder** + rec3 | init v9 (size-agnostic + zero-init) | 3D M1.0 24→21.6, Turb ~26; modest |
| run21 | v11 | **n_p 32→48** (ADA temporal capacity) | init v10 (**pad-load**) | best 21.18%; modest continued gain |
| run22 | v12 | **universal geom branch + surface normals(∇SDF, mesh) + dims3 32** | init v11 (zero-init + gain reset) | shapenet 30→24 (normals help); running |
| run23 | v13 (planned) | **dual basis (Fourier ‖ Jacobi-ADA, encoder-selected, trainable α,β)** | init v12 (Jacobi zero-gated → output unchanged) | pending |

**Recurring lessons.** (a) Divergence of minority families was a *sampling* bug, not capacity or
ADA. (b) Capacity levers (dims3, n_p) give *modest* 3D gains; geometry levers help mesh; neither
fixes turbulence. (c) Every "it got worse" was diagnosed with data (train-vs-test, eval trajectory,
exhaustion timing) before acting. (d) Validate new synthesis math in numpy — a transpose bug in
the Jacobi projection was invisible at α=β=0 and only caught by a round-trip test.

---

## Part 4 — Roadmap

1. **Dual-basis (in progress).** Jacobi-ADA parallel to Fourier; trainable α,β for boundary-layer
   adaptation on wall/mesh families. Warm-safe.
2. **Scale to B-level (~150–600M).** Continue warm-start growth (dims, n_p, banks, depth). Width
   (d) growth is the one non-trivial step → net2net-style expansion.
3. **Physics-residual self-supervision (endgame).** Past a competence threshold, add an
   unsupervised loss = the PDE residual `R[u] = ∂ₜu − F(u,∇u,Δu,…)`, computed with the ADA's
   **closed-form derivatives** (exact & cheap, unlike autodiff through a black box), with IC/BC
   satisfied structurally. The model self-evaluates its own solution and self-trains from PHYSICS
   given only (PDE, IC, BC, geometry) — the PDE analogue of an LLM that self-evolves past a
   threshold. Needs: differentiable symbolic-PDE→residual assembly, BC null-space projector,
   self-eval→self-train loop + problem generator. **Hardest exactly where we lag (turbulence/
   shock)** — credible first on smooth/elliptic/parametric families.

**The enabler:** the ADA/Jacobi substrate was chosen precisely because analytic derivatives +
structural IC/BC are the prerequisites for physics self-learning. The current design is a
coherent staircase toward that endgame.
