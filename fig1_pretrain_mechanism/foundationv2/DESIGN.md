# foundationv2 — Tensor-Product ADA (TP-ADA): analytic integration in x, y, and t

**Status: design draft (2026-07-12). Numerical closed-form validation: `validate_tpada_math.py` (all checks pass).**

## -1. Project end-goal (stated by the user, 2026-07-12): a GENERAL PDE foundation model

foundationv2's ultimate target is one pretrained weight set that (a) handles 2D AND 3D problems,
(b) accepts grids, meshes, or point clouds (Transolver-style input), with or without internal
geometry, and (c) is consumed Poseidon-style: finetune-based downstream adaptation. Training merges
the maximum available corpus — the PROSE/BCAT fluid families AND the Poseidon pretraining operators
AND mesh/geometry datasets (Geo-FNO/Transolver benchmarks, CFDBench geometry, PDEBench 3D) — even
though prior comparisons used them separately. The TP-ADA machinery below is the enabling core:
analytic synthesis makes OUTPUT mesh-free natively (arbitrary query coordinates via eval tables),
the per-axis tables make 2D→3D a one-more-mode-product change, and hard IC/BC + the learnable
symbol M are axis-structured and therefore dimension-agnostic. Architecture sketch ("universal
latent-box TP-ADA"): point/mesh/grid encoder → latent box (2D or 3D) → shared axis-factorized
physics core (+ symbol M, SL transport) → TP-ADA synthesis → pointwise decode at arbitrary (x, t).
Known risks: steady-state mesh benchmarks need a relaxation-trajectory or steady-head formulation
(precedent: phase2 Poisson-Gauss); 3D memory requires a latent 32³–48³ box and reduced N_p.
Roadmap: P1 = current 2D arms (evidence) → P2 = 2D corpus unification + mesh encoder/decoder →
P3 = 3D latent box on PDEBench-3D → P4 = joint 2D+3D single-weights pretrain + finetune suite
(Poseidon downstream + Transolver benchmarks + 3D).

## 0. One-paragraph summary

The current foundation model (manuscript, "advb2" arm now converging on VESSL) applies ADA as a
*pointwise temporal* operator: each of the 128×128 pixel-channels carries an independent panel vector
`W(x) ∈ R^{N_p}` that is analytically integrated in time only. All spatial structure is injected upstream
(windowed-transformer encoder, tendency heads) or patched downstream (semi-Lagrangian transport branch),
and the manuscript itself names the structural gap: *"a pointwise temporal integral cannot move a feature
across grid cells."* foundationv2 removes that gap at the representation level: the **same** panel tensor
`W ∈ R^{128×128×N_p}` is reinterpreted as a 3-D panel field on (x-panels)×(y-panels)×(t-panels), and the
fixed panel→spectrum projection + closed-form anti-derivative machinery of ADA is applied **along all
three axes** (tensor product). The result is a single analytic expression `u(x,y,t)` that is (a) globally
coupled in space, (b) evaluable at any (x,y,t) — native super-resolution, (c) equipped with exact hard
initial conditions in t **and hard boundary conditions in x,y** via null-space projection of the panel
tensor (the ADA-L terminal-bias construction, `ada_l_terminal_bias_full.pdf`, generalized per axis), and
(d) analytically differentiable — ∂t, ∇, Δ in closed form, enabling exact mesh-free PDE-residual losses.
The encoder, physics bank, and transport branch of the manuscript architecture are retained unchanged;
only `evolve` (the basis synthesis) and the panel-forming map change, and both are fixed linear maps.

## 1. What exists today (facts the design builds on)

From NTO-ADA (`v6/NTO-ADA`, `models/panel_basis_common.py`, `lpa_basis.py`, `adaf_basis.py`):

- ADA is a **pure anti-derivative operator**: `z(t) = ic + ∫₀ᵗ f(s)ds`. No damping/λ anywhere.
- The network's DOF is `W`: piecewise-constant panel values of the integrand `f` on a uniform partition.
- `W` is projected onto Legendre (LPA) or Fourier (ADAF) modes by a **fixed, precomputed, exact-L²
  projection matrix** `C` (`a_m = Σ_i C[m,i] W_i`); anti-derivatives `Ψ_m(t)` are precomputed symbolically
  with `Ψ_m(0)=0`, so `g₁(0)=ic` holds structurally.
- Crucially, the whole machinery is **axis-agnostic**: pure einsum over the last (panel) axis, batched over
  everything else. Nothing in it is specific to time.

From the foundation model (manuscript §2, `pdefoundation/code/model.py`, latest = `…task2_advb2[_ext]`):

- Encoder: windowed MHSA, evaluated **once** on the input (physics-space feature extractor, not a router).
- Physics bank: zero-init-gated learned tendency heads (dilated-conv transport/multi-scale, 1×1 reaction,
  buoyancy, shear, wave); learned heads beat the hand-coded ∇/Δ operator bank.
- ADA synthesis: dual Fourier+Legendre bases blended by learned per-pixel μ(x); hard IC; single forward.
- Semi-Lagrangian transport branch (time-modulated velocity vocabulary, backward characteristics,
  zero-init residual): "the single most consequential component" on advection-dominated families.
- Best matched-budget result: class-avg rel-L2 **3.89%** vs BCAT 6.11% / PROSE-FD 6.63%.

Documented weaknesses this project targets:

1. **Weak spatial correlation of snapshots.** Per-pixel time integration never couples neighbors inside
   the integrator; turbulent PDEArena shows chaotic *phase* decorrelation (~2× gap to BCAT), and the
   transport/phase banks were bolted on precisely to compensate.
2. **Resolution-locked output.** Continuous in t, discrete 128² in x,y; super-resolution requires learned
   upsamplers with known trade-offs (lup helps velocity high-freq, hurts smooth density families).
3. **Boundary conditions unimplemented.** DESIGN.md (v1) specified hard-BC modulators; `modulators/` is
   empty; all trained runs are periodic.

From `ada_l_terminal_bias_full.pdf` (ADA-L working note):

- Any endpoint value of any anti-derivative order is a **linear functional of the panel weights**,
  `v^⊤W` with `v = C^⊤h` (h = Legendre endpoint sequence: `P_n(±1)`, `I⁽ᵏ⁾_n(1)`, `δ_{n0}`).
- Stacking constraints into `B W = c` and forming `W = B⁺c + (I − B⁺B) W̃` makes every predicted panel
  satisfy them **exactly, by construction, penalty-free** — condition-number-1 projection, scale-preserving.

## 2. Core formulation

### 2.1 Representation

Domain `Ω×[0,T] = [x_a,x_b]×[y_a,y_b]×[0,T]`, mapped per axis to the reference interval ξ∈[−1,1]
(Legendre axes) or [0,L] (Fourier axes). The network predicts, per (batch, latent channel),

```
W̃ ∈ R^{N_px × N_py × N_pt}        (e.g. 128 × 128 × 64 — SAME shape/size as today's W)
```

interpreted as coefficients of a **piecewise-linear (PL) panel field** (per-axis nodal hat functions;
piecewise-constant is the degenerate fallback and keeps today's semantics 1:1):

```
w(x,y,t) = Σ_{ijk} W_{ijk} · h_i(x) h_j(y) h_k(t)          (piecewise trilinear)
```

PL panels are the user-requested upgrade: the projection integrals `∫ h_i(ξ) P_n(ξ) dξ` (and the Fourier
analogues `∫ h_i cos/sin`) remain closed-form (Legendre recurrences / elementary trig integrals —
verified in `validate_tpada_math.py` §1), and a C⁰ integrand yields one extra order of smoothness in u.

### 2.2 Per-axis spectral projection (fixed maps)

```
A = W ×₁ Cˣ ×₂ Cʸ ×₃ Cᵗ ,   A ∈ R^{(Nx+1)×(Ny+1)×(Nt_m+1)}
```

`Cᵃˣⁱˢ[n,i] = ⟨h_i, φ_n⟩/⟨φ_n,φ_n⟩` precomputed symbolically once per axis; `φ` = Legendre `P_n`
(non-periodic axes) or Fourier `{1, cos k_n·, sin k_n·}` (periodic axes). This is the exact multi-axis
generalization of `panel_basis_common`; the three mode-products are three einsums,
`O(B·C·N_px·N_py·N_pt·max(N))` — cheap relative to the encoder.

### 2.3 Analytic synthesis and the integration orders (k_x, k_y, k_t)

```
u(x,y,t) = u_lift(x,y,t) + Σ_{nml} A_{nml} · Φ⁽ᵏˣ⁾_n(x) Φ⁽ᵏʸ⁾_m(y) Ψ⁽ᵏᵗ⁾_l(t)
```

where `Φ⁽ᵏ⁾_n` is the k-th closed-form anti-derivative of basis function n (k=0 → the basis itself), all
normalized to vanish (with derivatives) at the axis base point, and `u_lift` carries the IC/BC data
(§2.5). `k_t = 1` always (that *is* ADA: hard IC, tendency semantics). For space, two staged variants:

- **V-spec (k_x = k_y = 0)** — space is a pure spectral expansion of the tendency field. Minimal change
  from today: the temporal integral is unchanged, but the spatial field is now a *global analytic*
  expansion instead of 128² independent samples. Already delivers: continuous (x,y) evaluation
  (super-resolution), exact ∇/Δ by termwise differentiation, hard BC via endpoint functionals (§2.4),
  and gradient coupling across space through shared coefficients. **This is the Phase-A/B workhorse.**
- **V-int (k_x = k_y = 1)** — full "integrate along all axes": `w` becomes a mixed density
  (∂³-like object) and u gains two orders of spatial smoothness per axis; boundary values *and* normal
  derivatives become endpoint functionals (Dirichlet+Neumann hard BC, k=2 if both ends of both types).
  Strictly more structure, strictly harder to normalize (deep anti-derivatives shrink scale by
  `scale^{-k}`); treat as an ablation after V-spec works, not as the default.

Both variants were verified numerically: the 2-D tensor-product closed integral matches nested
quadrature to the quadrature's own error, and hard IC holds to 2e-16 (`validate_tpada_math.py` §2–4).

### 2.4 Hard boundary conditions (the ADA-L projection, per axis)

For each non-periodic axis, boundary functionals are `v^⊤` acting on that axis of `W`:
value at endpoints `v(±1) = C^⊤P(±1)` (V-spec) or anti-derivative endpoint columns (V-int).
Stack the axis's conditions into `B_x ∈ R^{q×(N_px+1)}` and apply, along axis 1 (and similarly axis 2):

```
W = lift(c) + (I − B_x⁺B_x) ⊗ (I − B_y⁺B_y) ⊗ I  · W̃
```

- **Homogeneous BC** (Dirichlet-0, Neumann-0): the per-axis null-space projectors are Kronecker factors
  on different modes → they **commute exactly** (verified to 3e-15); order of application is irrelevant.
- **Inhomogeneous BC**: standard tensor-product boundary lifting — `u_lift` = sum of edge terms minus
  corner bilinear correction (transfinite interpolation), with the edge data expanded in the
  complementary axis basis. Closed-form, precomputed per family geometry.
- **Periodic axes** (NS): use the Fourier basis on that axis — periodicity is automatic, no projector.
- **Time axis**: hard IC as today. For the residual-decoder path (hi-latent style), `z(x,y,0)=0`
  structurally and `u = u₀ + Dec(z)` with bias-free Dec; for super-resolution output `u₀` is queried by
  spectral (sinc) interpolation, as NTO-ADA Burgers already does.

This is exactly the manuscript's missing `modulators/` implemented as **fixed linear algebra**, and it is
the direct 2-D generalization of the terminal-bias note: same `C`, same `B⁺`/null-space construction,
condition number 1, zero-init-friendly.

### 2.5 What the analytic representation buys downstream

1. **Spatial coupling inside the integrator.** Every output point reads every panel through `A`; the loss
   gradient at one location distributes over spatially-extended panel sets. Snapshots are C⁰/C¹ fields by
   construction rather than bundles of independent pixel trajectories.
2. **Native super-resolution / mesh-free query.** `u` at arbitrary (x,y,t) = three small evaluation
   einsums against precomputed tables. Train at 128², evaluate at 256²/512² or at scattered points —
   no learned upsampler. (PL sampling of the *input* remains a floor; the representation adds none.)
3. **Exact physics losses.** ∂t u, ∇u, Δu, u·∇u are closed-form (termwise anti-derivative/derivative
   tables; Δ is diagonal in Fourier, banded in Legendre coefficient space). PDE residuals can be
   evaluated at random collocation points with zero discretization error — a mesh-free physics-informed
   term that supervises sub-grid spatial structure, impossible with the FD/FFT residuals used so far.
4. **Coefficient-space physics heads (optional, later).** Tendency heads may act directly on `A`
   (e.g. exact heat semigroup `e^{−νk²τ}` on Fourier axes), turning some banks from learned stencils
   into exact operators.

### 2.6 What it does NOT claim

- Chaotic phase decorrelation on turbulence is a *predictability/dynamics* limit; TP-ADA changes the
  representation, not the chaos. The semi-Lagrangian transport branch stays (and gains: the foot of the
  characteristic can sample the *analytic* u₀ representation at exact locations instead of bilinear).
- Spectral truncation `N ≤ N_p` per axis is a smoothing prior — good for the smooth/low-freq regime ADA
  already favors, a hyperparameter to fight on broadband turbulence (keep N as high as panel count
  permits on spatial axes; Legendre recurrence evaluation is stable at order ~128).

## 3. Architecture: kept vs. replaced

| Component | Disposition |
|---|---|
| Windowed-transformer encoder (physics-space features, evaluated once) | **kept** |
| Learned tendency-head physics bank, zero-init gates, lead-time embedding | **kept** |
| Semi-Lagrangian transport branch (time-modulated velocity vocabulary) | **kept** (samples analytic u₀) |
| Dual Fourier/Legendre time bases + learned mix μ | **kept** (per-axis basis choice now also encodes BC type) |
| `W` producer (`operator`) — shape `(B, 128,128, C_w, N_p)` | **kept**; add per-axis BC projector as the final fixed map |
| `evolve` — per-pixel 1-D ADA einsum | **replaced** by TP-ADA synthesis (three mode-product einsums + evaluation tables) |
| `modulators/` (empty) | **implemented**: BC projectors + transfinite lifting |
| Learned upsampler (lup) | **dropped** (native analytic evaluation) |
| FD/FFT PDE residual | **replaced** by exact collocation residual (when w_pde>0) |

Framework: stay on TF 2.15 to reuse the advb2 codebase and the mGPU/XLA recipe (the new module is
einsums + precomputed constant tables — trivially portable later). The tf.while_loop panel rollout is
unaffected: TP-ADA changes only the synthesis stage after `W` exists.

## 4. Phased plan

**Phase 0 — numerics module (1 file + tests).** `tpada/basis.py`: per-axis table builder
(PL/PC panels × Legendre/Fourier × anti-derivative order k × BC projector), sympy-precomputed, cached to
disk; `tpada/synthesis.py`: mode products + evaluation at grids/points; extend `validate_tpada_math.py`
into a test suite (add: Fourier axes, V-int k=1, inhomogeneous lifting with corners, gradient checks).

**Phase A — Burgers (x,t), the 2-D closed-form proof.** Port the NTO-ADA Burgers `ado` to TP-ADA
(x: Legendre + hard Dirichlet projector, t: dual-basis ADA). Deliverables: (i) match or beat the
per-pixel baseline at 128; (ii) **super-resolution table** — train 128, evaluate against the 512 MUSCL
ground truth at 256/512 with no retraining; (iii) hard-BC exactness at machine precision; (iv) exact
collocation PDE-residual ablation.

**Phase B — Navier–Stokes 64² (x,y,t), periodic.** Fourier spatial axes. Deliverables: (i) spatial
energy-spectrum and two-point-correlation comparison of snapshots vs the per-pixel-ADA baseline (the
direct test of the "weak spatial correlation" complaint); (ii) super-resolution vs a 256² reference run;
(iii) SL-branch-on-top interaction.

**Phase C — foundation integration.** Swap TP-ADA into the advb2 architecture; multi-family training on
the Benchmark-I corpus; per-family axis-basis/BC configuration (periodic → Fourier, walls → Legendre +
projector); add the super-resolution and hard-BC evaluations as new benchmark axes no baseline
(BCAT/PROSE-FD/Poseidon) can enter — this is the paper's differentiating claim.

## 4b. Correlated TP-ADA: learnable stationary space-time symbol (added 2026-07-12)

The fixed band-limit S imposes correlation by truncation only, per time panel independently. The
extension (user direction: *learnable, minimal structure*): project W to (x-mode, y-mode, t-panel-DCT)
space and multiply by ONE free learnable diagonal `M[n_x, n_y, m_t]` (channel-shared, init = 1 ⇒
bit-identical to fixed-S at start; ~467k weights at H=42, N_p=64). Diagonality in modes is the only
retained bias — it is exactly the class of stationary (translation-equivariant) space-time correlation
operators; the kernel shape, including cross-axis coupling (nearby panels ↔ nearby pixels), is fully
learned. Mathematically an FNO-style spectral multiplier, but placed on the *tendency* panel tensor
inside the analytic integrator: hard IC/BC and the continuous-evaluation machinery are untouched.
Implementation: `tpada_sym` flag in `model.py` (`_tpada_spatial`), fixed transforms
`fourier_projection`/`center_eval_matrix`/`dct_matrix` in `code/basis/tpada_basis.py`, config
`phase1_eff_ada_pwb_archbias_combo_tpada_sym.py`. Note: heads are zero-init ⇒ W≡0 ⇒ dL/dM = 0 at
step 0 (verified nonzero once W ≠ 0) — same warm-up behavior as every zero-init bank in this codebase.

## 4c. Universal architecture spec (P2+, committed 2026-07-12)

Decision (user): enough 2D evidence is accumulating — go DIRECTLY to 3D + point I/O in one step,
rather than 2D-mesh first. The 2D 6-family protocol stays as a REGRESSION GATE: the universal
model must match the combo/tpada curves on it.

```
input  : point set {(x_i ∈ R^K, u_i(t_0..t_{Tin-1}), g_i)}, K ∈ {2,3}; grid = cell-center points
         g_i = SDF φ, mask χ, normals ∇φ, coefficient slots; global descriptor (dropout-able)
         HYBRID CONDITIONING (user, 2026-07-13): T_in is VARIABLE (1..10, validity-masked — the
         pdearena_uncond t14 precedent) and the descriptor is droppable, trained with stochastic
         masking of both. One weight set then serves all four quadrants {IVP, history}×{equation
         given, inferred}: history frames = implicit system identification (PROSE/BCAT mode);
         descriptor = explicit operator (Poseidon-finetune mode); IVP without either is honestly
         under-determined (identical ICs under different coefficients) and the measured degradation
         of that quadrant QUANTIFIES the ambiguity. ADA's hard IC anchors on the last observed
         frame, so history length is free; Poseidon trajectories train as all2all pairs.
         SYMBOLIC EQUATION (optional, user 2026-07-13): third conditioning source — the governing
         equation serialized as an operator/field/coefficient token sequence (coefficients embedded
         as log-scale Fourier features), encoded by a small (~2-5M) transformer into e_eq, injected
         at the descriptor site (the 6-dim descriptor is its degenerate special case). Absent ⇒
         learned null token; trained with stochastic dropout like the descriptor. IVP + symbolic is
         the WELL-POSED quadrant (resolves the IVP ambiguity explicitly, vs Poseidon's per-task
         finetune workaround); symbolic composition targets zero-shot structure transfer on unseen
         term combinations (the Wave-Layer OOD failure mode). Data cost ~0: one symbolic string per
         family (all corpus equations are known), per-sample coefficients where available. PROSE
         precedent: symbolic-token equation input is its defining feature — natural comparison axis.
[1] encoder    grid path: windowed-MHSA patchify (validated; warm-start compatible)
               mesh/point path: cross-attention scatter — latent-box nodes query input points
[2] core       latent box (2D 96²–128², 3D 32³–48³, D=16–32); archbias tendency heads (zero-init)
               + K-dim SL transport branch + learnable symbol M; pw_batched panel rollout → W
[3] synthesis  TP-ADA per-axis closed-form tables; per-axis basis = BC metadata
               (periodic → Fourier, wall → Legendre + null-space projector); dual time basis,
               hard IC; N_p 24–32 in 3D
[4] decoder    analytic synthesis at arbitrary (y, t) → bias-free pointwise MLP → u = u0(y) + δ
               loss = random collocation subset of NATIVE data coordinates (no full-grid output)
```

Optional-input contract (user Qs, 2026-07-13): every auxiliary input is a (value, validity) pair;
DERIVABLE quantities are preprocessing, GENUINE information gets learned nulls + dropout:
- mesh input: connectivity-free point-cloud view; latent-box nodes cross-attend to points within a
  radius with relative-position encoding (GINO-neighborhood-like); grids keep the patchify path
  (degenerate case, shared downstream). Point-free latent regions: learned default token +
  point-density "coverage" channel.
- SDF: NOT a data requirement — a deterministic preprocessing output of whatever geometry exists
  (none → sentinel + flag 0; mask → distance transform; surface mesh → signed point-to-surface
  distance on the lattice; native SDF → as-is).
- symbolic equation: genuine information → learned null token + dropout even when available (all
  our corpora have known equations — the "absent" quadrant is trained by masking, not by data
  scarcity); PARTIAL specification supported via an unknown-remainder token (e.g. RANS closure).

Split policy (user Q, 2026-07-13): (1) OFFICIAL splits wherever they exist — DrivAerNet++ ID
files (~70/15/15 by design), AirfRANS task splits (incl. Reynolds/AoA extrapolation OOD),
Geo-FNO file conventions, Poseidon downstream protocol; external comparability depends on this.
(2) Otherwise deterministic TRAJECTORY/DESIGN-unit splits 80/10/10 train/val/test (upgrade from
P1's val==test conflation: inline eval and ckpt selection on VAL; TEST touched once for final
numbers). (3) Leakage rule: split units are trajectories/designs, never windows/frames (multi-
window sampling stays within one split — the shard-per-file mechanism already guarantees this).
(4) Foundation-level partition: reserve whole datasets as downstream-only finetune tasks
(Poisson-Gauss, Wave-Gauss, CE-RM, FNS-KF, Helmholtz, SE-AF + ShapeNet-Car as the held-out mesh
task); mesh sets otherwise participate in pretraining. All splits materialized ONCE as manifests
in /corpus/meta/splits/ (ids + provenance + seed); every preprocessing/training job reads
manifests only.

Mechanism banks + steady/unsteady bifurcation (user directive, 2026-07-14; core/banks.py):
- COMBO tendency banks (NOT adv-specialized): 5 zero-init gated mechanism heads on the latent box
  — reaction (pointwise), diffusion (Laplacian), buoyancy (gravity-axis ∂), shear (streamwise ∂),
  wave (2nd-order) — plus a small-init always-on base. Additive: W = base + Σ σ_k·head_k. Gates σ_k
  learnable AND multiplied by a per-family metadata mask (registry `operators`) so irrelevant
  mechanisms stay pruned. Rationale: the manuscript showed adv_bank sharpens in-domain but HURTS
  OOD transfer; a foundation model built for finetune-transfer wants the general combo vocabulary.
- STEADY vs UNSTEADY bifurcation (user: "steady solves better WITHOUT the time banks"): registry
  `time` selects the path. UNSTEADY → TP-ADA tendency trajectory (banks → W → coef → integrate,
  hard IC). STEADY → SteadyField elliptic head: box features → field coefficients → spectral
  field_at(query); NO time integration, NO tendency banks, no IC anchor. This is the per-problem
  pruning at the steady/unsteady granularity; directly targets airfrans/elasticity/Poisson which
  the trajectory path leaves stuck at rel-L2≈1.0 (confirmed in the run7 first EVAL). Batching now
  buckets by (K, steady) so each micro-batch is mode-homogeneous.
- RECURSION dial (to add): unsteady re-applies the banks on the evolved state (combo-rec), a
  speed↔accuracy lever; all of {combo banks, recursion, ADA} present and per-problem gated.

Four committed design decisions:
1. **One-weights 2D/3D via axis-factorized operators**: core convs/attention are compositions of
   per-axis 1D ops + pointwise mixing, so parameters are dimension-independent (K axes = K
   applications). Consistent with TP-ADA's per-axis tables end-to-end. Risk: 2D regression vs
   full-2D convs — gated by the 6-family protocol.
2. **Symbol M factorized for 3D**: free-diagonal M explodes at 3D; use
   M = M_r(|k|, m_t) · Π_a M_a(n_a) (radial component + per-axis 1D corrections, ~10³ params,
   dimension-agnostic). The 2D free-M arm (tpada_sym, running) empirically informs this: inspect
   the learned M's radial structure to justify the factorization.
3. **Steady problems = relaxation trajectories** (pseudo-time to steady state; precedent: phase2
   Poisson-Gauss), so transient and steady share one temporal machinery.
4. **BC as per-dataset metadata**, not learned: outer box faces hard (projectors); internal
   geometry soft (SDF channels + masked loss) first; embedded hard BC is follow-up research.

Point-I/O reference models (mesh-specialized, beyond Transolver/GNO):
- **GINO** — GNO encoder → regular latent grid → FNO core → GNO decoder at query points; the
  closest prior to our latent-box pattern. Our delta: replace the FNO rollout core with TP-ADA
  analytic synthesis (continuous output, hard IC/BC, whole trajectory in one forward).
- **UPT / AB-UPT** — supernode point encoders at automotive scale (500k+ points), anchored decode.
- **GNOT / OFormer** — heterogeneous cross-attention query decoding (our decoder pattern).
- **Transolver / Transolver++** — physics-attention slices on industrial meshes (baseline + encoder candidate).
- **MeshGraphNets** — mesh-intrinsic message passing (baseline; not our pattern).
- **PointTransformer v3 / Erwin** — hierarchical point attention for 10⁵–10⁶-point efficiency.
- **F-FNO / FactFormer** — factorized spectral/attention layers; supports decision 1.
Subsampling practice (DrivAerNet++: ~5·10⁵ surface points → 20k random per step) matches our
collocation-loss design natively.

**Scale plan (2026-07-12): build at B, train the flagship at L.** Bring-up/regression at ~150M
(= Poseidon-B 158M, apples-to-apples with all existing comparisons; cheap debugging). The universal-
corpus flagship pretrain scales to ~600M (= Poseidon-L 629M): P1 showed capacity must follow data
(100M lost to 5M at n900, won with full data), the merged corpus ≫ the 1M-view budget, and the
memory constraint is rollout activations — not parameters — so axis-factorized encoder/head growth
scales params without touching the activation ceiling. Reference points: Poseidon T/B/L =
21M/158M/629M, scOT 40M, DPOT-S/M = 30M/122M, Transolver-class = millions-scale (not foundation).

**Recursion as a speed↔accuracy dial (user, 2026-07-12).** P1 evidence: ADA-combo-rec (same heads
re-evaluated on the Euler-evolved state = the AR mechanism) was the decisive NS lever — overall
4.78→4.14, uncond 10.71→8.47 (beats BCAT 9.80), pdearena 13.65→12.39 — at ~3× step cost, which is
why the single-shot was the main model. Universal design: (1) train DUAL-MODE with stochastic
recursion depth so ONE weight set serves both single-shot (~1s, real-time) and recursive (~3s,
turbulence/precision) inference; AR-favoring downstream tasks finetune in rec mode. (2) New
hypothesis to test: recursion on top of tpada/sym re-evaluates a CORRELATED (symbol-filtered)
state rather than raw per-pixel fields, so the AR error-accumulation channel should be better
conditioned than in P1 — measure whether rec's gain grows. (3) 3D cost controls: family-gated
recursion (turbulent only), shallow K=2 recursion first, optional rec-teacher → single-shot-student
distillation.

## 5. Risks / open questions

1. **Scale bookkeeping in V-int** (`scale^{-k}` per axis) — normalize per axis as `lpa_basis` does; start V-spec.
2. **Memory**: `A` is comparable to `W`; evaluation tables are tiny. Rollout-activation ceiling
   (O(N_p×heads)) unchanged. Mode products add ~one W-sized temporary per axis — fits the 58.7GB/80GB profile.
3. **Inhomogeneous corners** (Phase A/C, Dirichlet walls): transfinite lifting is standard but must be
   validated to machine precision before training (Phase 0 test).
4. **Truncation vs turbulence**: keep spatial N at panel count; ablate. TP-ADA is not expected to close
   the BCAT turbulence gap by itself (see §2.6) — the win axes are correlation structure, super-res, BC, physics loss.
5. **IC at super-resolution**: hard IC anchors to spectrally-upsampled u₀; document as the input-information floor.

## 6. Provenance

- NTO-ADA formulation & shapes: `v6/NTO-ADA/{burgers,navier_stokes}/code/models/*` (agent survey, 2026-07-12).
- Foundation attempts & lessons: `v6/pdefoundation/{DESIGN,STATUS,PHASE1_ATTEMPTS_SUMMARY,phase2*}` (agent survey).
- Current architecture & results: `foundationv2/manuscript.pdf` §2, §4–6; latest arm =
  `code/train/configs/prose_150M_fluid5_6x_unified_task2_advb2_ext.py` (VESSL, convergence extension).
- Hard-constraint machinery: `foundationv2/ada_l_terminal_bias_full.pdf`.
- Closed-form validation: `foundationv2/validate_tpada_math.py`.
