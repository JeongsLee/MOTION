# foundationv2 — Experiment Log (2026-07-16 → 07-18)

Universal PDE foundation model, single weight set over 2D + 3D + mesh (26 families).
Metric = class-average rel-L2 (normalized). Trained on VESSL H100s.
Rendered view: https://claude.ai/code/artifact/28cdb59b-c68d-4141-8a2b-90a10a68201f

## Headline
- **Best: 18.09%** — `v30` (run40), MoE E16 + full physics banks, warm-started.
- **After 18.09, 0 of 8 levers beat the floor.**
- **Root cause: coarse-box representation ceiling** (not optimization).

## What moved the number (the wall-breaker)
| Step | Change | class-avg |
|---|---|---|
| baseline | pre-physics | 20.6% |
| run35 | + nonlinear advection `−u·∇h` (learned u) | 18.83% |
| run37 | + 11 physics banks | 18.60% |
| run40 (**v30**) | + boundary-layer bank + MoE E16 (sparse upcycle) | **18.09%** |
| run43 | kitchen: + dual-core | 18.08% |

The breakthrough was **structure, not scale**: the parent model (advb2) had hardcoded ∇/Δ operators
with a *nonlinear* self-advection; foundationv2 had dropped it (all banks linear). Restoring the full
vocabulary — advection, shock (upwind), elliptic, geometry (curvature), Bernoulli, kinematic, boundary-layer
— plus MoE capacity set the floor.

## Every lever tried after 18.09 (all warm-safe, warm-started from v30)
| Lever | Mechanism | Result | Verdict |
|---|---|---|---|
| Box 2× warm-start (run41, 96/48) | double box resolution from v28 | 23% flatline | ❌ warm-start shake, no recover |
| Dual-core h1⊥h2 (run43) | parallel orthogonal core, VICReg decorrelation | ablation Δ≈0 | ❌ h2 different but not useful |
| Recursion full-window (run44, rec4/6) | re-apply banks at window end | no gain | ❌ rec>3 breaks XLA; rec6 OOM |
| Semi-Lagrangian (run45/46, `--sl_steps`) | N sub-windows, re-eval banks along pathline; exact telescoping | no gain | ⚠️ warm-safe (2e-10) but floor-stuck; xla thrash/OOM |
| steady_mode=field (run47) | direct SteadyField elliptic head | 20.6% | ❌ worse than pseudo-time ADA |
| steady transient-mask | drop wave/shock/buoyancy for steady | confounded | ⚠️ op_multihot already prunes; no win |
| Geometry — box path (8-ch recache) | SDF/normal/curvature into encoder | Δ 0.09% | ❌ box scatter-mean blurs it |
| Geometry — decoder FiLM (drivaernet ft) | native-res per-query normal/curv modulates decoder | 28% (=base) | ❌ can only modulate a box-limited base |
| Big box from scratch (v40, 2×H100) | 96/48 dense, MirroredStrategy | plateau 25% | ❌ above the warm-start floor |

## Diagnosis (three probes)
1. **Data clean** — drivaernet pressure std≈1, scale≈120, coords∈[0,1]³, NaN=0, coords↔label aligned. Not a label bug.
2. **Geometry blurred, not absent** — encoder geom_gate norm 8.16 (active) but zeroing all geom channels moves drivaernet rel-L2 by **0.09%**. The coarse box destroys the signal.
3. **Coarse-box ceiling** — 32³ box + band-limited ADA synthesis can't represent fine mesh/turbulent fields. GINO/ME-GNN reach 14–16% at mesh-native resolution. Our box limits both geometry input and field output.

Every post-18.09 lever rearranged the latent/integration path; none touched the resolution bottleneck → all hit the same floor.

## Where the error lives (v30)
- **Converged (<13%):** CE-KH 1.4, shallow_water 0.5, ACE 3.3, incom_ns 3.3, com_ns 3.6, cfdbench 4.8, diff_react 5.5, geofno_pipe 7.5, Poisson 8.0, CE-Gauss 11.4, 3D-CNS M0.1 12.4, geofno_elasticity 14.7.
- **Laggards (>18%):** NS-Sines 47.2, Wave-Layer 44.0, airfrans 35.7, pdearena_ns 31.7, NS-Gauss 30.6, CE-CRP 30.3, drivaernet 28.0, 3D-Turb 24.3, pdearena_uncond 24.3, CE-RP 22.9, 3D-CNS M1.0 20.5, shapenet 18.8.
- Two clusters, one root: **turbulence** (chaotic/high-freq) + **mesh geometry** (surface detail). Both need resolution the box can't supply.

## Vs benchmarks
| Family | Ours (v30) | SOTA | Standing |
|---|---|---|---|
| drivaernet pressure | 28.0% | 16% GINO / 14.16% ME-GNN | behind (coarse-box) |
| 3D-CNS M1.0 | 20.5% | 22.6% DPOT | **ahead** |
| 3D-CNS M0.1 | 12.4% | ≈ DPOT range | competitive |

Caveat: one weight set over all 26 families, not per-benchmark specialists.

## Code landed (warm-safe, reusable)
- 8-channel geometry pipeline + full re-cache (pyvista/scipy): `data/adapters.py` `_geom8`/`_surface_curvature`, `data/loader.py` geom lever, `data/registry.py` geom_kind, GEOM_MAX=8.
- Geometry-FiLM decoder: `core/pointio.py` `hid·(1+geom_gate(geom_feat(geom_q)))`; per-query geom threaded loader→batching→`point_traj_*`→dec.
- Semi-Lagrangian: `core/model.py` `from_points_sl`/`point_traj_sl_g`, `--sl_steps`; exact time-telescoping warm-safety.
- Steady transient-bank mask: `core/banks.py` `STEADY_DROP`, `steady=` flag threaded.
- 2-GPU data-parallel: `core/pretrain.py` MirroredStrategy + `ReductionToOneDevice` (NCCL broken in container); transparent at 1 GPU.
- Warm-start toolkit: axis-0 pad load (feature-width growth), enum-strip name remap, duplicate-half basis split, zero-init gates.

## Conclusion
The 18.09% floor is a **representation ceiling**, not optimization. More resolution failed both ways:
warm-start into a big box shook and stalled (23%); from-scratch big box plateaued *above* the warm-start
floor (25%). The remaining path is a **mesh-native (GNO/GINO-style) encoder–decoder** at native resolution
— a re-architecture, not a lever.
