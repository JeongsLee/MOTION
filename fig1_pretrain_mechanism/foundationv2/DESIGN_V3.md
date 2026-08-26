# DESIGN V3 — bylfa-paradigm universal AR model (2026-07-30)

## 0. Why a redesign

Three empirical verdicts drive v3 (user decisions, 07-30):

1. **Full-step AR + physics-meaning output slots won.** On Benchmark-I the byl arm family
   (semantic slots + typed decode) with **full 10-segment AR** (`bylfa`) is the final SOTA:
   2.774% vs byl5s 3.072 / byl 3.175 / arh5 3.501 / BCAT-fair 3.81. AR depth was monotone.
   foundationv2's single-shot window synthesis (TP-ADA over the whole horizon) is retired.
2. **Mesh geometry must be a first-class input.** v2's coarse-box scatter-mean blurred
   geometry (zeroing all 8 geom channels moved drivaernet by 0.09%); GeoFNO/Transolver/GINO
   ingest geometry natively. v2's EXPERIMENTS.md conclusion: mesh-native encoder–decoder at
   native resolution is a re-architecture, not a lever.
3. **Equation/IC/BC must be explicit inputs.** An IVP solver takes (equation, IC, BC).
   v2 had weak conditioning (op_multihot mask only); v3 promotes symbolic conditioning to an
   architecture-level input contract.

## 1. Assets inherited

| Asset | Source | What we take |
|---|---|---|
| bylfa paradigm | `/eu/code_mirror/code` (fetched 07-30, scratchpad/code_mirror) | per-frame AR chaining with stop-grad + hard-IC re-anchor each segment; `ar_fair_loss` (ONE masked-mean over all segments); clamped rel loss; 1-future ADA segment (N_p=8, n_modes=4); Lagrangian splat decode for transport slots (`lag_channel/lag_native`); buoyancy head; balanced sampling |
| 26-family corpus | foundationv2 `data/` | registry/adapters/schema/splits/materialized cache at `/corpus/cache/unified`; 9-slot channel schema + cmask/tmask; 8-channel canonical geometry; symbolic op/param vocab |
| K-agnostic substrate | foundationv2 `core/axops.py` | RankFreeLN, AxialConv, AxialAttention, PointwiseMLP — one weight set serves K=2 and K=3 (proven, `tests/test_core_kagnostic.py`) |
| Physics vocabulary | foundationv2 `core/banks.py` | the 12-mechanism tendency heads (advection −u·∇h with learned u, shock upwind, buoyancy, shear, reaction, diffusion, …) that broke the 20.6→18.09 wall |
| Ops discipline | DESIGN_AND_EVOLUTION.md | $0.30 CPU gate before every GPU launch; never-drop-family interleave; std-only normalization; named npz ckpts |

## 2. Architecture

### 2.1 Input contract (per sample)
- `window` (T_in≤10, N, S=9): field values at N sampled nodes (grid cells raveled or mesh points),
  right-aligned, std-only normalized. Steady: window = zeros (IC=0 relaxation start).
- `coords` (N, K) in [0,1]^K, K ∈ {2,3}.
- `geom` (N, 8): [SDF, BL mask, nx, ny, nz, mean curv, gauss curv, interior]; zero where absent.
- `cond` tokens: operator multi-hot (16) ⊕ param log-Fourier feats ⊕ per-axis BC type one-hot
  (periodic/dirichlet/neumann/open ×K) ⊕ steady flag ⊕ K — from `data/registry.py` + `data/symbolic.py`.
- `cmask` (9,), `tmask` (T,) as in v2. Targets: values at collocation query points per future frame.

### 2.2 One AR step (the model call) — predicts exactly ONE next frame
```
window(T_in,N,S) ⊕ geom ⊕ Fourier(coords) ⊕ e_cond
  → PointEncoder: kernel-weighted bin scatter into latent grid  (GINO-lite, O(N))
      w_p = MLP(Δx_cell, geom_p) per point; cell feat = Σ w_p·f_p / Σw ⊕ log-count
      [+ optional 1-block local surface attention over SDF≈0 points, flag `surf_attn`]
  → Core: depth × [AxialConv → AxialAttention → PointwiseMLP], each block FiLM-modulated by e_cond
  → TendencyBanks (v2's 12 mechanisms, zero-init gates, masked by operator multi-hot)
      → W (latent, d_w × N_p=8 panels)
  → ADA 1-future segment: dual-basis (Fourier ‖ Legendre, blend gated by BC token), n_modes=4,
      hard IC at segment start (closed-form anti-derivative, g(0)=0)
  → Decode at query coords q:
      Eulerian slots: multilinear latent interp at q → bias-free MLP × (1+gate(Fourier(q)))
                      × (1+gate(geom_q))   ← native-res geometry FiLM (the mesh fix)
      u(q) = u_prev(q) + δ(q)              ← hard IC preserved (δ(seg start)=0)
      Transport slots (2D grid families, `lag_native`): forward-Lagrangian particle splat —
      particles at grid nodes carry slot value along basis trajectories, differentiable bilinear
      splat; output = splat, not residual; gate by descriptor×temporal-variance as in byl.
```
`e_cond = MLP(cond tokens)` (d_cond=128), consumed as per-block FiLM (scale/shift, zero-init)
and as bank-gate bias. This is requirement 3: equation+BC condition every stage; IC enters
structurally (hard anchor).

### 2.3 AR rollout (training AND eval — no train/eval shift)
```
win = window0
for k in 1..K:                      # K = n_future frames (transient: 10; steady: K_relax=4)
    pred_k = model(win, q=nodes ∪ colloc_k)         # one frame
    loss_frames.append(pred_k at colloc_k vs target frame k)
    win = concat(win[1:], stop_gradient(pred_k at nodes))   # free-running, no BPTT
loss = ONE masked-mean over all K frames (ar_fair) with cmask ⊗ tmask, clamped rel-L1 (τ=4)
```
- Feedback happens at the **fixed sampled node set** (enc_n nodes) — identical mechanics for
  2D grid, 3D grid, and mesh; this is what makes full-step AR mesh-agnostic.
- Steady = t→∞ relaxation as K_relax AR steps from zero window; supervise the final state
  (+0.25-weighted penultimate for fixed-point consistency). Same operator, same loop.
- Hard IC is re-imposed at every boundary (bylfa property (i)): re-anchoring costs no
  consistency error, sequential depth K is independent of N_p.

### 2.4 What is deliberately NOT in v1 of v3
- No spatial anti-derivative (TP-ADA spatial path was measured dead in v2).
- No MoE at start (dense core; MoE E16 upcycle is a known warm-safe follow-up lever).
- No lag splat on 3D/mesh (transport slots don't exist there yet; flag-gated for later).
- Symbolic conditioning is token-features → FiLM, not a string transformer: registry equation
  strings are per-family constants, so a sequence encoder adds nothing over multi-hot+params
  at 26 families. The contract (§2.1) is what matters; the encoder can be swapped later.

## 3. Scale & config (v3ar_r1)
- d=384, depth=8, d_w=16, dims2=64², dims3=32³, enc_n=8192 (grid) / 16384 (mesh),
  n_colloc=2048/frame, d_cond=128. Target ~100–140M params.
- Optimizer: Adam, lr 3e-4 cosine (α=0.05), warmup 2000, clipnorm 1.0, XLA on, bf16 off
  (v2 stack is fp32; revisit), batch 2/replica × 2 GPU (MirroredStrategy, ReductionToOneDevice).
- Interleave: infinite balanced per-family streams (never-drop, v2 fix). Buckets by (K, steady, T).
- Ckpt: name-keyed npz (`core/pretrain.py` toolkit), ckpt_best on class-avg, keep 3.
- From scratch (re-architecture; big-box warm-start was proven to fail in v2).
- Save: `/corpus/results/v3ar_r1`. Launch: capella, `resourcespec-ch100x2`,
  image tensorflow/tensorflow:2.15.0-gpu, /corpus + /code-vol mounts, gated by $0.30 CPU check.

## 4. Success bars
- Class-avg < 18.09% (v2 wall) within 40k steps; then push toward per-family bars in
  BASELINE_TARGETS.md.
- Mesh families must move: drivaernet < 28→ toward GINO 16 / ME-GNN 14.16; airfrans < 35.7.
- Turbulence families must show the AR effect seen in bylfa: pdearena_ns/NS-Sines/Wave-Layer
  down from 32/47/44.
- 3D-CNS M1.0 stays ahead of DPOT 22.6.

## 5. Risks
- Full-step AR cost: ~K× encoder+core per view. Mitigation: 1-future segment is tiny
  (N_p=8, modes=4), nodes subsampled, XLA graph reuse across segments (fixed shapes per bucket).
- AR divergence early in training (more re-anchor boundaries): warmup + clamp + (if NaN)
  scheduled-sampling ramp (start K=2, grow to full K) — flag `ar_ramp`.
- Node-set feedback undersamples fine mesh structure: mitigate with enc_n 16384 on mesh +
  near-wall biased node sampling (v2 lever, kept).
