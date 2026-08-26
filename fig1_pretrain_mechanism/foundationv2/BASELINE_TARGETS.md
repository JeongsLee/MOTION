# foundationv2 — baseline accuracy targets (what we must beat / match)

Compiled 2026-07-14 from the manuscript, phase1task1_v2, phase2, and the published PDE-foundation
literature. Metric conventions differ per benchmark — each row states its own. "rel-L2" = relative
L2; "rel-L1@T" = Poseidon's median relative-L1 at final time. Lower is better.

## 1. Benchmark-I — joint 6-family fluids, matched budget (1M views), class-avg rel-L2 (frames 10–19)
The head-to-head that defines our in-distribution bar. Source: phase1task1_v2/RESULTS_efficacy.md,
manuscript §4.4. Same optimizer/budget/test/metric for all.

| model | class-avg | SWE | com_ns | incom_ns | pdearena_ns | cfdbench | uncond |
|---|---:|---:|---:|---:|---:|---:|---:|
| PROSE-FD | 6.07 | 0.30 | 1.84 | 9.34 | 12.53 | 0.65 | 11.75 |
| BCAT | 5.59 | 0.42 | 2.16 | 8.29 | **11.68** | 1.18 | 9.80 |
| **ADA-combo (ours, single-shot)** | **4.78** | 0.12 | 2.22 | 1.86 | 13.65 | 0.11 | 10.71 |
| **ADA-combo-rec (ours, recursive)** | **4.14** | — | 1.94 | 1.84 | 12.39 | — | **8.47** |
| advb2 (ours, latest, task2 chain) | 3.60 @160k | 0.08 | 1.93 | 1.54 | 10.76 | 0.10 | 7.18 |

**Targets for foundationv2 (universal, single weights):**
- MUST beat PROSE-FD (6.07) and BCAT (5.59) class-avg → **< 5.5**, ideally match ADA-combo **~4.8**.
- STRETCH: match advb2 **~3.6** with the universal model (harder — one model over 20 families vs a
  fluid-specialized 6-family model). Turbulent pdearena is the known laggard (BCAT's AR wins there).
- Recursion arm target: sub-**4.2** class-avg, uncond < BCAT's 9.80.

## 2. Benchmark-II — Poseidon few-shot transfer, median rel-L1 @ final time, N=128 (unless noted)
Downstream held-out tasks. Source: phase2/STATUS.md, REVIEW_n64, Poseidon paper (arXiv 2405.19101).
Poseidon-B = 158M, Poseidon-L = 629M, scOT ~40M. These are the finetune-transfer bars.

| task | Poseidon-B | Poseidon-L | ours (phase2) | foundationv2 target |
|---|---:|---:|---:|---:|
| NS-PwC (in-family) | 6.53 | ~4–5 | **4.41** (win) | ≤ Poseidon-B; hold our 4.4 |
| ACE (OOD reaction) | 1.30 | ~1.0 | 2.02 (+reaction bank) | approach 1.3 |
| Wave-Layer (OOD hyperbolic) | **20.28** | ~15 | 34.75 (loss) | close gap: < 25 |
| Poisson-Gauss (steady elliptic) | (Poseidon strong) | — | strongest OOD win | keep the win (steady head) |
| Gauss / SE-AF / Helmholtz | Poseidon baselines | — | — | match within ~1.5× |

**Targets:** win or match Poseidon-B on in-family (NS-PwC) and steady (Poisson); CLOSE the OOD
hyperbolic (Wave-Layer) gap that the specialized fluid banks caused — the combo (non-adv) banks +
few-shot are meant to help here. Beating Poseidon-L (629M) is the stretch flagship goal at ~600M.

## 3. Mesh / geometry benchmarks (finetune downstream; new for foundationv2)
Metrics are each dataset's standard (rel-L2 on surface/volume fields, or drag Cd error).

| dataset | reference model | reference score | foundationv2 target |
|---|---|---|---|
| AirfRANS (volume u,p) | GraphSAGE/GNO (NeurIPS'22), Transolver | Transolver ~SOTA | within ~2× of Transolver at first; competitive after finetune |
| ShapeNet-Car (surface p) | GINO | test rel-L2 **9.47%** | ≤ ~12% (GINO-class), stretch ≤ 9.5 |
| Geo-FNO (airfoil/elasticity/pipe) | Geo-FNO | ~1–2% rel-L2 per task | within ~2× per task |
| DrivAerNet++ (surface p, Cd) | AB-UPT / RegDGCNN | AB-UPT ~SOTA | foundation-vs-specialist: near-AB-UPT = a win |

**Framing:** these are SPECIALIST-model bars. As a *foundation* model finetuned per task, matching
or approaching a dedicated model (GINO/Transolver/AB-UPT) IS the headline claim, even without
beating it. GINO's 9.47% on ShapeNet-Car is the concrete number to chase first.

## 3b. 3D volumetric (PDEBench 3D compressible NS, 128³) — literature bars
Sources: PDEBench (NeurIPS'22), DPOT (arXiv 2403.03542). Metric = nRMSE / relative-L2 (per source).

| model | 3D CNS result | note |
|---|---|---|
| PDEBench FNO/U-Net baselines | HIGH nRMSE (CNS is the hardest PDEBench family; 3D inviscid slightly easier than 2D due to smooth low-res) | no single clean % published; CNS ≫ Burgers error |
| **DPOT (foundation, finetuned on 3D NS)** | **~41% → 22.6% rel-L2** | the concrete foundation-model 3D bar to chase |

**foundationv2 3D target**: match/approach DPOT's finetuned **~22.6%** rel-L2 on 3D CNS. Our run13
@12.5k (normalized-space, not converged, no finetune) already shows 3D-CNS M0.1 15.5 / M1.0 28.9 /
Turb 32.2 — in the right ballpark before finetune; converged + finetuned should reach ~22% region.
The differentiator: same weights do 3D AND 2D AND mesh, whereas DPOT finetunes a 2D-pretrained model.

## 3c. Mesh — concrete SOTA numbers (surface pressure rel-L2)
Sources: Transolver (AAAI), RETO, ME-GNN, GINO, GTF-Net (2024–2026).

| dataset | GINO | Transolver | current SOTA | foundationv2 target |
|---|---:|---:|---:|---|
| **ShapeNet-Car** (surface p) | 9.47% | **7.5%** | RETO 6.3% / ME-GNN **5.56%** | ≤ ~9.5% (GINO-class) first; stretch ≤ 7.5% (Transolver) |
| **DrivAerNet++** (surface p) | 15.7% | — | GTF-Net 14.5% / ME-GNN **14.16%** | ≤ ~16% (GINO-class) |
| **AirfRANS** (lift/fields) | — | Transolver strong; LinearNO beats it >60% on lift | — | within ~2× of Transolver, then finetune-competitive |

**Framing (unchanged):** these are SPECIALIST models. As a finetuned *foundation* model, approaching
Transolver/GINO (ShapeNet-Car 7.5–9.5%) IS the headline. ME-GNN 5.56% is the current ceiling to note.

## 3d. MEASURED physical rel-L2 (run13 ckpt_8000, mid-training d384, NO finetune) — 2026-07-14
First physical-scale eval (core/eval_physical.py; predictions denormalized ×per-channel std).

| family | PHYSICAL | normalized | vs literature target |
|---|---:|---:|---|
| **pdebench3d_cns M0.1** | **6.40%** | 11.33 | **beats DPOT finetuned 22.6%** (mid-train, no finetune!) |
| pdebench3d_cns M1.0 | 27.34 | 28.71 | ~ DPOT 22.6 |
| pdebench3d_cns Turb | 50.00 | 33.45 | hard (turbulence, zero-mean -> physical worse) |
| geofno_airfoil | 20.02 | 20.02 | single-channel -> phys=norm; far from Geo-FNO ~1-2% (mid-train) |
| geofno_elasticity | 31.70 | 31.70 | single-channel; mid-train |
| airfrans | 128.5 | 80.5 | BROKEN on pressure (std 198 dominates physical); needs per-channel loss weighting |

Takeaways: (1) 3D CNS is the strongest card — M0.1 physical 6.4% already beats the DPOT 3D bar
before convergence/finetune (physical metric rewards the large-DC density/pressure). (2) airfrans
pressure genuinely diverges — physical exposes what normalized hid; fix = per-channel loss
weighting + separate pressure normalization. (3) single-channel mesh (airfoil/elasticity) phys=norm
(rel-L2 scale-invariant), so 20%/32% are true and far from Geo-FNO specialist (~1-2%) — capacity +
convergence needed. Re-run at d512 convergence + finetune for the official comparison.

## 4. Parameter-scale reference (context for the accuracy bars)
Poseidon T/B/L = 21M/158M/629M · scOT 40M · DPOT-S/M = 30M/122M · our advb2 = 152M · Transolver
= millions-scale (not a foundation model). foundationv2: bring-up ~150M (Poseidon-B parity),
flagship ~600M (Poseidon-L parity). Memory binds on activations, not params.

## 5. The single sentence
**Beat PROSE-FD/BCAT class-avg (→ ~4.8, stretch ~3.6) with ONE weight set; match Poseidon-B on
finetune transfer (win in-family + steady, close the hyperbolic gap); approach GINO/Transolver/
AB-UPT on mesh after finetune — the value being 2D+3D+mesh coverage in a single foundation model,
not per-benchmark SOTA.**
