# motion_torch — PyTorch port of MOTION (weight loading, inference, fine-tuning)

PyTorch re-implementation of the MOTION PDE model of `../motion_tf` (TensorFlow, `v3/model.py V3Model`) for the code
paths used by the released checkpoints. It loads the TF weights-only `.npz` checkpoints directly, runs autoregressive,
single-shot and receding-horizon inference, and fine-tunes with the TF training objective. **Pretraining stays in
TensorFlow.** The package does not import TensorFlow (`tests/test_torch_smoke.py` asserts it).

Tested with torch 2.0.1 (CUDA 11.7) and 2.5.1, numpy 1.26, Python 3.10. Parity against TF 2.15 for MOTION-S,
MOTION-L, MOTION-NS and MOTION-IVP (`tests/parity_*.py`). `fast.py` provides an accelerated execution path with the
same parameters (fused axial convolutions and attention projections, spectral filter as discrete Fourier transform
matrix products, `torch.compile`); `bench_ivp.py` and `bench_fast.py` time it against the reference path.

## Layout

| file | content | TF reference |
|---|---|---|
| `config.py` | `MotionConfig` (architecture + every behaviour-changing env switch), `PRESETS` | env reads in v3/, core/, data/ |
| `model/layers.py` | Dense, RankFreeLN, AxialConv, AxialAttention (Keras MHA), PointwiseMLP, CoordEncoder | core/axops.py, core/pointio.py |
| `model/encdec.py` | BinEncoder (cell scatter-mean), CondEncoder (+FiLM), ContinuousDecoder (fam_head) | core/binenc.py, v3/cond.py, core/pointio.py |
| `model/banks.py` | TendencyBanks: 19 typed heads, H0, gate_cond, op masks, generic heads (+merge), knock-out mask | core/banks.py |
| `model/v3model.py` | V3Model: `trunk`, `step` (AR + transport warp), `step_ss` (panels, `leads`, MOTION_SS_WARP), `interp_box` | v3/model.py |
| `convert.py` | TF npz -> torch (`load_tf_npz`, `build_model`, CLI) with the `load_ckpt` rules | core/pretrain.load_ckpt |
| `data/` | registry (verbatim), symbolic, `make_example_ar`, `stack_batch`, cond, policy, global norm, LOCAL_UNITS | data/, v3/data_ar.py, v3/cond.py, v3/policy.py |
| `inference.py` | `rollout_ar`, `singleshot`, `rollout_rh`, `predict`, `rel` metrics, `trajectory_error` | eval_stats.py, v3/train.py |
| `finetune.py` | minimal fine-tuning script (rollout / single-shot / rh losses) | v3/train.py |
| `tests/` | `parity_motion_s.py`, `parity_motion_ns.py`, `parity_motion.py` (TF vs torch), `tf_reference.py` (TF side), `test_torch_smoke.py` | |

Parameters are kept in the Keras layout (Dense kernel `(in, out)`, Conv1D `(k, in, out)`, MHA `(d, heads, key_dim)`)
and every parameter records its TF variable name (`model.tf_name_map()`), so conversion is a name-keyed copy.
Tensors are channels-last `(B, *dims, C)` as in TF.

## Usage

```bash
# convert once (optional: build_model() also reads the .npz directly)
python -m motion_torch.convert --ckpt checkpoints/m1f_ckpt_best.npz --preset motion_s --out motion_s.pt
python -m motion_torch.convert --ckpt checkpoints/zsc_ckpt_160000.npz --preset motion_ns --out motion_ns.pt
python -m motion_torch.convert --ckpt checkpoints/m2s2_ckpt_best.npz --preset motion --out motion.pt
```

```python
import numpy as np, torch
from motion_torch import PRESETS, build_model
from motion_torch.data.example import load_local_units, make_example_ar, stack_batch, box_dims_for
from motion_torch.inference import to_torch, predict, trajectory_error, to_physical

torch.backends.cuda.matmul.allow_tf32 = False          # float32 parity with TF (optional for plain use)
cfg = PRESETS["motion_s"]
model = build_model(cfg, "m1f_ckpt_best.npz", device="cuda")          # or "motion_s.pt"
unit = load_local_units("units_nss_mtf.pkl")["incom_ns"][0]            # any unified-corpus sample dict
ex = make_example_ar(unit, 10, 10, 4, 2048, 8192, np.random.default_rng(0), cfg,
                     box_dims=box_dims_for("incom_ns", cfg))
b = to_torch(stack_batch([ex]), "cuda")
out = predict(model, b)            # 'ar' -> 10-step AR rollout, 'ss' -> single shot, 'rh' -> hop-1 rollout
out["pred"]                        # (B, 10, N_nodes + Q, 16) normalized-space prediction (nodes first)
trajectory_error(out)              # per-trajectory physical rel-L2 (PROSE/BCAT joint convention)
to_physical(out["pred"], b, cfg)   # physical units
```

Lower-level calls mirror the TF API: `model.step(coords_node, win, fmask, geom_node, K, roles, cond, op_multihot,
coords_q, geom_q, u_prev_q, steady=, prev_grid=, grid_dims=, fam_id=, sdf_box=, box_dims=, fbmask=)` and
`model.step_ss(..., kf=, leads=, prev_grid=, grid_dims=)`. Eval-only switches of eval_stats.py are available as
`model.banks.ko_index / ko_mask` (head knock-out, order `model.banks.all_heads()`), `model.banks.active`,
`model.warp_scale` (warp off) and `model.z_scale`.

Fine-tuning (rel-L2 per channel by default, `--l1 1` for rel-L1; Adam/AdamW; cosine with warmup; clipnorm per tensor
as Keras or `--clip_mode global`):

```bash
python -m motion_torch.finetune --ckpt m1f_ckpt_best.npz --preset motion_s \
    --units checkpoints/units_nss_mtf.pkl --families incom_ns --steps 200 --lr 1e-4 --save ft.pt
```

Parity tests (need the TF 2.15 env for the reference side; it is called as a subprocess, results cached in
`~/.cache/motion_torch_parity`, summaries written to `tests/parity_<tag>.json`):

```bash
python tests/parity_motion_s.py        # MOTION-S, synthetic + 32 real units
python tests/parity_motion_ns.py       # MOTION-NS (+ IVP-S code paths with --sswarp 1)
python tests/parity_motion.py          # MOTION (156.9M)
python tests/test_torch_smoke.py       # port only, no TF, no checkpoint
```
Environment overrides: `TF_PYTHON`, `MOTION_FV2` (reference tree), `LOCAL_UNITS`, `PARITY_CACHE`.

## Released models: environment flags -> config

| model | checkpoint | TF environment / args | preset |
|---|---|---|---|
| MOTION-S (paper m1f, 20.7M) | `checkpoints/m1f_ckpt_best.npz` | `MOTION_LEGACY_VOCAB=1 MOTION_VORTEX2D_LEGACY=1 NO_DEAD_REPAIR=1`; d 384, depth 8, d_w 16, n_p 16, t_in 10; per-family box (BOX_TABLE) | `motion_s` |
| MOTION (m2s2, 156.9M) | `checkpoints/m2s2_ckpt_best.npz` | as MOTION-S, `--d 896 --depth 12 --d_w 48` | `motion` |
| MOTION-NS (zsc, 21.8M) | `checkpoints/zsc_ckpt_160000.npz` | `MOTION_WARP_LIVE=1 MOTION_BANK_NGENERIC=8 MOTION_BANK_GENERIC_MERGE=1 MOTION_BOX_UNIFORM=64 MOTION_INPUT_ZSCORE=1 MOTION_COND_STATS=1` (`MOTION_SPLIT_MTF=1` = data split only) | `motion_ns` |
| MOTION-IVP-S (not local) | — | `MOTION_FORCE_MODE=ss MOTION_SS_WARP=blend MOTION_WARP_LIVE=1 MOTION_INPUT_ZSCORE=1 MOTION_GLOBAL_NORM=1 MOTION_BANK_NGENERIC=8 MOTION_BANK_GENERIC_MERGE=1 MOTION_FT_NOMETA=1`, `--n_p 32 --t_in 1 --k_fut 20 --box_table 0 --dims2 64` | `motion_ivp_s` |

What each switch does in the port (`MotionConfig` field):

| env | field | effect |
|---|---|---|
| MOTION_LEGACY_VOCAB | `legacy_vocab` | 13-token operator vocabulary (no `phase_change`): cond width 146 instead of 147, phase_interface head absent. Stored 14-token multi-hots are cut (`fit_multihot`) |
| MOTION_VORTEX2D_LEGACY | `vortex2d_legacy` | vortex head (its bias) also evaluated for K = 2 |
| MOTION_WARP_LIVE | `warp_live` | init only (random `disp`); no effect on a loaded checkpoint |
| MOTION_BANK_NGENERIC / _MERGE | `bank_ngeneric`, `bank_generic_merge` | G generic heads (stencil x low-rank + anisotropic spectral), merged behind one gate |
| MOTION_BOX_UNIFORM | `box_uniform` | uniform latent box side (`box_dims_for`), else BOX_TABLE_2D/3D |
| MOTION_INPUT_ZSCORE | `input_zscore` | instance z-score (mean in `shift`, added back by the physical metric) |
| MOTION_COND_STATS | `cond_stats` | cond += [sign(mu) log1p|mu|, log1p sigma] per slot (+32) |
| MOTION_GLOBAL_NORM | `global_norm` | fixed per-family mean/std: pass `global_norm={fam: compute_global_norm(train_units)[:2]}` to `make_example_ar` |
| MOTION_FORCE_MODE | `force_mode` | temporal mode of every transient family (`ss` = single shot) |
| MOTION_SS_WARP | `ss_warp` | `blend`: AR carrier (`disp`, `tgate`) in single-shot, frame k warped by k*disp; `1`: Taylor warp |
| MOTION_RH_FAMS | `rh_fams` | receding-horizon mode (`rollout_rh`) |
| MOTION_FT_NOMETA | `ft_nometa` | metadata-free examples (all op tokens on, params removed) |
| MOTION_TI_FIXED | `ti_fixed` | observed history length at a t_in-10 model |
| NO_DEAD_REPAIR | `no_dead_repair` | converter keeps all-zero banks tensors exactly (else: fresh init, as load_ckpt) |

`MotionConfig.from_env()` reproduces the TF behaviour for the current environment; `cfg.env()` gives the TF
environment of a config. Not ported (asserted off in `from_env`): MOTION_CORE=cnn|fno, MOTION_STD_DECODER,
MOTION_BANK_GENERIC / NOMETA / NOH0 / SINGLE / PHYSGATE, MOTION_WARP_MODE=advbank|physvel, slice attention /
surface path, dictionary decode, d_shape, MoE, MOTION_META_MODE. MOTION_FAST_ATTN is an exact re-association; the
port implements the default path.

## Parity (TF 2.15 eager vs torch 2.0.1, float32, TF32 off, RTX 4070 Ti)

Same numpy example in both (the port's `make_example_ar` output is **bit-identical** to the reference in every case).
Numbers are max |torch - TF| / max |TF| over all nodes + collocation points and slots ("rel"). Synthetic = random
fields (B = 2, 64^2 grids / 16^3 for 3-D, as tests/test_v3_smoke.py); real = held-out units of
`units_nss_mtf.pkl` (MTF test split), 10-step AR rollout, B = 1, dense 128^2.

| model | case | one step | 10-step rollout | per-trajectory phys rel-L2 TF / torch (max abs diff) |
|---|---|---|---|---|
| MOTION-S | real incom_ns (8) | 1.3e-7 | 3.2e-7 | 2.7037% / 2.7037% (5.6e-9) |
| | real com_ns (8) | 1.8e-7 | 4.6e-7 | 4.0639% / 4.0639% (3.7e-9) |
| | real shallow_water (8) | 7.2e-7 | 6.7e-6 | 0.3551% / 0.3551% (2.1e-9) |
| | real pdearena_ns (8, box 128^2) | 2.1e-6 | 5.3e-6 | 25.0074% / 25.0074% (0) |
| | synthetic incom_ns / ACE (ss) / 3-D M0.1 | <= 7.2e-7 | <= 3.3e-6 | |
| | synthetic CE-RP | 2.3e-6 | 7.6e-5 * | |
| | synthetic Poisson-Gauss (steady) | 6.9e-6 | 8.1e-6 | |
| MOTION-NS | real incom_ns / com_ns / shallow_water / pdearena_ns / cfdbench (8 each) | <= 3.3e-6 | <= 4.6e-6 | 0.8063 / 2.1288 / 0.0521 / 15.4150 / 0.0522 % identical (<= 4.5e-8) |
| | synthetic incom_ns / CE-RP / ACE / 3-D | <= 1.3e-6 | <= 5.2e-6 | |
| | synthetic Poisson-Gauss (steady) | 3.3e-5 * | 2.5e-5 * | |
| MOTION-NS weights, IVP-S code paths (FORCE_MODE=ss, SS_WARP=blend, RH_FAMS=com_ns) | single shot, lead subsets, hop-1 rh; synthetic + 4 real | <= 7.9e-7 | <= 4.5e-6 | identical (<= 3.7e-9) |
| MOTION (156.9M) | real incom_ns / com_ns / shallow_water / pdearena_ns (4 each) | <= 4.0e-6 | <= 5.0e-6 | 3.0053 / 3.8537 / 0.2342 / 17.3322 % identical (<= 2.4e-8) |
| | synthetic incom_ns / ACE / 3-D | <= 1.6e-6 | <= 1.2e-5 * | |
| | synthetic CE-RP / Poisson-Gauss | 2.8e-6 / 1.4e-5 * | 1.2e-4 * / 1.0e-5 | |

\* Above 1e-5 only where the case itself amplifies float32 rounding. For every synthetic case the port is also run
in float64 (Fourier phases kept in float32, as TF computes them): TF-float32 and torch-float32 are equally far from
it, e.g. synthetic CE-RP after 10 steps: TF 3.6e-5 / torch 4.6e-5 (MOTION-S), 3.9e-5 / 8.0e-5 (MOTION); steady
Poisson-Gauss step 1: TF 1.6e-5 / torch 2.8e-5 (MOTION-NS). The parity criterion is "<= 1e-5, or <= 3x the
float32-vs-float64 distance"; all cases pass (numbers vary by ~10% between runs: GPU scatter-adds are
non-deterministic in both frameworks) (`tests/parity_*.json` hold every number incl. the trunk tensors
h (<= 1.8e-6) and W (<= 3.5e-6; <= 6e-5 in the steady zero-window case)).

Fine-tuning (FT parity not attempted): MOTION-S, 20 steps on incom_ns units, fixed-batch loss 0.0393 -> 0.0389;
MOTION-NS overfit check on one com_ns batch, 10 steps 0.520 -> 0.453; single-shot (ACE) 1.891 -> 1.622.

## Notes

* **cuFFT on RTX 40xx with the cu117 wheel**: `torch 2.0.1+cu117` raises `CUFFT_INTERNAL_ERROR` on sm_89 GPUs.
  The spectral bank probes the GPU FFT once and falls back to the CPU for the (8-channel) spectral transforms;
  a cu118+ wheel runs them on the GPU. Results are unaffected.
* The Fourier coordinate features are always computed in float32 (phases up to 2^15*pi*x), as in TF, also when the
  model is cast to float64/bfloat16.
* `dec_famW` of MOTION-S (26 rows) and MOTION (30 rows) is zero-padded to the current 32 families on load, exactly
  as `load_ckpt`; fam_id = index in `data.registry.FAMILIES`.
* Single-shot MOTION_SS_WARP passes the IC grid only on dense 2-D grids (TF raises for sub-sampled 3-D windows).
* MOTION-IVP-S (2026-11-09): parity-tested with the pretrained checkpoint motion_ivps_gn3w/ckpt_best (stage-3 option
  C; tests/parity_motion_ivp.py): 412/412 variables, one-step <= 2.5e-6, ten-step <= 2.0e-5 (float32 noise),
  per-trajectory error <= 2.4e-7 (synthetic CE-RP/CE-KH single-shot, NS-Sines/NS-Gauss receding horizon). Its
  dec_famW has 47 family rows (the 15 FT-* tasks appended), so the preset uses n_fams=47 and data/registry.py contains
  the FT-* families. Configuration of the FT runs: `t_in=10, ti_fixed=1`, `global_norm` from `compute_global_norm`.
* See `PORT_LOG.md` for the full log of the port and of the investigations.
