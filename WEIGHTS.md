# MOTION pretrained weights

Weights are hosted on Zenodo (CC-BY-4.0): **DOI: [10.5281/zenodo.22104636](https://doi.org/10.5281/zenodo.22104636)**
(concept DOI; the version for the revised manuscript adds MOTION-NS and MOTION-IVP).
Extract each archive into `checkpoints/` at the repository root.

| Model | File | Parameters | Pretraining | Test error | Used in |
|---|---|---|---|---|---|
| MOTION-S | `motion_s_ckpt_best.npz` | 20.7M | 19 families, two stages (110,000 + 20,000 updates) | 8.25% (19-family class average, relative L2) | Figs 1–3 |
| MOTION-L | `motion_l_ckpt_best.npz` | 156.9M | 19 families, two stages (106,250 + 40,000 updates) | 6.99% (19-family class average) | Supplementary Notes 1, 4 |
| MOTION-NS | `motion_ns_ckpt_160000.npz` | 21.8M | 6 Navier–Stokes-type families, 160,000 updates | 2.84% (6-family mean) | Fig. 4a,d |
| MOTION-IVP | `motion_ivp_ckpt_best.npz` + `motion_ivp_global_norm.json` | 25.6M | 6 Poseidon pretraining families, three stages | starting point of the fifteen downstream tasks | Fig. 4b,d |

MOTION-S and MOTION-L are the checkpoints released with the submitted version (there named
`MOTION_weights_motion_s_20.7m` and `MOTION_weights_motion_156.9m`); the files are unchanged.

SHA-256:

```
c78f2e7929e02405b4be8f8f8fffbaed6921267044d8e9c65a2f802c908a30fe  motion_s_ckpt_best.npz
6fc2701e647b3fe2fea1db6818158406e64e5420ffea643ac4f14c16bdbbef99  motion_l_ckpt_best.npz
96ec3134829ae685709ac645f67ab1f94f2d184cf7a6eb42b9587d7b7a2a0734  motion_ns_ckpt_160000.npz
3a24b27b2f25fedd0d1f0c6e02bf528f8f96594823b4d3b3e130263faf43dd3d  motion_ivp_ckpt_best.npz
130be1be3344f3dca80111310f17efdf1dad7f6661f5ce54f097e2e0718fac07  motion_ivp_global_norm.json
```

## Settings needed to load each checkpoint

The checkpoints are TensorFlow 2.15 named-array `.npz` files. Evaluation always uses `NO_DEAD_REPAIR=1`.

| Model | Environment | Architecture options |
|---|---|---|
| MOTION-S | `MOTION_LEGACY_VOCAB=1 MOTION_VORTEX2D_LEGACY=1` | defaults (`--d 384 --depth 8 --d_w 16 --n_p 16`) |
| MOTION-L | `MOTION_LEGACY_VOCAB=1 MOTION_VORTEX2D_LEGACY=1` | `--d 896 --depth 12 --d_w 48` |
| MOTION-NS | `MOTION_WARP_LIVE=1 MOTION_BANK_NGENERIC=8 MOTION_BANK_GENERIC_MERGE=1 MOTION_SPLIT_MTF=1 MOTION_INPUT_ZSCORE=1 MOTION_COND_STATS=1 MOTION_BOX_UNIFORM=64` | defaults |
| MOTION-IVP | `MOTION_BANK_NGENERIC=8 MOTION_BANK_GENERIC_MERGE=1 MOTION_IC_MAXFRAC=0.95 MOTION_FT_NOMETA=1 MOTION_FORCE_MODE=ss MOTION_SS_WARP=blend MOTION_WARP_LIVE=1 MOTION_INPUT_ZSCORE=1 MOTION_GLOBAL_NORM=1 MOTION_EVAL_L1=1 MOTION_BOX_UNIFORM=64` | `--n_p 32` |

In PyTorch the same settings are the presets `motion_s`, `motion`, `motion_ns` and `motion_ivp_s` of
`motion_torch/config.py`:

```python
from motion_torch import PRESETS, build_model
model = build_model(PRESETS["motion_ns"], "checkpoints/motion_ns_ckpt_160000.npz", device="cuda")
```

Score each checkpoint in the precision it was trained in (MOTION-S float32; MOTION-L, MOTION-NS and MOTION-IVP
bf16 mixed precision), see the reproduction notes in `README.md`.
