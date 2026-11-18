# MOTION pretrained weights

Weights are hosted on Zenodo (CC-BY-4.0), version 2.0.0:
**DOI: [10.5281/zenodo.24334472](https://doi.org/10.5281/zenodo.24334472)**
(all versions: [10.5281/zenodo.22104635](https://doi.org/10.5281/zenodo.22104635)).
Extract the two archives and place all checkpoints in `checkpoints/` at the repository root.

| Model | TensorFlow checkpoint | PyTorch file | Parameters | Pretraining | Test error | Used in |
|---|---|---|---|---|---|---|
| MOTION-S | `MOTION_weights_motion_s_20.7m.tar` (contains `motion_s_20.7m/ckpt_best.npz`) | `motion_s.pt` | 20.7M | 19 families, two stages (110,000 + 20,000 updates) | 8.25% (19-family class average, relative L2) | Figs 1–3 |
| MOTION-L | `MOTION_weights_motion_156.9m.tar` (contains `motion_156.9m/ckpt_best.npz`) | `motion_l.pt` | 156.9M | 19 families, two stages (106,250 + 36,000 updates) | 6.99% (19-family class average) | Supplementary Notes 1, 4 |
| MOTION-NS | `motion_ns_ckpt_160000.npz` | `motion_ns.pt` | 21.8M | 6 Navier–Stokes-type families, 160,000 updates | 2.84% (6-family mean) | Fig. 4a,d |
| MOTION-IVP | `motion_ivp_ckpt_best.npz` + `motion_ivp_global_norm.json` | `motion_ivp.pt` | 25.6M | 6 Poseidon pretraining families, three stages | starting point of the fifteen downstream tasks | Fig. 4b,d |

The two archives of MOTION-S and MOTION-L are unchanged from version 1.0.0 (the submitted manuscript). The `.pt`
files hold the same weights with their configuration, written by `python -m motion_torch.convert`. The two models of
version 1.0.0 that the revised manuscript no longer uses remain available in that version
([10.5281/zenodo.22104636](https://doi.org/10.5281/zenodo.22104636)).

SHA-256:

```
b9c733042d60b795a17aad2adf4463aa8f37f5e5d728f37f637c10fed63d87fa  MOTION_weights_motion_s_20.7m.tar
5c58d7130db81ffa41d546f2580a1b286085506eedc9bcae193f98036c41f4b1  MOTION_weights_motion_156.9m.tar
c78f2e7929e02405b4be8f8f8fffbaed6921267044d8e9c65a2f802c908a30fe  motion_s_20.7m/ckpt_best.npz
6fc2701e647b3fe2fea1db6818158406e64e5420ffea643ac4f14c16bdbbef99  motion_156.9m/ckpt_best.npz
96ec3134829ae685709ac645f67ab1f94f2d184cf7a6eb42b9587d7b7a2a0734  motion_ns_ckpt_160000.npz
3a24b27b2f25fedd0d1f0c6e02bf528f8f96594823b4d3b3e130263faf43dd3d  motion_ivp_ckpt_best.npz
130be1be3344f3dca80111310f17efdf1dad7f6661f5ce54f097e2e0718fac07  motion_ivp_global_norm.json
740098378678209fd758828162655abc4888474c8ccca5c9e9ddd7a2b5044daf  motion_s.pt
6f35afa072b097b54905d6caaf6c1be46906419bd8f58b06e4bdc9cb4835bd59  motion_l.pt
9ecb28aec4cdfadb8183686a6f152b258fbb830e9d7e2a39c0175d8ce9664aca  motion_ns.pt
e24dd21a42684cfb10da54e43a5f62147f4d3a1a1b2e8323427fd4ddd2bbcc36  motion_ivp.pt
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

or, from a converted file, which carries its configuration:

```python
model = build_model(None, "checkpoints/motion_ns.pt", device="cuda")
```

Score each checkpoint in the precision it was trained in (MOTION-S and MOTION-L float32; MOTION-NS and MOTION-IVP
bf16 mixed precision), see the reproduction notes in `README.md`.
