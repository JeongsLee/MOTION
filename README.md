# MOTION — Mechanism-Oriented Tendency-Integration Operator Networks

Code, trained weights and figure scripts for

> J. Lee, *Mechanism-oriented tendency-integration operator networks for PDE foundation models* (2026, under review).

MOTION is a PDE foundation model whose latent rate of change is a gated sum of mechanism heads, each applying a
prescribed differential, spectral or pointwise operator to the latent field, plus an ungated state head. The rate is
integrated in closed form and decoded so that the initial condition is satisfied exactly. Because each head has a
fixed computational role, the learned dynamics can be examined by head knockout, by commands applied to the operator
inputs, and by tests under physically equivalent transformations.

This branch holds the code of the **revised** manuscript. The state of the repository for the submitted version,
including the second architecture that the revised paper no longer uses, is kept under the tag
[`v1.0-submitted`](https://github.com/JeongsLee/MOTION/tree/v1.0-submitted).

## Layout

```
motion_tf/      TensorFlow implementation: model (core/, v3/), data (data/), training (python -m v3.train),
                evaluation (eval_stats.py, eval_contrib.py, eval_equiv.py, score_ivp_l1.py) and the
                latent-command experiments (latent_*.py)
motion_torch/   PyTorch implementation for loading the released weights, inference and fine-tuning, with a weight
                converter, parity tests against TensorFlow and an accelerated execution path (see its README)
experiments/    job scripts of every experiment as they were run, and the table that maps each result of the
                paper to its script and settings (experiments/README.md)
analysis/       metrics (D, AUC, same-form separation), statistics (bootstrap, paired tests, regime tests),
                equivalence statistics, the pseudo-spectral reference solvers and table builders
baselines/      BCAT / PROSE-FD (retraining, released weights, physical input features, symmetry evaluation)
                and Poseidon (data preparation, fine-tuning, scoring)
figures/        scripts of the main and supplementary figures, with their reference data
demo/           CPU demo of the knockout protocol (no data download, no checkpoint)
WEIGHTS.md      released checkpoints, their settings and checksums
```

## Models

| Model | Parameters | Pretraining data | Metadata | Transport map / generic tendency | Backbone | Used in |
|---|---|---|---|---|---|---|
| MOTION-S | 20.7M | 19 families | yes | inactive / no | width 384, depth 8 | Figs 1–3 |
| MOTION-L | 156.9M | 19 families | yes | inactive / no | width 896, depth 12 | Supplementary Notes 1, 4 |
| MOTION-NS | 21.8M | 6 Navier–Stokes-type families | yes | active / yes | width 384, depth 8, 64² latent grid | Fig. 4a,d |
| MOTION-IVP | 25.6M | 6 Poseidon pretraining families | no | active / yes | width 384, depth 8, 64² latent grid | Fig. 4b,d |

The matched-budget ablation variants of Fig. 3 are MOTION-S trained for 20,000 updates with one component
changed; each variant is selected by an environment switch listed in `experiments/README.md`.

## System requirements

**Software.** Linux with Python 3.11 and TensorFlow 2.15.0. Every MOTION run in the paper
used the `tensorflow/tensorflow:2.15.0-gpu` image (Python 3.11, CUDA 12.2, driver >= 535),
and the code is tested on that image. Dependencies and the exact versions the release is
tested against are pinned in [`requirements.txt`](requirements.txt): numpy 1.26.4,
h5py 3.10.0, scipy 1.17.1, netCDF4 1.7.4, pyyaml 6.0.1, matplotlib 3.8.0,
scikit-image 0.26.0. No other operating systems have been tested.

**Hardware.** No non-standard hardware is required, but the useful configurations differ:

| task | hardware | note |
|---|---|---|
| `demo/run_demo.py` | any x86-64 CPU, ~2 GB RAM | no GPU needed |
| inference / knockout from a released checkpoint | 1 GPU, >= 24 GB | 128^2-native boxes exhaust memory in eager mode; the scripts use graph mode |
| 19-family pretraining | 4x NVIDIA H100 80 GB | about 570 H100-hours for MOTION-L (156.9M) and about 154 for MOTION-S (20.7M) |

Third-party baselines (BCAT, PROSE-FD, scOT/Poseidon) are not vendored; each runs in the
environment pinned by its own repository (PyTorch 2.4.0-2.7.1, see `requirements.txt`).
The runner scripts under `baselines/` record the exact images, commands and settings
used to retrain or rescore them under the matched protocol.

The PyTorch implementation needs PyTorch 2.0 or later (the accelerated path was timed with PyTorch 2.5.1) and does
not import TensorFlow.

## Installation

```bash
git clone https://github.com/JeongsLee/MOTION.git && cd MOTION
pip install -r requirements.txt
export PYTHONPATH=$PWD/motion_tf:$PWD/analysis:$PWD:$PYTHONPATH
```

Typical install time is about 20 seconds inside the `tensorflow/tensorflow:2.15.0-gpu` image and a few minutes in
a fresh virtual environment. There is nothing to compile. The demo sets its own import path.

## Demo

The demo trains a small MOTION on simulated advection-diffusion data and then knocks out
individual mechanism heads - the protocol of Fig. 1c at a scale that runs on a laptop CPU.
It needs no benchmark download and no checkpoint.

```bash
python demo/run_demo.py                  # ~2 minutes on 3 CPU cores
DEMO_STEPS=600 python demo/run_demo.py   # longer run, ~3.5 minutes
```

The dataset (`demo/data/demo_advdiff.npz`, 1.2 MB, 16 trajectories of 12 frames on a
periodic 32x32 grid) is committed; `python demo/make_demo_data.py` regenerates it in about
a second. It solves `dc/dt + (u.grad)c = nu lap(c)` with a frozen divergence-free velocity
field, pseudo-spectrally, so the data is generated by exactly two of MOTION's mechanisms.

Expected output (3 vCPU, TensorFlow 2.15.0 CPU build; the error values move by a few
tenths of a percent with thread count and seed):

```
[demo] 12 train / 4 held-out trajectories, grid 32x32, 12 frames
[demo] persistence baseline on the held-out set: 10.22%
[demo] model: 0.31M parameters, 191 tensors
  step    1/300  train loss 0.1297  (51s)
  ...
  step  300/300  train loss 0.0559  (94s)

[demo] held-out error after 300 steps: 4.28%

=== mechanism knockout (gate set to zero at evaluation) ===
mechanism               gate |w|   error %    change
full model                     -      4.28         -
w/o advection            0.02798      4.29     +0.01
w/o diffusion            0.02907      4.24     -0.04
w/o wave                 0.00000      4.28     +0.00
w/o reaction             0.00000      4.28     +0.00
DEMO OK
```

What to read from it: the model beats the persistence baseline (10.22% -> ~4.5%) after a
two-minute fit, and the knockout handle behaves as the paper's analysis requires. `wave`
and `reaction` are closed by this family's equation metadata, so their gates stay at
exactly zero through training and zeroing them is provably a no-op - the grey cells of
Fig. 1c, and the reason a gate knockout is a clean intervention. `advection` and
`diffusion` carry nonzero gates and do feed the prediction, but the *sign and size* of a
knockout damage is a property of a converged model, not of a two-minute fit: at this scale
either sign can appear, and a run-to-run swing of a few tenths of a percent is normal. The
paper's knockout damages come from the 19-family checkpoints in [`WEIGHTS.md`](WEIGHTS.md).

Expected run time: **~110 s** total at the default 300 steps (about 50 s of that is the
TensorFlow import, model build and first graph trace), **~210 s** at 600 steps.

## Weights

See [`WEIGHTS.md`](WEIGHTS.md) for the four pretrained models, the environment each needs, and checksums.

## Data

No datasets are redistributed. The nineteen families are drawn from public benchmarks:

| source | families |
|---|---|
| [PDEBench](https://github.com/pdebench/PDEBench) | shallow water, diffusion–reaction, 2D/3D compressible Navier–Stokes, incompressible NS |
| [PDEArena](https://github.com/pdearena/pdearena) | Navier–Stokes 2D (conditioned / unconditioned) |
| [PDEgym / Poseidon](https://github.com/camlab-ethz/poseidon) | NS-PwC, ACE, Wave-Layer, Poisson-Gauss, CE-RM and the IVP pretraining corpus |
| [CFDBench](https://github.com/luo-yining/CFDBench) | channel flow — used in the 6-family joint benchmark (Fig. 2) and the equivalence tests (Fig. 4); not in the 19-family pretraining corpus |

Loaders and one-off cache builders live in `motion_tf/data/`.
They expect the raw HDF5/NetCDF files and prebuilt caches at the paths given by the
`PREBUILT_DIR`, `PROSE_DATA_DIR` and `CORPUS_ROOT` environment variables; see the
docstrings at the top of each loader.

## Training and evaluation settings of the paper

The commands below are run from `motion_tf/`. The effective batch size is `--batch` times the number of GPUs.

| Manuscript setting | MOTION-S | MOTION-L | MOTION-NS | MOTION-IVP | Ablation variants | Option |
|---|---|---|---|---|---|---|
| Width / depth | 384 / 8 | 896 / 12 | 384 / 8 | 384 / 8 | 384 / 8 | `--d`, `--depth` (`--d_w 16`; MOTION-L `--d_w 48`) |
| Temporal panels | 16 | 16 | 16 | 32 | 16 | `--n_p` |
| Latent grid | per family | per family | 64² | 64² | per family | `--box_table 1`, or `--box_table 0 --dims2 64` |
| Families | 19 | 19 | 6 | 6 | 19 | `--families` |
| GPUs × batch (effective) | 2 × 4 (8), then 1 × 4 | 4 × 2 (8) in both stages | 2 × 4 (8) | 2 × 8 (16) | 1 × 8 (8) | `--batch` |
| Updates (run length) | 134,900 + 25,150 | 106,250 + 40,000 | 160,000 | 80,000 + 100,000 + 50,000 | 20,000 | `--steps` |
| Reported checkpoint | 110,000 + 20,000 | 106,250 + 36,000 | 160,000 | best of stage 3 | 20,000 | — |
| Learning rate | 3e-4, then 5e-5 | 3e-4, then 5e-5 | 1e-4, cosine to 5% | 1e-4, 5e-5, 5e-5, cosine to 5% | 3e-4 | `--lr`, `--lr_sched cosine --lr_alpha 0.05` |
| Warm-up updates | 2,000, then 1,000 | 2,000, then 1,000 | 4,000 | 2,000 | 2,000 | `--warmup` |
| Second-stage loss | fluctuation-normalized | fluctuation-normalized | — | — | — | `--fluct_loss 1` |
| Precision | float32 | float32 | bf16 | bf16 | bf16 | `--bf16 1` |
| Observed / forecast frames | 10 / 10 | 10 / 10 | 10 / 10 | 1–10 / 20 | 10 / 10 | `--t_in`, `--k_fut` |
| Encoder / collocation points | 8,192 / 2,048 | same | same | same | same | `--enc_n`, `--n_colloc` |
| Evaluation | test split, up to 128 trajectories per family, batch 8 | same | 100 per family | official test split | 128 per family | `eval_stats.py --split test --n … --batch 8` |

Example (matched-budget reference of Fig. 3, one H100):

```bash
cd motion_tf
python -m v3.train --families motion --steps 20000 --batch 8 --batch3d 1 --d 384 --depth 8 --d_w 16 --n_p 16 \
  --box_table 1 --gate_cond 1 --fam_head 1 --dict_decode 0 --t_in 10 --k_fut 10 --k_relax 4 \
  --enc_n 8192 --n_colloc 2048 --reuse3d 8 --xla 1 --lr 3e-4 --warmup 2000 --fluct_loss 0 --bf16 1 \
  --save results/ablation_reference
NO_DEAD_REPAIR=1 python eval_stats.py --ckpt results/ablation_reference/ckpt_20000.npz --split test --n 128 \
  --batch 8 --batch3d 2 --xla 1 --out results/ablation_reference/stats.npz
```

The exact command, environment and job script behind every figure and table are listed in
[`experiments/README.md`](experiments/README.md).

## Compatibility switches

Some behaviour changed between the submitted checkpoints and the current code. The switches below reproduce each
released checkpoint exactly; they are read from the environment.

| Switch | Meaning | Needed for |
|---|---|---|
| `MOTION_LEGACY_VOCAB=1` | 13-token operator vocabulary of the MOTION-S and MOTION-L checkpoints (the current vocabulary has a fourteenth token, `phase_change`). Without it the conditioning vector of these checkpoints is misaligned. | MOTION-S, MOTION-L |
| `MOTION_VORTEX2D_LEGACY=1` | Evaluates the vortex-stretching head in two dimensions as these checkpoints were trained. In the current code the head is evaluated in three dimensions only, so its two-dimensional contribution is exactly zero. | MOTION-S, MOTION-L |
| `NO_DEAD_REPAIR=1` | Restores a checkpoint exactly. Without it, head weights that are identically zero are re-initialized on loading, which is intended for training only. The evaluation job scripts set it. | every evaluation |
| `MOTION_WARP_LIVE=1`, `MOTION_BANK_NGENERIC=8`, `MOTION_BANK_GENERIC_MERGE=1` | Active Lagrangian transport map and generic tendency. | MOTION-NS, MOTION-IVP |
| `MOTION_INPUT_ZSCORE=1`, `MOTION_COND_STATS=1` | Per-window mean–standard-deviation normalization with the window statistics appended to the conditioning. | MOTION-NS |
| `MOTION_FT_NOMETA=1`, `MOTION_GLOBAL_NORM=1`, `MOTION_FORCE_MODE=ss`, `MOTION_SS_WARP=blend` | No equation metadata, per-dataset normalization, panel-wise forecast of MOTION-IVP. | MOTION-IVP |

**Interface-sensitive head.** The vocabulary in `motion_tf/core/banks.py` contains one more head,
`phase_interface` (operator |∇h| ∇²h), kept from development for equations with interface dynamics. It is opened
only by the `phase_change` token, which no family of the present corpus carries: it is not instantiated in the
MOTION-S and MOTION-L checkpoints (13-token vocabulary) and is instantiated but closed in every family in the
ablation variants. It can be enabled by adding the `phase_change` operator to a family in `motion_tf/data/registry.py`.

## Reproduction notes

1. **Numeric mode is part of the learned function.** Score each checkpoint in the precision it was trained in.
2. The forward pass at 128² needs a GPU and graph mode (`--xla 1`), as the scripts use.
3. Checkpoints are TensorFlow 2.15 named-array `.npz` files; `motion_torch` reads the same files.
4. The job scripts under `experiments/` and `baselines/` are records of the runs. They keep the directory layout
   of the cluster on which they ran (`/corpus`, `/eu`, `/work`); substitute your own paths.
5. Latent-command experiments (Fig. 2) were run on one workstation GPU from the launchers in
   `experiments/latent_control/`.
6. Several supplementary tables were assembled by hand from the outputs of the listed scripts;
   `experiments/README.md` states which.

## License

Code: [MIT](LICENSE). Weights: CC-BY-4.0 (see the Zenodo record). Third-party baselines and datasets remain under
their own licenses.

## Citation

```bibtex
@article{lee2026motion,
  title   = {Mechanism-oriented tendency-integration operator networks for PDE foundation models},
  author  = {Lee, Jeongsu},
  year    = {2026},
  note    = {under review}
}
```
