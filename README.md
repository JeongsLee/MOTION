# MOTION — Mechanism-Oriented Tendency-Integration Operator Networks

Code, trained weights and figure data for

> J. Lee, *Mechanism-oriented tendency-integration operator networks for PDE foundation models* (2026, submitted).

MOTION is a PDE foundation model whose per-step tendency is a gated sum of explicit
physical-mechanism heads (transport, diffusion, pressure/density coupling, reaction,
wave, curvature, …) plus an ungated state head. Because each mechanism has a fixed
computational role, the learned dynamics can be interrogated by mechanism knockout and
by physical-equivalence tests. A single set of weights covers nineteen equation
families spanning nine governing systems in two and three dimensions.

## Layout

```
fig1_pretrain_mechanism/   19-family 2D+3D pretraining, mechanism knockout / contribution analysis (paper Fig. 1)
fig2_joint_benchmark/      6-family joint benchmark against BCAT / PROSE-FD at matched budget (paper Fig. 2)
fig3_ivp_transfer/         IVP-mode pretraining on the Poseidon corpus + few-shot transfer (paper Fig. 3)
fig4_physical_equivalence/ physical-equivalence (ε_equiv) tests and equivalence-ensemble averaging (paper Fig. 4)
common/motion_tf/          shared TensorFlow package used by Fig. 2 and Fig. 3
common/figscripts/         manuscript panel renderers and their panel data
refdata/                   benchmark curves and Fig. 1 panel inputs
WEIGHTS.md                 checkpoint table and Zenodo DOI
```

Figure-script file names run one number ahead of the manuscript (`render_fig3*` → paper
Fig. 2/3, `render_fig4*` → paper Fig. 4).

## Environment

All MOTION runs used the `tensorflow/tensorflow:2.15.0-gpu` image (Python 3.11).

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD/common:$PWD/fig1_pretrain_mechanism/foundationv2:$PYTHONPATH
```

Baselines (BCAT, PROSE-FD, scOT/Poseidon, Walrus) are third-party; their code and
weights are not vendored. The runner scripts under `*/jobscripts/` record the exact
commands and settings used to retrain or rescore them under the fair protocol.

## Weights

Trained checkpoints (5.4 GB total) are on Zenodo — see [`WEIGHTS.md`](WEIGHTS.md) for
the table, DOI and SHA-256 sums. Extract each archive into `weights/`.

## Data

No datasets are redistributed. The nineteen families are drawn from public benchmarks:

| source | families |
|---|---|
| [PDEBench](https://github.com/pdebench/PDEBench) | shallow water, diffusion–reaction, 2D/3D compressible Navier–Stokes, incompressible NS |
| [PDEArena](https://github.com/pdearena/pdearena) | Navier–Stokes 2D (conditioned / unconditioned) |
| [PDEgym / Poseidon](https://github.com/camlab-ethz/poseidon) | NS-PwC, ACE, Wave-Layer, Poisson-Gauss, CE-RM and the IVP pretraining corpus |
| [CFDBench](https://github.com/luo-yining/CFDBench) | channel flow — used in the 6-family joint benchmark (Fig. 2) and the equivalence tests (Fig. 4); not in the 19-family pretraining corpus |

Loaders and one-off cache builders live in `common/motion_tf/data/` (`prep_multi.py`,
`download_prose.py`, `poseidon.py`) and `fig1_pretrain_mechanism/foundationv2/data/`.
They expect the raw HDF5/NetCDF files and prebuilt caches at the paths given by the
`PREBUILT_DIR`, `PROSE_DATA_DIR` and `CORPUS_ROOT` environment variables; see the
docstrings at the top of each loader.

## Entry points

**Fig. 1 — pretraining and mechanism analysis** (`fig1_pretrain_mechanism/foundationv2/`)

```bash
# MOTION-S (20.7M). MOTION (156.9M) uses the same recipe with --d 896 --depth 12 --d_w 48.
python -m v3.train --families motion --steps 160000 --batch 4 --batch3d 1 \
  --d 384 --depth 8 --d_w 16 --n_p 16 --box_table 1 --gate_cond 1 --fam_head 1 \
  --dict_decode 0 --t_in 10 --k_fut 10 --k_relax 4 --enc_n 8192 --n_colloc 2048 \
  --xla 1 --eval_every 2000 --eval_n 8 --ckpt_every 2000 --save results/motion_s
```

- `core/banks.py` — the mechanism-tendency vocabulary (including the vortex-stretching
  head, identically zero in 2D: the structural-zero row of the knockout matrix).
- `ablate_bank.py`, `probe_surface.py` — knockout damage and contribution-norm protocol.
- `fig1_dump.py`, `fig1_extra.py`, `render_fig1*.py` — panel pipeline; manuscript panel
  inputs are in `refdata/fig1_m2s2/`.
- `infer3d/` — in-graph 3D rotation-ensemble inference.

**Fig. 2 — joint benchmark** (`fig2_joint_benchmark/`, `common/motion_tf/`)

- `motion_tf/model.py`, `motion_tf/train/train_prose_mgpu.py`; config lineage
  `motion_tf/train/configs/prose_150M_fluid5_6x_unified_task2_advb2_arh5_bylfa*.py`.
- Fair-protocol baselines: `jobscripts/bcat-fair.*`, `prose-fair.json`; curves in
  `refdata/fair_baseline_curves.csv`, reference curve `refdata/bylfa_eval_curve_phys.csv`.

**Fig. 3 — IVP pretraining and few-shot transfer** (`fig3_ivp_transfer/`, `common/motion_tf/`)

- `motion_tf/train/train_ivp.py` — IVP-mode trainer (autoregressive hop-2 refeed).
- Pretrain config `motion_tf/train/configs/poseidon_pretrain_ivp_combo_158m_scratch.py` (800k steps).
- Finetuning arms (the config module differs per arm):
  NS-PwC direct = `_pad` cfg + pad2 checkpoint + `FT_UV_ONLY=1 FT_IC_FRAC=1.0`;
  NS-PwC AR = x4 checkpoint + base cfg + hop-2;
  ACE / Wave-Layer = `_pad_t` cfg + pad checkpoint;
  Poisson-Gauss = base cfg + x4 checkpoint + `FT_IC_FRAC=0 FT_LOSS_CAP=100`.
- `jobscripts/scot_ourmetric2.py`, `scot-runner2ic.sh` — scOT rescoring under the same metric.

**Fig. 4 — physical equivalence** (`fig4_physical_equivalence/`)

- `scorers/eps_equiv_motion.py`, `eps_equiv_walrus.py` — ε_equiv(T) = ‖T⁻¹G(Tu) − G(u)‖ / ‖G(u)‖.
- `scorers/wens.py`, `group_size_curve.py` — equivalence-ensemble averaging; `boost_*` — the
  Galilean-boost negative result, derivation in `boost_legality.md`.
- `render/` + `refdata/equiv_readouts.json` — numbers behind Fig. 4c/4d.
  Validity rule used throughout: ε ≤ err_id + err_g.

## Reproduction notes

1. **Numeric mode is part of the learned function.** Score each checkpoint in the mode it
   was trained in (bf16); e.g. Poisson gives 6.26% in bf16 but 13.92% in f32.
2. The 3D ensemble-inference scripts (`infer3d/infer3d_graph*.py`) default to the MOTION-S
   architecture; MOTION needs `ARCH_D=896 ARCH_DEPTH=12 ARCH_DW=48`.
3. The MOTION forward pass needs a GPU; 128²-native boxes run out of memory in eager
   mode — use graph mode (`tf.function`) as the scripts do.
4. Checkpoints are TF 2.15 named-array `.npz` files loaded positionally; use the same TF
   major version.
5. Runner scripts retain the cluster identifiers (volume mounts, job slugs) of the original
   runs for provenance; substitute your own paths.

## License

Code: [MIT](LICENSE). Weights: CC-BY-4.0 (see the Zenodo record).
Third-party baselines and datasets remain under their own licenses.

## Citation

```bibtex
@article{lee2026motion,
  title   = {Mechanism-oriented tendency-integration operator networks for PDE foundation models},
  author  = {Lee, Jeongsu},
  year    = {2026},
  note    = {submitted}
}
```
