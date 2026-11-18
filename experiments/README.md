# Experiments of the revised manuscript: results, scripts and settings

`jobs/` holds the cluster job scripts as they were run, and `latent_control/` the workstation launchers of the
latent-command experiments. They are records of the runs: they keep the directory layout of the cluster
(`/corpus` = corpus root, `/eu` = data and baseline results, `/work` = code and run outputs), which has to be
replaced by your own paths. Several runs received part of their `MOTION_*` environment from the job submission
rather than from the script; the tables below give the complete environment.

Scripts named without a folder are in `motion_tf/`. `analysis/` and `figures/` are top-level folders.

## 1. Environment of the released models

| Model | Environment | Options |
|---|---|---|
| MOTION-S, MOTION-L (and runs continued from them) | `MOTION_LEGACY_VOCAB=1 MOTION_VORTEX2D_LEGACY=1`; `NO_DEAD_REPAIR=1` at evaluation | MOTION-L: `--d 896 --depth 12 --d_w 48` |
| Ablation variants (20,000 updates) | the variant switch of Section 2 only | — |
| MOTION-NS | `MOTION_WARP_LIVE=1 MOTION_BANK_NGENERIC=8 MOTION_BANK_GENERIC_MERGE=1 MOTION_SPLIT_MTF=1 MOTION_INPUT_ZSCORE=1 MOTION_COND_STATS=1`; evaluation adds `MOTION_BOX_UNIFORM=64` | `--box_table 0 --dims2 64` |
| MOTION-IVP | `MOTION_BANK_NGENERIC=8 MOTION_BANK_GENERIC_MERGE=1 MOTION_IC_MAXFRAC=0.95 MOTION_FT_NOMETA=1 MOTION_FORCE_MODE=ss MOTION_SS_WARP=blend MOTION_WARP_LIVE=1 MOTION_INPUT_ZSCORE=1 MOTION_GLOBAL_NORM=1 MOTION_EVAL_L1=1`; evaluation adds `MOTION_BOX_UNIFORM=64` | `--n_p 32` |

## 2. Matched-budget variants (Fig. 3, Supplementary Note 1.3)

All variants are trained with `jobs/run_A20.sh` (one H100, 20,000 updates, batch 8, learning rate 3e-4, bf16) and
evaluated with `jobs/run_eval20k.sh` (`MODE=stats|contrib|equiv`), with the switch below set in the environment of
both. Seeds: `SEED=1,2`.

| Manuscript name | Switch |
|---|---|
| MOTION-S (matched-budget reference) | none |
| w/o mechanism | `MOTION_BANK_GENERIC=1` |
| w/o integration | `MOTION_STD_DECODER=1` |
| w/o mechanism and integration | `MOTION_BANK_GENERIC=1 MOTION_STD_DECODER=1` |
| w/o metadata | `MOTION_BANK_NOMETA=1` |
| w/o mechanism and metadata | `MOTION_BANK_GENERIC=1 MOTION_BANK_NOMETA=1` |
| w/o mechanism, integration and metadata | `MOTION_BANK_GENERIC=1 MOTION_STD_DECODER=1 MOTION_BANK_NOMETA=1` |
| with transport map | `MOTION_WARP_LIVE=1` |
| w/o mechanism, with transport map | `MOTION_BANK_GENERIC=1 MOTION_WARP_LIVE=1` |
| with generic tendency | `MOTION_BANK_NGENERIC=8 MOTION_BANK_GENERIC_MERGE=1` |
| with transport map and generic tendency | `MOTION_WARP_LIVE=1 MOTION_BANK_NGENERIC=8 MOTION_BANK_GENERIC_MERGE=1` |
| nineteen heads plus eight generic heads | `MOTION_WARP_LIVE=1 MOTION_BANK_NGENERIC=8` |
| single head, linear / nonlinear / linear plus eight generic | `MOTION_WARP_LIVE=1 MOTION_BANK_SINGLE=lin` / `=mlp` / `=lin MOTION_BANK_NGENERIC=8` |
| five-mechanism vocabularies (Top-5, Bottom-5, Random-5 a, b) | `MOTION_WARP_LIVE=1 MOTION_BANK_ACTIVE=` `geometry,shock,reaction,diffusion,advection` / `dilatation,wave,buoyancy,shear,elliptic` / `advection,diffusion,elliptic,reaction,wave` / `advection,buoyancy,elliptic,geometry,shear` |
| transport displacement forms | `MOTION_WARP_LIVE=1 MOTION_WARP_MODE=advbank` / `physvel` |
| w/o state head | `MOTION_BANK_NOH0=1` |
| FNO backbone (with the three bank / decoder settings) | `MOTION_CORE=fno` (+ `MOTION_BANK_GENERIC=1`, + `MOTION_STD_DECODER=1`) |
| continued training after head removal | `jobs/run_contko.sh`, `KOHEAD=diffusion` or `spectral`, 10,000 updates from MOTION-S |
| wrong or missing metadata (evaluation only) | `jobs/run_meta.sh`, `MOTION_META_MODE=all_on|none|swap|drop1|add1` |

Head names in the code and in the manuscript: `compressible` density coupling, `gradvec` directional gradient,
`spectral` spectral nonlocal, `coefficient` coefficient-weighted diffusion, `geometry` mean curvature, `kinematic`
velocity-gradient invariants, `vortex` vortex stretching, `boundary_layer` near-wall coupling, `shock` upwind
transport, `bernoulli` dynamic pressure, `elliptic` domain mean, `shear` first-axis gradient, `buoyancy` gravity
gradient, `wave` wave Laplacian, `reaction` / `reaction_nl` linear / nonlinear reaction.

## 3. Results and the scripts behind them

| Result | Computation | Job or launcher | Figure / table script |
|---|---|---|---|
| Pretraining, MOTION-S and MOTION-L | `v3/train.py` | `jobs/pretrain_motion_s_stage1.sh`, `pretrain_motion_s_stage2.sh`, `pretrain_motion_l_stage1.sh`, `pretrain_motion_l_stage2.sh` (the original job commands); `jobs/run_staged.sh` replays the MOTION-S second stage with the present code | `figures/render_s1_history.py` |
| Pretraining, MOTION-NS | `v3/train.py` | `jobs/run_nss.sh` | `figures/render_figS_ns_curves.py` |
| Pretraining, MOTION-IVP (three stages) | `v3/train.py`, `pad_tin.py` | `jobs/run_ivps_ss.sh`, `jobs/run_ivps_s3w.sh` | — |
| Second-stage loss control | `v3/train.py`, `eval_stats.py` | `jobs/run_staged.sh`, `jobs/run_staged_eval.sh` | `figures/fig_staged_curves.py` |
| Per-family test errors | `eval_stats.py` | `jobs/run_eval20k.sh` (`MODE=stats`, `N=128`) | table assembled by hand; intervals with `analysis/stats_compare.py` |
| Sample averaging in three dimensions | `eval_ens3d.py` | `jobs/run_ens3d_node.sh` | table assembled by hand |
| Fig. 1b fields | `fig1_dump.py` | `jobs/run_fig1_m1f_b.sh` | `render_fig1b_r1.py` |
| Fig. 1c, full and unnormalized maps | `eval_stats.py`, `eval_contrib.py` | `jobs/run_eval20k.sh` (`MODE=stats`, `MODE=contrib NC=128`) | `figures/fig1c_m1f.py`, `figures/fig1c_raw.py` |
| Knockout signatures in the fields | `fig1_dump.py` | `jobs/run_fig1_m1f.sh` | `figures/render_figS2_ko.py` |
| Regime dependence (Mach number and the other pre-specified pairs) | `reliance_diag.py`, `eval_stats.py` | 128 test trajectories per family | `analysis/reliance_corr.py --perm 1000`, `figures/fig_regime.py` |
| Fig. 2a,b uniform commands and calibration | `latent_control_k.py` | `latent_control/chain8.sh`, `chain9.sh`, `chain10.sh` | `figures/fig2_control.py` |
| Fig. 2c,d shear and vortex commands against the solver | `latent_control3.py`, `analysis/ns_spectral.py` | `latent_control/chain11.sh`, `chain13.sh`, `chain25.sh` | `figures/fig2_control.py`, `figures/render_figS_cmd_specificity.py` |
| Fig. 2c,e diffusivity command | `latent_knob.py`, `analysis/ns_spectral.py` | `latent_control/chain18.sh`, `21`, `23`, `24`, `26` | `figures/fig2_control.py` |
| Fig. 2f–h expansion and compression | `latent_control3.py` | `latent_control/chain14.sh`, `chain29.sh` | `figures/fig2_control.py` |
| Long-horizon commands | `latent_control3.py --K 30 --horizon_all 1` | `latent_control/chain30.sh` | `figures/render_figS_long_horizon.py` |
| Latent velocity and probes | `latent_dump.py` | `jobs/run_latent.sh` | `figures/latent_analysis.py` |
| Latent composition | `latent_compose.py` | `latent_control/run_compose.sh` | `figures/fig_latent_compose.py` |
| Fig. 3 and the variant tables | Section 2 | `jobs/run_A20.sh`, `jobs/run_eval20k.sh` | `figures/fig3_ablation_r1.py` with `analysis/fx_common.py` (D, same-form separation) and `analysis/align_metrics.py` (AUC, permutation test) |
| Operator form of the generic heads | `eval_stencil_func.py` | `jobs/run_stencil_func.sh` | table assembled by hand |
| Pairwise and class knockouts | `eval_redund.py`, `eval_stats.py` | `jobs/run_redund.sh`, `jobs/run_groups.sh` | `analysis/group_analysis.py`, `analysis/hybrid_analysis.py` |
| Shortened observed history | `eval_stats.py --t_obs … --ar_only 1` | `jobs/run_horizon.sh` | `figures/fig_horizon.py` |
| Transport-map and state-head tables | variant runs of Section 2 | — | `analysis/tables/warp_tab.py`, `v5_h0.py`, `v5_ko.py`, `make_table.py` |
| Metadata sensitivity | `eval_stats.py` | `jobs/run_meta.sh` | `figures/meta_sensitivity.py` |
| Fig. 4a, MOTION-NS | `eval_stats.py` | `jobs/run_nss_final.sh` | table in the manuscript |
| Fig. 4a, retrained BCAT and PROSE-FD | upstream code with the loader and evaluation patch | `baselines/retrained_bcat_prose/` | — |
| Released BCAT / PROSE-FD weights | `baselines/pubweights/` | — | table assembled by hand |
| Baselines with physical input features | `baselines/basefeat/` (`BASEFEAT=phys`; knockouts `BASEFEAT_KO`) | — | `baselines/basefeat/ko_metrics.py` |
| MOTION-NS mechanism metrics and latent commands | `eval_contrib.py`, `eval_stats.py`; `latent_control_k.py`, `latent_control3.py` | `jobs/run_nss_mech.sh`; `latent_control/chain31.sh`, `33`, `34`, `39`, `40` | `analysis/align_metrics.py`; `figures/render_figS_ns_control_ab.py` |
| Fig. 4b, MOTION-IVP on the fifteen downstream tasks | `warmstart_famrow.py`, `v3/train.py`, `score_ivp_l1.py` | `jobs/run_ft15_motion_v4.sh` (Section 4) | `figures/make_fig4b.py` |
| Fig. 4b, Poseidon-T and Poseidon-B | upstream scOT; `baselines/poseidon_ft_data/eval_scot_ft15.py` | `baselines/poseidon_ft_data/run_ft15_scot.sh` | `figures/make_fig4b.py` |
| Paired comparison per trajectory | `score_ivp_l1.py` | `jobs/run_ft15_rescore.sh` | `analysis/tables/paired_stats.py`, `make_tab_paired.py` |
| Inference-mode and recipe sensitivity | as Fig. 4b | `jobs/run_ft15_motion_v5.sh` (`MIX_SS=1.0`) and the options of Section 4 | table assembled by hand |
| MOTION-IVP mechanism metrics | `eval_contrib.py` | `jobs/run_ivp_contrib.sh`, `jobs/run_ft15_contrib.sh` | `analysis/align_metrics.py` |
| Fig. 4d and the equivalence tables, MOTION models | `eval_equiv.py --members d4xshift` | `jobs/run_nss_equiv.sh`, `jobs/run_ivp_equiv_c.sh`, `jobs/run_m1f_equiv_full.sh`, `jobs/run_m1f_equiv_pch.sh` | `analysis/equivalence/stats_equiv3.py`, `stats_equiv3_heldout.py`; `figures/render_fig4b_replica.py` |
| Equivalence tables, Poseidon | `poseidon_equiv.py`; split `baselines/poseidon_ft_data/heldout_both.py` | `jobs/run_poseidon_equiv.sh` | same |
| Equivalence tables, BCAT and PROSE-FD | `baselines/symmetry_dumps/`; `analysis/baseline_symmetry/` | `jobs/run_baseline_pertraj.sh` | same |
| Equivariance defect of the variants | `eval_equiv.py` | `jobs/run_eval20k.sh` (`MODE=equiv NE=32`) | `analysis/equivalence/bank_backbone.py`, `pooled.py`, `backbone_fixed_bank.py`, `factorial_contrasts.py` |
| Example trajectories under transformation | `eval_equiv.py`; `analysis/baseline_symmetry/swe_onetraj*.py`, `cfd_onetraj.py` | `jobs/run_nss_equiv_onetraj.sh` | `figures/fig_r1_swe_onetraj.py`, `figures/fig_cfd_onetraj.py` |
| Inference time, MOTION | `eval_equiv.py` with `EQUIV_TIME_ONLY=1` | `jobs/run_time_infer.sh` | — |
| Inference time, BCAT and PROSE-FD | upstream evaluation with timers | `jobs/run_time_baselines.sh` | — |
| Training-step profile | `analysis/bench_speed.py` | — | — |
| Reference and accelerated PyTorch timings | `motion_torch/bench_ivp.py`; `baselines/poseidon_ft_data/bench_scot_time.py` | — | — |

## 4. Fine-tuning on the fifteen Poseidon downstream tasks

`jobs/run_ft15_motion_v4.sh` with the MOTION-IVP environment plus `N_P=32 MOTION_CACHE_UNITS=1 MOTION_DEC_FUSE=1`.
Training: `--batch 4 --grad_accum 10 --steps 32000` (3,200 updates of 40 trajectories) `--lr 5e-5 --warmup 100
--lr_sched cosine --lr_alpha 0.05 --t_in 10 --t_in_min 1 --ic_rand 2 --loss_l1 1 --per_channel_loss 0 --clamp 4
--max_per_fam 64 --transport 1 --bf16 1`. Scoring: `score_ivp_l1.py` on the official test split with the best
validation checkpoint.

| Tasks | Additional setting |
|---|---|
| NS-PwC, NS-SVS, NS-BB, NS-SL, NS-Tracer-PwC, CE-RPUI, FNS-KF, CE-RM, ACE | `MOTION_RH_FAMS=FT-<task> MOTION_RH_ROLL=10 MOTION_RH_LEADS=1` |
| Wave-Gauss, Wave-Layer | the same and `MOTION_COEF_GLOBAL=1` |
| GCE-RT | the same and `MOTION_FT_CE4=1` |
| Poisson-Gauss, Helmholtz | `K_RELAX=1 MOTION_COEF_GLOBAL=1` |
| SE-AF | `K_RELAX=1` |

Poseidon-T and Poseidon-B: `baselines/poseidon_ft_data/run_ft15_scot.sh` (`TASK`, `PMODEL=T|B`; 64 trajectories,
3,200 updates, batch 40, the released scOT recipe).

## 5. What the scripts do not cover

- The MOTION-S and MOTION-L pretraining launchers are the original job commands of August 2026; they ran the code
  of that date. `jobs/run_staged.sh` replays the MOTION-S second stage with the present code.
- Tables marked "assembled by hand" were typed from the outputs of the listed scripts.
- Figure scripts read the outputs of the listed runs from `results/` (or `$MOTION_RESULTS`) and, for the latent
  commands, from `$MOTION_LOCAL`; the values behind every main-text figure are provided as Source Data with the
  paper.
- Baseline code (BCAT, PROSE-FD, scOT) is fetched from the upstream repositories by the runner scripts.
