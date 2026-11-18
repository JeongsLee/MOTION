#!/bin/bash
# REVISION-R1 (user 2026-10-25): FT15 from the stage-3 option-C checkpoint (motion_ivps_gn3w: t_in-10 window filled by
# the model's own hop-1 rollout). Same protocol as run_ft15_motion_v2.sh (FT15_PLAN.md: 64 official training
# trajectories, official val for checkpoint selection, official test for the score, 3,200 updates x effective batch 40,
# lr 5e-5 cosine over the UPDATES, bf16), with the option-C mode for every time-dependent task:
#   receding horizon (MOTION_RH_FAMS = the task), RH_ROLL 10, observed history drawn in [1, 10] (--t_in_min 1), the rest
#   of the 10-frame window masked; val track and test score = strict IVP (1 observed frame at t = 0, hop-1 AR over the
#   horizon, each prediction appended to the window). Steady tasks: MOTION relaxation from an empty window, K_RELAX 10
#   iterates (each appended to the window), the same length as the transient rollout.
#   Speed-ups on: in-memory units + threaded batch prefetch (MOTION_CACHE_UNITS), fused lead decoding (MOTION_DEC_FUSE),
#   batched scoring (SCORE_BATCH 16). Batch 4 x grad_accum 10 (RH_ROLL 10 at batch 8 needs > 80 GB).
# Env: TASK, INIT (default motion_ivps_gn3w ckpt_best), TAG (default ivps_c), EVAL_ONLY, SCORE_SPLITS, SCORE_CKPTS,
#      BATCH, GACC, STEPS (micro-steps), EVAL_EVERY (micro-steps).
set -e
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
export MOTION_BANK_NGENERIC=8 MOTION_BANK_GENERIC_MERGE=1 MOTION_IC_MAXFRAC=0.95 MOTION_FT_NOMETA=1 \
       MOTION_FORCE_MODE=ss MOTION_SS_WARP=blend MOTION_WARP_LIVE=1 MOTION_INPUT_ZSCORE=1 MOTION_GLOBAL_NORM=1 \
       MOTION_EVAL_L1=1 N_P=32 MOTION_CACHE_UNITS=1 MOTION_DEC_FUSE=1
TASK=${TASK:?TASK}; FAM=FT-$TASK; INIT=${INIT:-/corpus/results/motion_ivps_gn3w/ckpt_best.npz}
# task -> (warm-start family, horizon, receding-horizon?, scored slots for the Poseidon comparison, test size)
case "$TASK" in
  NS-PwC|NS-SVS|NS-BB|NS-SL|FNS-KF) FROM=NS-Sines; KFUT=20; RH=1; SLOTS=0,1;   NTE=240;;
  NS-Tracer-PwC)                    FROM=NS-Sines; KFUT=20; RH=1; SLOTS=0,1,6; NTE=240;;
  CE-RPUI)                          FROM=CE-RP;    KFUT=20; RH=0; SLOTS=3,0,1,4; NTE=240;;
  CE-RM)                            FROM=CE-RP;    KFUT=20; RH=0; SLOTS=3,0,1,4; NTE=130;;
  GCE-RT)                           FROM=CE-RP;    KFUT=10; RH=0; SLOTS=3,0,1,4; NTE=130;;
  Wave-Gauss)                       FROM=NS-Sines; KFUT=15; RH=0; SLOTS=10;  NTE=240;;
  Wave-Layer)                       FROM=NS-Sines; KFUT=20; RH=0; SLOTS=10;  NTE=240;;
  ACE)                              FROM=NS-Sines; KFUT=19; RH=0; SLOTS=6;   NTE=240;;
  Poisson-Gauss|Helmholtz)          FROM=CE-RP;    KFUT=1;  RH=0; SLOTS=4;   NTE=$([ "$TASK" = Helmholtz ] && echo 512 || echo 240);;
  SE-AF)                            FROM=CE-RP;    KFUT=1;  RH=0; SLOTS=3;   NTE=240; export SCORE_EXCLUDE_GEOM=1;;
  *) echo "unknown TASK $TASK"; exit 2;;
esac
# steady tasks: pseudo-transient relaxation with K_RELAX steps (default 10 = the pretraining window length, so the
# 10-frame window fills with the relaxation iterates as it fills with the rollout in the transient tasks)
KREL=${K_RELAX:-10}
# option C: every time-dependent task in receding-horizon mode; strict-IVP val track over the task horizon
[ "$KFUT" -gt 1 ] && RH=1
[ "$RH" = 1 ] && export MOTION_RH_FAMS=$FAM MOTION_RH_ROLL=${RH_ROLL:-10} MOTION_RH_LEADS=1 MOTION_EVAL_TI=1 MOTION_EVAL_KF=$KFUT
# Poseidon-paper metric (Herde et al. 2024, SM D.1 / Table 6): lead 14 (GCE-RT 7), functions of interest as slot groups
PLEAD=14; [ "$TASK" = GCE-RT ] && PLEAD=7
case "$TASK" in
  NS-Tracer-PwC) PFUNCS="0,1;6";; CE-RPUI|CE-RM) PFUNCS="3;0,1;4";; GCE-RT) PFUNCS="3;0,1;4;2";;
  *) PFUNCS=$SLOTS;;
esac
S=/corpus/results/ft15/${TAG:-ivps_c}/$TASK; MIRROR=/work/runs/ft15/${TAG:-ivps_c}/$TASK
mkdir -p $S $MIRROR
mkdir -p /tmp/corpus/raw /tmp/corpus/cache
ln -s /corpus/meta /tmp/corpus/meta
# REVISION-R1 10-24: the FT files have no materialized cache, so the lazy loader opens the .nc on cephfs for every
# sample (0.04 it/s measured). Copy this task's file(s) to local disk first.
FB=$TASK; [ "$TASK" = NS-Tracer-PwC ] && FB=NS-PwC
mkdir -p /tmp/ftdata/ft; cp /eu/data/poseidon_ft/ft/$FB.* /tmp/ftdata/ft/; ls -la /tmp/ftdata/ft
ln -s /tmp/ftdata /tmp/corpus/raw/poseidon_ft
ln -s /corpus/results /tmp/corpus/results
export CORPUS_ROOT=/tmp/corpus
cp -r /work/MOTION/motion_tf /tmp/fv2; cd /tmp/fv2
pip install -q "numpy<2" h5py scipy netCDF4 2>&1 | tail -1
env | grep -E "^MOTION_|^N_P|^TASK" | tee $S/env.txt
IDS=$(python -c "from data.registry import FAMILIES as F; k=list(F); print(k.index('$FROM'), k.index('$FAM'))")
set -- $IDS; FROM_ID=$1 TO_ID=$2 python warmstart_famrow.py $INIT $S/init_warm.npz
( while true; do sleep 300; cp -f $S/train.log $S/score.log $MIRROR/ 2>/dev/null || true; done ) &
ARGS="--families $FAM --batch ${BATCH:-4} --grad_accum ${GACC:-10} --t_in 10 --t_in_min 1 --k_fut $KFUT --ic_rand 2 --k_relax $KREL --d 384 --depth 8 --d_w 16 \
  --n_p 32 --box_table 0 --dims2 64 --gate_cond 1 --fam_head 1 --dict_decode 0 --enc_n 8192 --n_colloc 2048 --xla 1 \
  --bf16 1 --loss_l1 1 --per_channel_loss 0 --clamp 4 --fluct_loss 0 --max_per_fam 64 --transport 1"
[ "${EVAL_ONLY:-0}" = 1 ] || \
python -m v3.train $ARGS --steps ${STEPS:-32000} --lr ${LR:-5e-5} --warmup 100 --lr_sched cosine --lr_alpha 0.05 \
  --eval_every ${EVAL_EVERY:-4000} --eval_n 32 --ckpt_every ${EVAL_EVERY:-4000} --init_ckpt $S/init_warm.npz --save $S 2>&1 | tee -a $S/train.log
HOPARG=""; [ "$RH" = 1 ] && HOPARG="HOP=1"
# REVISION-R1 10-24: BOX_UNIFORM=64 = the training box (--box_table 0 --dims2 64); without it the scorer fell back to the
# 32-box default for the FT-* families (unlisted in BOX_TABLE_2D) and scored 54% on NS-PwC where the val track read 3.6%.
# SCORE_SPLITS (default test; "val test" adds a sanity check against the training val track), SCORE_CKPTS.
for C in ${SCORE_CKPTS:-$S/ckpt_best.npz $S/init_warm.npz}; do for SP in ${SCORE_SPLITS:-test}; do
  N=$NTE; [ "$SP" = val ] && N=${NEV_VAL:-32}
  echo "===== $C  ($SP, n=$N, slots $SLOTS)" | tee -a $S/score.log
  # strict IVP: 1 observed frame at t = 0 (MOTION_TI_FIXED, scoring process only), the window fills during the rollout
  env CKPT=$C FAMS=$FAM KF=$KFUT K_RELAX=$KREL T_IN=10 MOTION_TI_FIXED=1 SCORE_BATCH=${SCORE_BATCH:-16} NEV=$N SPLIT=$SP DECODE=ss \
    SCORE_SLOTS=$SLOTS BOX_UNIFORM=64 $HOPARG \
    PAPER_LEAD=$PLEAD PAPER_FUNCS="$PFUNCS" \
    python -u score_ivp_l1.py 2>&1 | grep -E 'EVAL-L1|@final|HOP|hop|PAPER|global norm' | tee -a $S/score.log
done; done
cp -f $S/train.log $S/score.log $S/env.txt $MIRROR/ 2>/dev/null || true
echo FT_DONE | tee $S/done.txt $MIRROR/done.txt
