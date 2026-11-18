#!/bin/bash
# REVISION-R1 10-24 (user): stage 3 of MOTION-IVP-S with a 10-frame window FILLED BY THE MODEL'S OWN ROLLOUT.
#   Warm start: motion_ivps_gn2 ckpt_best (t_in 1), widened by pad_tin.py to t_in 10 (the old encoder rows go to the
#   latest-frame slot, the 9 older slots start at zero -> output on a 1-frame input unchanged at load).
#   All six families in receding-horizon mode (step_ss, same k/kf time map and n_p 32 as gn2): each call emits lead 1,
#   which is appended to the window (fmask 1), MOTION_RH_ROLL 10 calls per example, so the window fills from the
#   observed frames up to 10 frames of the model's own trajectory and the frame-to-frame changes reach the encoder.
#   Observed history drawn in [1, 10] (--t_in_min 1; the rest masked), IC position all2all (IC_MAXFRAC 0.95).
#   Val = strict IVP: 1 observed frame at t = 0, hop-1 AR over 20 frames (MOTION_EVAL_TI 1, MOTION_EVAL_KF 20).
#   Otherwise as gn2 (metadata off, global norm, rel-L1 clamp 4, blend warp, 19 typed + 1 merged generic head).
set -e
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
export MOTION_BANK_NGENERIC=8 MOTION_BANK_GENERIC_MERGE=1 MOTION_IC_MAXFRAC=0.95 MOTION_FT_NOMETA=1 \
       MOTION_FORCE_MODE=ss MOTION_SS_WARP=blend MOTION_WARP_LIVE=1 MOTION_INPUT_ZSCORE=1 MOTION_GLOBAL_NORM=1 \
       MOTION_EVAL_L1=1 MOTION_RH_FAMS=CE-RP,CE-KH,CE-CRP,CE-Gauss,NS-Sines,NS-Gauss MOTION_RH_ROLL=${RH_ROLL:-10} \
       MOTION_RH_LEADS=1 MOTION_EVAL_TI=1 MOTION_EVAL_KF=20
RUN=${RUN_NAME:-motion_ivps_gn3w}; SEED=${SEED:-0}
if [ "${SMOKE:-0}" = 1 ]; then OUT=/tmp/smoke; MIRROR=/work/runs/${RUN}_smoke
else OUT=/corpus/results/$RUN; MIRROR=/work/runs/$RUN; fi
mkdir -p $OUT $MIRROR
[ "${SMOKE:-0}" = 1 ] && exec > $MIRROR/smoke.txt 2>&1
mkdir -p /tmp/corpus/cache /tmp/corpus/raw/poseidon
ln -s /corpus/meta /tmp/corpus/meta
ln -s /corpus/cache/unified /tmp/corpus/cache/unified
ln -s /corpus/cache/unified_sdf /tmp/corpus/cache/unified_sdf
ln -s /eu/data/prose/prebuilt /tmp/corpus/cache/prose128
ln -s /eu/data/poseidon/_assembled /tmp/corpus/raw/poseidon_assembled
for p in /work/data/poseidon/*; do b=$(basename $p); [ "$b" = "NS-Gauss" ] || ln -s $p /tmp/corpus/raw/poseidon/$b; done
ln -s /eu/data/poseidon_raw/NS-Gauss /tmp/corpus/raw/poseidon/NS-Gauss
ln -s /corpus/raw/pdebench /tmp/corpus/raw/pdebench; ln -s /corpus/results /tmp/corpus/results
export CORPUS_ROOT=/tmp/corpus
cp -r /work/MOTION/motion_tf /tmp/fv2; cd /tmp/fv2
pip install -q "numpy<2" h5py scipy netCDF4 2>&1 | tail -1
env | grep -E "^MOTION_" | tee $OUT/env.txt || true
ARGS="--families CE-RP,CE-KH,CE-CRP,CE-Gauss,NS-Sines,NS-Gauss --batch 8 --batch3d 1 \
  --d 384 --depth 8 --d_w 16 --n_p 32 --box_table 0 --dims2 64 --gate_cond 1 --fam_head 1 --dict_decode 0 \
  --t_in 10 --t_in_min 1 --k_fut 20 --ic_rand 2 --k_relax 4 --enc_n 8192 --n_colloc 2048 \
  --loss_l1 1 --per_channel_loss 0 --clamp 4 --fluct_loss 0 --seed $SEED"
if [ "${SMOKE:-0}" = 1 ]; then
  # CPU memory: batch 1 (the first smoke at batch 8 was killed with no output); streamed, not tailed
  python -m v3.train $ARGS --batch 1 --steps 40 --xla 0 --bf16 0 --lr 1e-4 --warmup 10 --lr_sched cosine \
    --lr_alpha 0.05 --eval_every 40 --eval_n 1 --ckpt_every 40 --save $OUT 2>&1 | grep --line-buffered -vE "tmp_|}'>|autotuner|graph_launch" || true
  python - <<'PY'
import glob, numpy as np
f = sorted(glob.glob("/tmp/smoke/ckpt_*.npz"))[-1]; z = np.load(f, allow_pickle=True)
for k in z.files:
    if "ss_warp" in k or "disp" in k or "tgate" in k: print("WARPVAR", k, float(np.linalg.norm(z[k])))
print("NKEYS", len(z.files))
PY
  echo SMOKE_DONE; exit 0
fi
python pad_tin.py ${INIT:-/corpus/results/motion_ivps_gn2/ckpt_best.npz} /tmp/init_t10.npz 10 | tee -a $OUT/train.log
( while true; do sleep 300; cp -f $OUT/train.log $MIRROR/train.log 2>/dev/null || true; done ) &
nvidia-smi --query-gpu=name,memory.total --format=csv | tee $OUT/gpu.txt
python -m v3.train $ARGS --steps ${STEPS:-50000} --xla 1 --bf16 1 --lr ${LR:-5e-5} --warmup 2000 --lr_sched cosine \
  --lr_alpha 0.05 --eval_every 5000 --eval_n 8 --ckpt_every 5000 --save $OUT --init_ckpt /tmp/init_t10.npz ${EXTRA_ARGS:-} 2>&1 | tee -a $OUT/train.log
rc=${PIPESTATUS[0]}
cp -f $OUT/train.log $MIRROR/train.log; echo "rc=$rc" | tee $MIRROR/rc.txt
exit $rc
