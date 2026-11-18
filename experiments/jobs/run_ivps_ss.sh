#!/bin/bash
# MOTION-IVP-S v2 (revision R1, user 10-19): IVP pretraining from scratch aligned with the published MOTION-IVP
# (motion_tf poseidon_pretrain_ivp_combo_158m_scratch), which worked, while keeping the MOTION-S trunk
# (latent encoding -> 19 typed mechanism heads + 1 merged generic head -> closed-form ADA synthesis, d 384, depth 8):
#   single-shot      MOTION_FORCE_MODE=ss: one trunk call, panels span the 20-frame horizon (published: single-shot)
#   transport        MOTION_SS_WARP=blend (user 10-19: unified with MOTION-S / MOTION-NS-S): the AR carrier's own
#                    variables (disp, tgate x advection token, live init); frame k = IC backward-warped by k*disp(h).
#                    (MOTION_SS_WARP=1 = the published Taylor warp d = v0 t + a0 t^2/2; not used.)
#   normalization    MOTION_INPUT_ZSCORE=1: per-sample, per-channel instance z-score from the input window. The published
#                    MOTION-IVP used the Poseidon loader's per-dataset GLOBAL channel stats instead (its instance_norm flag
#                    was parity-only); 10-21 (user): pass -e MOTION_GLOBAL_NORM=1 for that (fixed per-family mean/std over the
#                    first 64 training trajectories x all frames; v3/data_ar.py; ckpt dir gets global_norm.json).
#   loss             --loss_l1 1 --per_channel_loss 0 --clamp 4: rel-L1 joint over channels per frame, clamped 4
#   IC sampling      all2all, IC frame in [0, 0.95 T] (published max_ic_frac 0.95), later frames masked
#   metadata         off (MOTION_FT_NOMETA=1; published IVP used dummy descriptors)
#   n_p 32           single-shot needs >= 1.6 panels/frame (v3/policy.py note); 20 frames -> 32
#   lr 1e-4 peak, warmup 2k, cosine to 5% (published 1e-4 decay). bf16, 2 H100 x batch 8.
# SMOKE=1: CPU smoke test (40 steps, no XLA/bf16), prints the warp variable norms; no GPU needed.
set -e
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
export MOTION_BANK_NGENERIC=8 MOTION_BANK_GENERIC_MERGE=1 MOTION_IC_MAXFRAC=0.95 MOTION_FT_NOMETA=1 \
       MOTION_FORCE_MODE=ss MOTION_SS_WARP=blend MOTION_WARP_LIVE=1 MOTION_INPUT_ZSCORE=1
RUN=${RUN_NAME:-motion_ivps_ss}; SEED=${SEED:-0}
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
  --t_in 1 --k_fut 20 --ic_rand 2 --k_relax 4 --enc_n 8192 --n_colloc 2048 \
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
( while true; do sleep 300; cp -f $OUT/train.log $MIRROR/train.log 2>/dev/null || true; done ) &
nvidia-smi --query-gpu=name,memory.total --format=csv | tee $OUT/gpu.txt
python -m v3.train $ARGS --steps ${STEPS:-100000} --xla 1 --bf16 1 --lr ${LR:-1e-4} --warmup 2000 --lr_sched cosine \
  --lr_alpha 0.05 --eval_every 5000 --eval_n 8 --ckpt_every 5000 --save $OUT ${EXTRA_ARGS:-} 2>&1 | tee -a $OUT/train.log
rc=${PIPESTATUS[0]}
cp -f $OUT/train.log $MIRROR/train.log; echo "rc=$rc" | tee $MIRROR/rc.txt
exit $rc
