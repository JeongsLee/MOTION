#!/bin/bash
# MOTION-NS-S: the Navier--Stokes multi-family model of the joint benchmark (Fig. 4a: vs BCAT / PROSE-FD), trained
# on the MOTION-S architecture of the Methods.
#   Same six families and the SAME training budget as the published MOTION-NS / BCAT / PROSE-FD runs:
#   160,000 updates x effective batch 8 (2 H100 x 4 per replica) = 1.28M sample views, single stage, bf16.
#   LR: the published MOTION-NS (bylfa) schedule exactly: peak 1e-4, linear warmup from 2% over 4k steps, cosine to 5%
#   at 160k (v3/train.py --lr_sched cosine). A 3e-4 peak left com_ns at persistence (MOTION-S stage 1 never learned it
#   at 3e-4; the stage-2 restart at 5e-5 did, with either loss).
#   MOTION-S design (d 384, depth 8, n_p 16) with live warp + 19 typed heads + 1 merged generic head (as MOTION-IVP-S),
#   uniform latent 64^2 (published MOTION-NS: 32^2), t_in 10 / k_fut 10, all six families autoregressive per frame
#   (v3/policy.py; = the published full-AR bylfa setting).
#   FLUCT=1 (revision 10-19, arm motion_nss_fl1): per-slot rel-L2 with mean-removed denominators (VRMSE-aligned; the
#   loss family of BCAT/PROSE/bylfa, which all train on per-sample z-scored targets); inputs keep the DC. Default 0.
#   MOTION_SPLIT_MTF=1: train/test split of the published runs (fv2 hash split would put ~80% of their test-600 in
#   training); in-training eval = their test-600 (val = test, first 100/family), every 5k, as their [EVAL] curves.
set -e
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
export MOTION_WARP_LIVE=1 MOTION_BANK_NGENERIC=8 MOTION_BANK_GENERIC_MERGE=1 MOTION_SPLIT_MTF=1
RUN=${RUN_NAME:-motion_nss}; SEED=${SEED:-0}
OUT=/corpus/results/$RUN; MIRROR=/work/runs/$RUN
mkdir -p $OUT $MIRROR
mkdir -p /tmp/corpus/cache /tmp/corpus/raw/poseidon
ln -s /corpus/meta /tmp/corpus/meta
ln -s /corpus/cache/unified /tmp/corpus/cache/unified
ln -s /corpus/cache/unified_sdf /tmp/corpus/cache/unified_sdf
ln -s /eu/data/prose/prebuilt /tmp/corpus/cache/prose128
ln -s /eu/data/poseidon/_assembled /tmp/corpus/raw/poseidon_assembled
# raw/poseidon from the object volume EXCEPT NS-Gauss, read from the /eu copy (objvol random access stalls)
for p in /work/data/poseidon/*; do b=$(basename $p); [ "$b" = "NS-Gauss" ] || ln -s $p /tmp/corpus/raw/poseidon/$b; done
ln -s /eu/data/poseidon_raw/NS-Gauss /tmp/corpus/raw/poseidon/NS-Gauss
ln -s /corpus/raw/pdebench /tmp/corpus/raw/pdebench; ln -s /corpus/results /tmp/corpus/results
export CORPUS_ROOT=/tmp/corpus
cp -r /work/MOTION/motion_tf /tmp/fv2; cd /tmp/fv2
pip install -q "numpy<2" h5py scipy netCDF4 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv | tee $OUT/gpu.txt
env | grep -E "^MOTION_" | tee $OUT/env.txt || true
ls -la /tmp/corpus/cache/prose128/ | grep -iE "cfdbench|pdearena|shallow|com_ns|incom" | tee $OUT/data.txt || true
( while true; do sleep 300; cp -f $OUT/train.log $MIRROR/train.log 2>/dev/null || true; done ) &
python -m v3.train --families shallow_water,com_ns,incom_ns,pdearena_ns,cfdbench,pdearena_uncond --steps ${STEPS:-160000} \
  --batch 4 --batch3d 1 --d 384 --depth 8 --d_w 16 --n_p 16 --box_table 0 --dims2 64 --gate_cond 1 --fam_head 1 --dict_decode 0 \
  --t_in 10 --k_fut 10 --k_relax 4 --enc_n 8192 --n_colloc 2048 --xla 1 \
  --lr ${LR:-1e-4} --warmup ${WARMUP:-4000} --lr_sched cosine --lr_alpha 0.05 --fluct_loss ${FLUCT:-0} --bf16 1 --seed $SEED \
  --eval_every 5000 --eval_n 100 --ckpt_every 5000 --save $OUT ${EXTRA_ARGS:-} 2>&1 | tee -a $OUT/train.log
rc=${PIPESTATUS[0]}
cp -f $OUT/train.log $MIRROR/train.log; echo "rc=$rc" | tee $MIRROR/rc.txt
