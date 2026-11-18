#!/bin/bash
# A20 ablation protocol run (revision R1). RUN_NAME, SEED and optional EXTRA_ARGS / EXTRA_ENV come from the job env.
#   H100x1, 2D batch 8 / 3D batch 1, 20k steps, stage-1 loss only, lr 3e-4 const (warmup 2k), bf16,
#   fv2_opt code (static gating; vortex K=3 only; fast attention off).
set -e
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
RUN=${RUN_NAME:?}; SEED=${SEED:-0}
OUT=/corpus/results/$RUN; MIRROR=/work/runs/$RUN
mkdir -p $OUT $MIRROR
# --- corpus shim (same as eval19 v2): 2D caches live on /eu and the object volume since 08-19
mkdir -p /tmp/corpus/cache /tmp/corpus/raw
ln -s /corpus/meta /tmp/corpus/meta
ln -s /corpus/cache/unified /tmp/corpus/cache/unified
ln -s /corpus/cache/unified_sdf /tmp/corpus/cache/unified_sdf
ln -s /eu/data/prose/prebuilt /tmp/corpus/cache/prose128
ln -s /eu/data/poseidon/_assembled /tmp/corpus/raw/poseidon_assembled
ln -s /work/data/poseidon /tmp/corpus/raw/poseidon
ln -s /corpus/raw/pdebench /tmp/corpus/raw/pdebench
ln -s /corpus/results /tmp/corpus/results
export CORPUS_ROOT=/tmp/corpus
cp -r /work/MOTION/motion_tf /tmp/fv2; cd /tmp/fv2
pip install -q "numpy<2" h5py scipy netCDF4 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv | tee $OUT/gpu.txt
env | grep -E "^MOTION_" | tee $OUT/env.txt || true
# mirror the log to the object volume every 5 min (for remote monitoring)
( while true; do sleep 300; cp -f $OUT/train.log $MIRROR/train.log 2>/dev/null || true; done ) &
python -m v3.train --families motion --steps 20000 --batch 8 --batch3d 1 \
  --d 384 --depth 8 --d_w 16 --n_p 16 --box_table 1 --gate_cond 1 --fam_head 1 --dict_decode 0 \
  --t_in 10 --k_fut 10 --k_relax 4 --enc_n 8192 --n_colloc 2048 --reuse3d 8 --xla 1 \
  --lr 3e-4 --warmup 2000 --fluct_loss 0 --bf16 1 --seed $SEED \
  --eval_every 2000 --eval_n 8 --ckpt_every 2000 --save $OUT $EXTRA_ARGS 2>&1 | tee -a $OUT/train.log
rc=${PIPESTATUS[0]}
cp -f $OUT/train.log $MIRROR/train.log; echo "rc=$rc" | tee $MIRROR/rc.txt
exit $rc
