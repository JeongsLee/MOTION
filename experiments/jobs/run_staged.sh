#!/bin/bash
# Staged-objective control on the representative model MOTION-S (R1-1 "training with the first-stage loss only").
# Both arms start from the SAME stage-1 snapshot of MOTION-S (motion_m1f/init_stage1.npz, stage-1 best 10.15%@110k)
# with the exact stage-2 recipe of the published run (job-bvsjyun4yz1a: 1 H100, batch 4, lr 5e-5, linear warmup 1000
# then constant, eval every 2000, eval_n 8) and differ ONLY in the objective:
#   ARM=s1 : first-stage loss continued (--fluct_loss 0, ||u|| denominators)
#   ARM=s2 : second-stage loss (--fluct_loss 1, ||u - mean u|| denominators) = re-run of the published stage 2
# Env: ARM, STEPS (default 20000 = the published stage-2 best step).
set -e
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 MOTION_LEGACY_VOCAB=1 MOTION_VORTEX2D_LEGACY=1
mkdir -p /tmp/corpus/cache /tmp/corpus/raw/poseidon
ln -s /corpus/meta /tmp/corpus/meta; ln -s /corpus/cache/unified /tmp/corpus/cache/unified
ln -s /corpus/cache/unified_sdf /tmp/corpus/cache/unified_sdf; ln -s /eu/data/prose/prebuilt /tmp/corpus/cache/prose128
ln -s /eu/data/poseidon/_assembled /tmp/corpus/raw/poseidon_assembled
for p in /work/data/poseidon/*; do b=$(basename $p); [ "$b" = "NS-Gauss" ] || ln -s $p /tmp/corpus/raw/poseidon/$b; done
ln -s /eu/data/poseidon_raw/NS-Gauss /tmp/corpus/raw/poseidon/NS-Gauss
ln -s /corpus/raw/pdebench /tmp/corpus/raw/pdebench; ln -s /corpus/results /tmp/corpus/results
export CORPUS_ROOT=/tmp/corpus
cp -r /work/MOTION/motion_tf /tmp/fv2; cd /tmp/fv2
pip install -q "numpy<2" h5py scipy netCDF4 2>&1 | tail -1
FL=$([ "$ARM" = s1 ] && echo 0 || echo 1); STEPS=${STEPS:-20000}
INIT=/corpus/results/motion_m1f/init_stage1.npz; S=/corpus/results/staged_m1f_$ARM; mkdir -p $S
ls -la $INIT; env | grep -E "^MOTION_" | tee $S/env.txt; echo "ARM=$ARM fluct_loss=$FL steps=$STEPS" | tee -a $S/env.txt
ARGS="--families motion --batch 4 --batch3d 1 --d 384 --depth 8 --d_w 16 --n_p 16 --box_table 1 --gate_cond 1 --fam_head 1 \
  --dict_decode 0 --t_in 10 --k_fut 10 --k_relax 4 --enc_n 8192 --n_colloc 2048 --reuse3d 8 --xla 1 --fluct_loss $FL"
python -m v3.train $ARGS --steps 30 --lr 5e-5 --warmup 200 --eval_every 100000 --init_ckpt $INIT --save /tmp/gate \
  > /tmp/gate.log 2>&1 || { tail -30 /tmp/gate.log; echo GATE_FAIL; exit 1; }
grep -E "init_ckpt\]" /tmp/gate.log; tail -2 /tmp/gate.log; echo GATE_OK
python -m v3.train $ARGS --steps $STEPS --lr 5e-5 --warmup 1000 --eval_every 2000 --eval_n 8 --ckpt_every 2000 \
  --init_ckpt $INIT --save $S 2>&1 | grep --line-buffered -v graph_launch | tee -a $S/train.log
echo "rc=$?" | tee $S/done.txt
