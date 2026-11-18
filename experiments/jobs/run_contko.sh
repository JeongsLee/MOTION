#!/bin/bash
# REVISION-R1 (R2 #1 "after zeroing out, can retraining recover performance?", R3 #1): continued training of the
# published MOTION-S with the DIFFUSION head removed (MOTION_TRAIN_KO=diffusion: contribution x 0 in training and
# evaluation) against the identical continuation without removal (ARM=ctl). Exactly the published stage-2 recipe
# (job-bvsjyun4yz1a / staged control): 1 H100, batch 4, lr 5e-5 after a 1,000-step warmup then constant, second-stage
# loss (--fluct_loss 1), f32, warp off (default), legacy vocab; start = published MOTION-S weights (ckpt_best).
# Then test-split evaluation of the final checkpoint: stats with every single-head knockout, and head contributions.
# Pre-registered expectations (REVISION_LOG 2026-10-18): P1 the families most damaged by the diffusion knockout
# (ACE, 3D CNS, PDEArena-u, PDEArena, NS-Sines) recover partly; P2 a gap to the control remains in them;
# P3 the role moves to other open heads (their knockout damage rises) and D / operator selectivity fall.
set -e
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 MOTION_LEGACY_VOCAB=1 MOTION_VORTEX2D_LEGACY=1
ARM=${ARM:-ko}; STEPS=${STEPS:-10000}
KOHEAD=${KOHEAD:-diffusion}                      # head removed in the ko arm (10-19: spectral = no substitute in the bank)
[ "$ARM" = ko ] && export MOTION_TRAIN_KO=$KOHEAD
TAG=contko_m1f_$ARM; [ "$KOHEAD" != diffusion ] && TAG=${TAG}_$KOHEAD; S=/corpus/results/$TAG; OUT=/work/runs/eval20k/$TAG; mkdir -p $S $OUT
mkdir -p /tmp/corpus/cache /tmp/corpus/raw/poseidon
ln -s /corpus/meta /tmp/corpus/meta; ln -s /corpus/cache/unified /tmp/corpus/cache/unified
ln -s /corpus/cache/unified_sdf /tmp/corpus/cache/unified_sdf; ln -s /eu/data/prose/prebuilt /tmp/corpus/cache/prose128
ln -s /eu/data/poseidon/_assembled /tmp/corpus/raw/poseidon_assembled
# raw/poseidon: everything from the object volume EXCEPT NS-Gauss, which is read from the /eu copy
# (objvol random access stalled >25 min on NS-Gauss; staged by rev-stage-nsgauss 2026-10-15)
for p in /work/data/poseidon/*; do b=$(basename $p); [ "$b" = "NS-Gauss" ] || ln -s $p /tmp/corpus/raw/poseidon/$b; done
ln -s /eu/data/poseidon_raw/NS-Gauss /tmp/corpus/raw/poseidon/NS-Gauss
ln -s /corpus/raw/pdebench /tmp/corpus/raw/pdebench; ln -s /corpus/results /tmp/corpus/results
export CORPUS_ROOT=/tmp/corpus
cp -r /work/MOTION/motion_tf /tmp/fv2; cd /tmp/fv2
pip install -q "numpy<2" h5py scipy netCDF4 2>&1 | tail -1
env | grep -E "^MOTION_" | tee $OUT/env.txt || true
env | grep -E "^MOTION_" | tee $S/env.txt $OUT/env.txt || true
INIT=/corpus/results/motion_m1f/ckpt_best.npz
ARGS="--families motion --batch 4 --batch3d 1 --d 384 --depth 8 --d_w 16 --n_p 16 --box_table 1 --gate_cond 1 --fam_head 1 \
  --dict_decode 0 --t_in 10 --k_fut 10 --k_relax 4 --enc_n 8192 --n_colloc 2048 --reuse3d 8 --xla 1 --fluct_loss 1"
python -m v3.train $ARGS --steps $STEPS --lr 5e-5 --warmup 1000 --eval_every 2000 --eval_n 8 --ckpt_every 2000 \
  --init_ckpt $INIT --save $S 2>&1 | grep --line-buffered -v graph_launch | tee -a $S/train.log
cp $S/train.log $OUT/train.log
CK=$S/ckpt_$STEPS.npz
HEADS=reaction,diffusion,buoyancy,shear,wave,advection,shock,elliptic,geometry,boundary_layer,dilatation,gradvec,bernoulli,kinematic,vortex,reaction_nl,spectral,coefficient,compressible
CFG="base$(for h in ${HEADS//,/ }; do printf ',ko:%s' $h; done)"
python -u eval_stats.py --ckpt $CK --out $OUT/stats.npz --split test --n 128 --batch 8 --batch3d 2 --xla 1 \
  --configs $CFG 2>&1 | grep --line-buffered -vE "graph_launch" | tee $OUT/stats.log
python -u eval_contrib.py --ckpt $CK --out $OUT/contrib.json --n 16 2>&1 | grep --line-buffered -vE "graph_launch" | tee $OUT/contrib.log
echo CONTKO_DONE | tee $OUT/done.txt
