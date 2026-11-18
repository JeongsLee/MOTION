#!/bin/bash
# Combined 20k evaluation for ONE model (revision R1). Env: TAG, CKPT, MODE (stats|equiv|contrib|all),
# N (stats trajectories, default 128), NE (equiv, default 32), NC (contrib, default 16), plus the
# model's MOTION_* variant switches. Paper ckpts additionally need MOTION_LEGACY_VOCAB=1
# MOTION_VORTEX2D_LEGACY=1. Exact restore: NO_DEAD_REPAIR=1.
set -e
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 NO_DEAD_REPAIR=1
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
cd /tmp/fv2
pip install -q "numpy<2" h5py scipy netCDF4 2>&1 | tail -1
# REVISION-R1: node-decoder 3D inference with K-sample latent-rate averaging (eval_ens3d.py), MOTION-S, test split.
# K=1 reproduces eval_stats; K=8 is the standard 3D protocol applied through the scoring (node) decoder.
export MOTION_LEGACY_VOCAB=1 MOTION_VORTEX2D_LEGACY=1
OUT=/work/runs/ens3d_node; mkdir -p $OUT
python -u eval_ens3d.py --ckpt /corpus/results/motion_m1f/ckpt_best.npz --out $OUT/ens3d_m1f_m10turb.npz --K 1,8 --fams pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08,pdebench3d_cns_Turb_M1.0_Eta1e-08_Zeta1e-08 2>&1 | grep --line-buffered -E "ens3d|ENS3D|Error|error" | tee -a $OUT/log.txt
echo ENS3D_NODE_DONE | tee $OUT/done.txt
