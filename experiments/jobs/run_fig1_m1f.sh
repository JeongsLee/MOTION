#!/bin/bash
# Fig. 1 panel data for MOTION-S (m1f) on the TEST split (revision: representative model = MOTION-S; R1 #5 / A02).
# b (GT vs prediction, lead 5), d (pdearena spectra under knockout), e (CE spatial structure under knockout).
# Panel c (knockout matrix) comes from ev20-m1f-stats (test, n=128) instead of this dump's small-n knockout.
set -e
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 NO_DEAD_REPAIR=1 MOTION_LEGACY_VOCAB=1 MOTION_VORTEX2D_LEGACY=1 FIG_SPLIT=test FIG_LEAD=5
OUT=/work/runs/fig1_m1f; mkdir -p $OUT
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
CK=/corpus/results/motion_m1f/ckpt_best.npz
python -u fig1_dump.py $CK $OUT 2>&1 | grep --line-buffered -vE "graph_launch" | tee $OUT/dump.log
ONLY_KO=1 KO_FAMS=pdearena_ns,incom_ns KO_N=16 KO_OUT=fig1d_ko python -u fig1_dump.py $CK $OUT 2>&1 | grep --line-buffered -vE "graph_launch" | tee $OUT/ko_d.log
ONLY_KO=1 KO_FAMS=CE-RM,incom_ns,CE-CRP KO_HEADS=shock,advection,compressible KO_N=16 KO_OUT=fig1e_ko python -u fig1_dump.py $CK $OUT 2>&1 | grep --line-buffered -vE "graph_launch" | tee $OUT/ko_e.log
echo FIG1_DONE | tee $OUT/done.txt
