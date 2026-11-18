#!/bin/bash
# REVISION-R1: local control experiments on MOTION-NS-S (zsc): copy the final checkpoint to the code volume and export
# the first 8 TEST units (MTF split, i.e. held out from NS-S training) of the joint-benchmark families.
set -e
export PYTHONUNBUFFERED=1 MOTION_SPLIT_MTF=1
OUT=/work/runs/zsc_local; mkdir -p $OUT
mkdir -p /tmp/corpus/cache /tmp/corpus/raw
ln -s /corpus/meta /tmp/corpus/meta; ln -s /eu/data/prose/prebuilt /tmp/corpus/cache/prose128
export CORPUS_ROOT=/tmp/corpus
cp -r /work/MOTION/motion_tf /tmp/fv2; cd /tmp/fv2
pip install -q "numpy<2" h5py scipy netCDF4 2>&1 | tail -1
cp /corpus/results/motion_nss_zsc/ckpt_160000.npz $OUT/zsc_ckpt_160000.npz; ls -la $OUT
python -u export_units.py $OUT/units_nss_mtf.pkl --fams incom_ns,com_ns,shallow_water,pdearena_ns,cfdbench --n 8 2>&1 | tee $OUT/export.log
echo ZSC_EXPORT_DONE | tee $OUT/done.txt
