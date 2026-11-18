#!/bin/bash
# REVISION-R1 11-06: one-trajectory CFDBench row mirror (fr_ns) for MOTION-NS (eval_equiv.py, first 16 test
# trajectories, last-frame dumps of id and fr_ns) and BCAT / PROSE-FD (cfd_onetraj.py from the /eu dumps). CPU only.
set -e -o pipefail
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
OUT=/work/runs/cfd_onetraj; mkdir -p $OUT
pip install -q "numpy<2" h5py scipy netCDF4 2>&1 | tail -1
python -u /work/MOTION/motion_tf/analysis/cfd_onetraj.py /eu $OUT/cfd_base.npz 2>&1 | tee $OUT/cfd_base.log
export EQUIV_DUMP_MEMBERS=fr_ns
export MOTION_WARP_LIVE=1 MOTION_BANK_NGENERIC=8 MOTION_BANK_GENERIC_MERGE=1 MOTION_SPLIT_MTF=1 MOTION_BOX_UNIFORM=64
RUN=motion_nss_zsc; CK=160000; export MOTION_INPUT_ZSCORE=1 MOTION_COND_STATS=1 MOTION_EQUIV_VA1=grid_npy:0
mkdir -p /tmp/corpus/cache /tmp/corpus/raw
ln -s /corpus/meta /tmp/corpus/meta; ln -s /eu/data/prose/prebuilt /tmp/corpus/cache/prose128
ln -s /eu/data/poseidon/_assembled /tmp/corpus/raw/poseidon_assembled; ln -s /work/data/poseidon /tmp/corpus/raw/poseidon
ln -s /corpus/raw/pdebench /tmp/corpus/raw/pdebench; ln -s /corpus/cache/unified /tmp/corpus/cache/unified
export CORPUS_ROOT=/tmp/corpus
cp -r /work/MOTION/motion_tf /tmp/fv2; cd /tmp/fv2
C=/corpus/results/$RUN/ckpt_$CK.npz; ls -la $C
python -u eval_equiv.py --ckpt $C --out $OUT/equiv.npz --split test --n 16 --batch 8 --xla 0 \
  --fams cfdbench --members d4 --dump 16 2>&1 | grep --line-buffered -v graph_launch | tee $OUT/equiv.log
echo CFD_ONETRAJ_DONE | tee $OUT/done.txt
