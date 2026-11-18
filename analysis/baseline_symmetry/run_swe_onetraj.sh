#!/bin/bash
# REVISION-R1 11-03: one-trajectory inverse-transformed baseline predictions (shallow water), CPU only.
set -e -o pipefail
export PYTHONUNBUFFERED=1
OUT=/work/runs/swe_onetraj; mkdir -p $OUT
pip install -q "numpy<2" 2>&1 | tail -1
python -u /work/MOTION/motion_tf/analysis/swe_onetraj.py /eu $OUT/swe_onetraj.npz 2>&1 | tee $OUT/swe_onetraj.log
echo SWE_ONETRAJ_DONE | tee $OUT/done.txt
