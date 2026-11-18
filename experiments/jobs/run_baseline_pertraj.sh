#!/bin/bash
# REVISION-R1 (10-22): per-trajectory BCAT / PROSE-FD equivariance metrics (test-600) from the /eu symmetry dumps, for paired
# tests against MOTION-NS (03_analyses/A01_swe_symmetry_bug/replica_all/baseline_pertraj.py). CPU only.
set -e -o pipefail
export PYTHONUNBUFFERED=1
OUT=/work/runs/baseline_pertraj; mkdir -p $OUT
pip install -q "numpy<2" 2>&1 | tail -1
ls /eu/_prosesymE_sh1_* /eu/_prosesymE_sh8_* 2>&1 | tee $OUT/prose_shift_dumps.txt
python -u /work/MOTION/motion_tf/analysis/baseline_pertraj.py /eu $OUT/baseline_pertraj.json 2>&1 | tee $OUT/baseline_pertraj.log
echo BASELINE_PERTRAJ_DONE | tee $OUT/done.txt
