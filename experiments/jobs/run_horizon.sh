#!/bin/bash
# Horizon / history-OOD test (revision R1-1, decoder ablation). Env: TAG, CKPT, N (default 64), model MOTION_* switches.
# Trained with t_in=10 observed frames + 10-step rollout. Evaluates (t_obs, k_fut) in {(10,10),(5,15),(2,18)}
# on autoregressive families, plus "z0" (readout input zeroed: IC-consistency of the decoder) at (10,10).
set -e
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 NO_DEAD_REPAIR=1
N=${N:-64}
OUT=/work/runs/horizon/$TAG; mkdir -p $OUT
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
env | grep -E "^MOTION_" | tee $OUT/env.txt || true
CFGS=${HZ_CFGS:-"10_10_base,z0 5_15_base 2_18_base"}
for cfg in $CFGS; do
  cfg=${cfg//_/ }
  set -- $cfg
  echo "=== t_obs=$1 k_fut=$2 ($3)"
  python -u eval_stats.py --ckpt $CKPT --out $OUT/h$1_k$2.npz --split test --n $N --batch 8 --batch3d 2 --xla 1 \
    --t_obs $1 --k_fut $2 --ar_only 1 --skip_ss ${SKIP_SS:-0} --configs $3 2>&1 | grep --line-buffered -vE "graph_launch" | tee $OUT/h$1_k$2.log
done
echo HORIZON_DONE | tee $OUT/done.txt
