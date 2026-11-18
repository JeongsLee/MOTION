#!/bin/bash
# Limited-sharing test (V2 question): group knockouts, within-group vs across-group pairwise knockouts, singles.
# Groups fixed a priori by operator class. Env: TAG, CKPT, MOTION_* switches.
set -e
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 NO_DEAD_REPAIR=1
OUT=/work/runs/eval20k/$TAG; mkdir -p $OUT
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
SINGLES="diffusion wave advection shock dilatation gradvec shear buoyancy reaction reaction_nl elliptic spectral coefficient geometry compressible"
CFG=base; for h in $SINGLES; do CFG="$CFG,ko:$h"; done
GRPS="diffusion+wave advection+shock+dilatation gradvec+shear+buoyancy reaction+reaction_nl+elliptic spectral+coefficient geometry+compressible"
WITHIN="diffusion+wave advection+shock advection+dilatation shock+dilatation gradvec+shear gradvec+buoyancy shear+buoyancy reaction+reaction_nl spectral+coefficient"
ACROSS="diffusion+advection wave+gradvec shock+reaction diffusion+gradvec advection+reaction_nl shear+spectral wave+shock dilatation+coefficient"
for g in $GRPS $WITHIN $ACROSS; do CFG="$CFG,ko:$g"; done
CFG=$(echo $CFG | tr ',' '\n' | awk '!s[$0]++' | paste -sd,)
FAMS=$(python -c "from data.registry import FAMILIES,MOTION_FAMILIES;print(','.join(f for f in MOTION_FAMILIES if FAMILIES[f].K==2))")
echo "=== groups ($CFG)"
python -u eval_stats.py --ckpt $CKPT --out $OUT/groupko.npz --split test --n 64 --batch 8 --xla 1 --fams $FAMS --configs $CFG \
  2>&1 | grep --line-buffered -vE "graph_launch" | tee $OUT/groupko.log
echo GROUPS_DONE | tee $OUT/done_groups.txt
