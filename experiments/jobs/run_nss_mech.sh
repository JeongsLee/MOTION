#!/bin/bash
# REVISION-R1: mechanism metrics of MOTION-NS-S checkpoints (contribution shares -> D/AUC; single-head knockouts on
# the six benchmark families, test-600 = MTF split first 100) for comparison with the published MOTION-NS feature
# masking (A17 mtf_mask2). Env: RUN (results dir), CK (ckpt step), STATS_FAMS (families for the knockouts;
# default all six), MOTION_BANK_PHYSGATE (pass 1 for the physical-gating arm). Extra configs: warp_off, z0 (whole
# latent rate removed: anchor only), ko:generic.
set -e
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
export MOTION_WARP_LIVE=1 MOTION_BANK_NGENERIC=8 MOTION_BANK_GENERIC_MERGE=1 MOTION_SPLIT_MTF=1 MOTION_BOX_UNIFORM=64
RUN=${RUN:-motion_nss_lr1e4}; CK=${CK:-50000}
OUT=/work/runs/nss_mech/${RUN}_${CK}; mkdir -p $OUT
mkdir -p /tmp/corpus/cache /tmp/corpus/raw
ln -s /corpus/meta /tmp/corpus/meta; ln -s /eu/data/prose/prebuilt /tmp/corpus/cache/prose128
export CORPUS_ROOT=/tmp/corpus
cp -r /work/MOTION/motion_tf /tmp/fv2; cd /tmp/fv2
pip install -q "numpy<2" h5py scipy netCDF4 2>&1 | tail -1
FAMS=shallow_water,com_ns,incom_ns,pdearena_ns,cfdbench,pdearena_uncond
C=/corpus/results/$RUN/ckpt_$CK.npz; ls -la $C
python -u eval_contrib.py --ckpt $C --out $OUT/contrib.json --n 16 --fams $FAMS 2>&1 | grep -v graph_launch | tee $OUT/contrib.log
HEADS=reaction,diffusion,buoyancy,shear,wave,advection,shock,elliptic,geometry,boundary_layer,dilatation,gradvec,bernoulli,kinematic,vortex,reaction_nl,spectral,coefficient,compressible
CFG="base$(for h in ${HEADS//,/ }; do printf ',ko:%s' $h; done)"
env | grep -E "^MOTION_" | tee $OUT/env.txt
python -u eval_stats.py --ckpt $C --out $OUT/stats.npz --split test --n 100 --batch 8 --xla 1 --bf16 1 --fams ${STATS_FAMS:-$FAMS} \
  --configs $CFG,ko:generic,warp_off,z0 2>&1 | grep -v graph_launch | tee $OUT/stats.log
echo NSS_MECH_DONE | tee $OUT/done.txt
