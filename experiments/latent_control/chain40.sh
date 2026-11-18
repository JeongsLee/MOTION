#!/bin/bash
# Local GPU chain 40 (user 10-22): does the incom_ns latent-velocity response fade with training in the 6-family NS-S runs?
# Early checkpoints of the same architecture / families (z-score lineage: zs 5k, 15k -> zsc 160k; std-only lineage:
# fl1 20k, pg 20k -> lr1e4 160k), uniform command K=1, same alphas, incom_ns / com_ns / pdearena_ns, MTF test units.
L=${MOTION_LOCAL:-motion_local}; O=$L/out_trend; mkdir -p $O
exec 9>$L/chain.lock; flock 9
cd ${MOTION_REPO:-.}/motion_tf
export TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1 CORPUS_ROOT=/tmp MOTION_WARP_LIVE=1 MOTION_BANK_NGENERIC=8 \
  MOTION_BANK_GENERIC_MERGE=1 MOTION_BOX_UNIFORM=64
PY=python; CAP="systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0"
U=$L/zsc/units_nss_mtf.pkl
for spec in "zs5k $L/early/motion_nss_zs_ckpt_5000.npz ZS=1" "zs15k $L/early/motion_nss_zs_ckpt_15000.npz ZS=1" \
            "zsc160k $L/zsc/zsc_ckpt_160000.npz ZS=1,CS=1" \
            "fl1_20k $L/early/motion_nss_fl1_ckpt_20000.npz -" "pg20k $L/early/motion_nss_pg_ckpt_20000.npz PG=1" \
            "lr1e4_160k $L/lr1e4/lr1e4_ckpt_160000.npz -"; do
  set -- $spec; tag=$1; C=$2; fl=$3
  E=""; [[ $fl == *ZS=1* ]] && E="$E MOTION_INPUT_ZSCORE=1"; [[ $fl == *CS=1* ]] && E="$E MOTION_COND_STATS=1"; [[ $fl == *PG=1* ]] && E="$E MOTION_BANK_PHYSGATE=1"
  for fam in incom_ns com_ns pdearena_ns; do
    env $E LOCAL_UNITS=$U $CAP $PY -u latent_control_k.py --ckpt $C --fams $fam --K 1 --calib_alphas=-4,-1,1,4 \
      --alphas=-4,-1,1,4 --out $O/k_${tag}_${fam}.npz > $O/k_${tag}_${fam}.log 2>&1
    echo "== $tag $fam ($E)"; grep -E "calib|R2|axis0 .*alpha [-+]4\.00|Traceback" $O/k_${tag}_${fam}.log | cut -c1-170
  done
done
echo CHAIN40_DONE
