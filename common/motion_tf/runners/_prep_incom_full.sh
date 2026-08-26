#!/bin/bash
# Rebuild the FULL incom_ns cache from ALL raw .h5 files (the cached shards were stale: ~200 traj vs
# ~356 now available). Deletes the stale shard dir and rebuilds from scratch (clean shard indexing).
# Run on deneb-kr CPU (resourcespec-grlxx3knwzps) — co-located with the Seoul data, so raw reads are fast.
set -e
export PYTHONUNBUFFERED=1
CODE_VOL="${CODE_VOL:-/code-vol}"
STALE="$CODE_VOL/data/prose/prebuilt/incom_ns_n100000_t20_shards"
pip install -q numpy h5py 2>/dev/null || true
cd /tmp && rm -rf work && mkdir work && cd work && cp -r "$CODE_VOL"/code .
echo "=== raw incom files available ==="
ls -1 "$CODE_VOL"/data/prose/pdebench/2D/NS_incom/ns_incom_inhom_2d_512-*.h5 2>/dev/null | wc -l
echo "=== removing stale incom cache: $STALE ==="
rm -rf "$STALE"
echo "=== rebuilding incom full (n_per=100000 = ALL) ==="
PROSE_DATA_DIR="$CODE_VOL/data/prose" python -u -m motion_tf.data.prep_multi --datasets incom_ns --n_per 100000
echo "=== result ==="
echo "shards: $(ls -1 "$STALE"/shard_*.npy 2>/dev/null | wc -l)   _DONE=$(cat "$STALE/_DONE" 2>/dev/null) traj"
echo "=== INCOM FULL PREP DONE ==="
