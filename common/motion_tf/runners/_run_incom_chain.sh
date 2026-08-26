#!/bin/bash
# AUTO-CHAIN for Phase-1 Task-2 data: wait until the incom download (run by the separate dl job) has
# populated all source .h5, THEN rebuild the incom_ns dt=1 sliding-window shards (rmtree old + all 274
# files). No kill / no race: this job only READS the file count while the dl job writes, then rebuilds
# once the count is complete (>=272 of ~274) or stable for 1h.
set -e
export PYTHONUNBUFFERED=1
CV="${CODE_VOL:-/code-vol}"
DST="$CV/data/prose/pdebench/2D/NS_incom"
echo "=== waiting for incom download to finish (poll $DST) ==="
prev=-1; stall=0
while true; do
  n=$(find "$DST" -name "ns_incom_inhom_2d_512-*.h5" -size +9G 2>/dev/null | wc -l)
  echo "  [wait $(date -u +%H:%M)] $n files >9G (target ~274)"
  if [ "$n" -ge 272 ]; then echo "  -> download complete ($n)"; break; fi
  if [ "$n" -eq "$prev" ]; then stall=$((stall+1)); else stall=0; fi
  if [ "$stall" -ge 6 ]; then echo "  -> stable at $n for ~1h, proceeding"; break; fi
  prev=$n; sleep 600
done
echo "=== rebuild incom_ns dt=1 shards (rmtree old + all .h5) ==="
mkdir -p /tmp/work && cd /tmp/work && cp -r "$CV"/code code/
pip install -q numpy h5py sympy 2>&1 | tail -1
export PROSE_DATA_DIR="$CV/data/prose" PREBUILT_DIR="$CV/data/prose/prebuilt"
PYTHONPATH=/tmp/work FAMILIES=incom_ns NPER=100000 TNUM=20 python "$CV/code/runners/_rebuild_dt1.py" 2>&1
echo "--- new incom shard count ---"
ls "$CV/data/prose/prebuilt/incom_ns_n100000_t20_shards/" 2>/dev/null | grep -c shard_ || true
echo "INCOM_CHAIN_DONE"
