#!/bin/bash
# Inventory everything on the volume that could carry a baseline's eval-vs-samples trajectory:
# result dirs, log/json/csv/yaml files, tensorboard event files, and checkpoint intervals.
R=/code-vol/results
echo "=== top-level result dirs ==="
ls -la "$R" 2>/dev/null
for d in "$R"/*/; do
  [ -d "$d" ] || continue
  echo ""
  echo "############ $d ############"
  echo "-- text/metric files --"
  find "$d" -maxdepth 4 -type f \( -name "*.log" -o -name "*.json" -o -name "*.csv" -o -name "*.txt" -o -name "*.yaml" -o -name "*.out" \) 2>/dev/null | head -25
  echo "-- tensorboard --"
  find "$d" -maxdepth 4 -type f -name "events.out*" 2>/dev/null | head -8
  echo "-- checkpoints (interval?) --"
  find "$d" -maxdepth 4 -type f \( -name "*.pt" -o -name "*.pth" -o -name "*ckpt*" -o -name "*.index" -o -name "*.h5" \) 2>/dev/null | sort | head -20
done
echo ""
echo "=== grep eval/rel-l2 from any .log files found ==="
for f in $(find "$R" -maxdepth 4 -type f -name "*.log" 2>/dev/null); do
  echo "------ $f ------"
  grep -iE "rel.?l2|eval|epoch [0-9]+|step [0-9]+|loss" "$f" 2>/dev/null | tail -25
done
echo "INSPECT_RESULTS_DONE"
