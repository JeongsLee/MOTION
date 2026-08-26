#!/bin/bash
# Compact checkpoint inventory: one line per results dir (ckpt count / latest / total size), plus a
# search for baseline (torch .pt/.pth) and MPP checkpoints anywhere on the volume, plus the prose dump.
R=/code-vol/results
echo "=== TF .npz checkpoints per results dir (OURS) ==="
for d in "$R"/*/; do
  n=$(ls "$d"ckpt_*.npz 2>/dev/null | wc -l)
  [ "$n" -eq 0 ] && continue
  last=$(ls "$d"ckpt_*.npz 2>/dev/null | sed -E 's/.*ckpt_([0-9]+)\.npz/\1/' | sort -n | tail -1)
  sz=$(du -sh "$d" 2>/dev/null | cut -f1)
  echo "  $(basename "$d"): $n ckpts, latest=$last, $sz"
done
echo ""
echo "=== dirs with NO npz ckpt (baselines / eval-only) ==="
for d in "$R"/*/; do
  n=$(ls "$d"ckpt_*.npz 2>/dev/null | wc -l)
  [ "$n" -eq 0 ] && echo "  $(basename "$d"): $(ls "$d" 2>/dev/null | wc -l) files, $(du -sh "$d" 2>/dev/null|cut -f1)"
done
echo ""
echo "=== ANY torch checkpoints on the whole volume (baseline weights)? ==="
find /code-vol -maxdepth 6 -type f \( -iname "*.pt" -o -iname "*.pth" -o -iname "*.ckpt" -o -iname "checkpoint*" \) 2>/dev/null | grep -ivE "models/Poseidon" | head -30
echo "  (none above = no baseline torch ckpts persisted)"
echo ""
echo "=== MPP / wandb / exp dirs anywhere ==="
find /code-vol -maxdepth 4 -type d \( -iname "*mpp*" -o -iname "*exp*" -o -iname "wandb" -o -iname "checkpoints" \) 2>/dev/null | grep -ivE "Poseidon" | head
echo ""
echo "=== prose_comns_dump.npz persisted? ==="
ls -la /code-vol/results/prose_comns_dump.npz 2>&1
echo INVENTORY_DONE
