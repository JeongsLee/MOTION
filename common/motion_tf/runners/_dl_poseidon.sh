#!/bin/bash
# Stage-2 prep: pre-download Poseidon code + Poseidon-B weights + downstream datasets to the volume, using
# python huggingface_hub.snapshot_download (robust; the huggingface-cli was renamed to `hf`). Reassembles
# chunked NetCDF via the repo's assemble_data.py if present. env: DSETS (space-sep downstream task names).
set -e
export PYTHONUNBUFFERED=1
CODE_VOL="${CODE_VOL:-/code-vol}"
OUT="$CODE_VOL/data/poseidon"; mkdir -p "$OUT" "$CODE_VOL/models"
which git >/dev/null 2>&1 || (apt-get update -q && apt-get install -y -q git) >/dev/null 2>&1
pip install -q huggingface_hub 2>&1 | tail -1
cd /tmp && rm -rf poseidon && git clone --depth 1 https://github.com/camlab-ethz/poseidon 2>&1 | tail -1
ASM=$(find /tmp/poseidon -iname "assemble*data*.py" 2>/dev/null | head -1); echo "assemble script: ${ASM:-<none found>}"

echo "=== Poseidon-B weights ==="
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('camlab-ethz/Poseidon-B', local_dir='$CODE_VOL/models/Poseidon-B', max_workers=4))"

DSETS="${DSETS:-NS-PwC NS-SL Wave-Layer ACE CE-RM Poisson-Gauss}"
for ds in $DSETS; do
  d="$OUT/$ds"
  echo "=== downstream: $ds -> $d ==="
  python -c "from huggingface_hub import snapshot_download; snapshot_download('camlab-ethz/$ds', repo_type='dataset', local_dir='$d', max_workers=4)" 2>&1 | tail -2
  # if no single .nc and we have an assemble script + chunk files, reassemble
  if ! ls "$d"/*.nc >/dev/null 2>&1; then
    AS="$ASM"; [ -z "$AS" ] && AS=$(find "$d" -iname "assemble*data*.py" 2>/dev/null | head -1)
    [ -n "$AS" ] && (cd "$d" && python "$AS" --input_dir . --output_file "$ds.nc" 2>&1 | tail -2) || echo "  (no assemble script; left as-is)"
  fi
  echo "  $ds files: $(ls "$d" 2>/dev/null | head -5 | tr '\n' ' ') | size: $(du -sh "$d" 2>/dev/null | cut -f1)"
done
echo "POSEIDON_DL_DONE"
