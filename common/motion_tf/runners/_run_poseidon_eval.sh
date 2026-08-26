#!/bin/bash
# Baseline: evaluate a pretrained Poseidon model (default Poseidon-B) on a PDEgym test set via scOT's own
# inference (median relative-L1). Gives the in-distribution target our pretraining should approach.
# env: MODEL (Poseidon-B|T|L), DS (scOT dataset id), NAME (output tag), FT (final_time).
set -e
export PYTHONUNBUFFERED=1
CV="${CODE_VOL:-/code-vol}"
which git >/dev/null 2>&1 || (apt-get update -q && apt-get install -y -q git) >/dev/null 2>&1
cd /tmp && rm -rf po && git clone --depth 1 https://github.com/camlab-ethz/poseidon po 2>&1 | tail -1
cd /tmp/po
python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'sm',torch.cuda.get_device_capability())" 2>&1 | tail -1
# scOT pins torch==2.0.1 which has NO sm_90 kernels -> 'no kernel image' on H100. Install scOT WITHOUT deps
# (keep the image's H100-capable torch) and add only its non-torch deps at scOT's pinned versions.
pip install -q -e . --no-deps 2>&1 | tail -1
pip install -q transformers==4.29.2 accelerate==0.31.0 "huggingface_hub<0.20" wandb einops timm netCDF4 h5py pandas pyyaml matplotlib safetensors 2>&1 | tail -3
export WANDB_MODE=disabled
python -c "import torch;print('after-deps torch',torch.__version__,torch.cuda.get_device_capability())" 2>&1 | tail -1
MODEL="${MODEL:-Poseidon-B}"; FT="${FT:-20}"
OUT="$CV/results/poseidon_eval"; mkdir -p "$OUT"
# PAIRS = "NAME:scOT_dataset_id" space-separated; default = confidently-known datasets across both families.
PAIRS="${PAIRS:-NS-Sines:fluids.incompressible.Sines CE-Gauss:fluids.compressible.Gaussians CE-KH:fluids.compressible.KelvinHelmholtz}"
for p in $PAIRS; do
  NAME="${p%%:*}"; DS="${p##*:}"
  echo "=================== EVAL $MODEL on $NAME ($DS) final_time=$FT ==================="
  python -m scOT.inference --mode eval \
    --model_path "$CV/models/$MODEL" \
    --dataset "$DS" \
    --data_path "${DATA_PATH:-$CV/data/poseidon/_assembled}" \
    --file "$OUT/${NAME}_${MODEL}.csv" \
    --ckpt_dir /tmp/ck \
    --initial_time 0 --final_time "$FT" --ar_steps "${AR:-1}" 2>&1
  echo "--- $NAME rc=$? ---"
done
echo "POSEIDON_EVAL_DONE"
