#!/bin/bash
set -e
export PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=0
pip install -q "huggingface_hub>=0.23" netCDF4 2>&1 | tail -1
echo "--- existing"; ls -la /eu/data/poseidon/_assembled 2>&1 | head -40; df -h /eu | tail -1
mkdir -p /eu/data/poseidon_ft
python -u /work/runs/poseidon_ft_data/fetch_poseidon_ft.py $TASKS 2>&1 | tee -a /eu/data/poseidon_ft/fetch.log
ls -la /eu/data/poseidon_ft/*; du -sh /eu/data/poseidon_ft/*; df -h /eu | tail -1
python -u /work/runs/poseidon_ft_data/verify_poseidon_ft.py 2>&1 | tee /eu/data/poseidon_ft/verify.log
