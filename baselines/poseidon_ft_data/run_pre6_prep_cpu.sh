#!/bin/bash
# REVISION-R1 10-24: CPU part of the pretraining-family check: build the pre6 eval files (/eu/data/poseidon_ft/pre;
# needs huggingface_hub>=0.23, which the scOT env pins below 0.20 -> diag-h100 failed) and the held-out-by-both lists.
set -e
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=0
pip install -q "huggingface_hub>=0.23" netCDF4 h5py 2>&1 | tail -1
python -u /work/runs/poseidon_ft_data/prep_pre6.py 2>&1 | tail -10
ls -la /eu/data/poseidon_ft/pre
python -u /work/runs/poseidon_ft_data/heldout_both.py
