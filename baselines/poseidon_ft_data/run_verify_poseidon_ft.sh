#!/bin/bash
set -e
export PYTHONUNBUFFERED=1
pip install -q "huggingface_hub>=0.23" netCDF4 h5py 2>&1 | tail -1
python -u /work/runs/poseidon_ft_data/verify_poseidon_ft.py 2>&1 | tee /eu/data/poseidon_ft/verify.log
du -shL /eu/data/poseidon_ft/*
