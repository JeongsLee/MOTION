#!/bin/bash
# 1) build the pretraining-family eval files (CPU part), 2) evaluate the PUBLIC Poseidon-T on them with scOT's dataset
# classes + eval_scot_ft15.py (val/test, direct/AR, KF 20, step 2).
set -e
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled HF_HUB_OFFLINE=0
(apt-get update -q && apt-get install -y -q git) >/dev/null 2>&1
pip install -q "huggingface_hub>=0.23" netCDF4 h5py 2>&1 | tail -1
for f in NS-Sines NS-Gauss CE-RP CE-CRP CE-KH CE-Gauss; do [ -f /eu/data/poseidon_ft/pre/$f.nc ] || NEED=1; done
[ "${NEED:-0}" = 1 ] && python -u /work/runs/poseidon_ft_data/prep_pre6.py
cd /tmp && rm -rf po && git clone --depth 1 https://github.com/camlab-ethz/poseidon po 2>&1 | tail -1; cd /tmp/po
pip install -q -e . --no-deps 2>&1 | tail -1
pip install -q transformers==4.29.2 accelerate==0.31.0 "huggingface_hub<0.20" wandb einops timm netCDF4 h5py psutil pandas safetensors matplotlib pyyaml 2>&1 | tail -1
python - <<'P'
import re
for p, old in (("scOT/problems/fluids/incompressible.py", "20000,"), ("scOT/problems/fluids/compressible.py", None)):
    s = open(p).read()
    if old: s = s.replace(old, "361,")
    s = re.sub(r"self\.N_max\s*=\s*10000", "self.N_max = 361", s); s = re.sub(r"self\.N_max\s*=\s*20000", "self.N_max = 361", s)
    open(p, "w").write(s)
P
grep -n "361" scOT/problems/fluids/incompressible.py scOT/problems/fluids/compressible.py | head
cp /work/runs/poseidon_ft_data/eval_scot_ft15.py .
OUT=/work/runs/ft15/pre6_poseidonT; mkdir -p $OUT
declare -A DSM=([NS-Sines]=fluids.incompressible.Sines [NS-Gauss]=fluids.incompressible.Gaussians [CE-RP]=fluids.compressible.Riemann [CE-CRP]=fluids.compressible.RiemannCurved [CE-KH]=fluids.compressible.KelvinHelmholtz [CE-Gauss]=fluids.compressible.Gaussians)
for t in NS-Sines NS-Gauss CE-RP CE-CRP CE-KH CE-Gauss; do for SP in val test; do for MO in direct ar; do
  TASK=$t MODEL=/work/models/Poseidon-T DS=${DSM[$t]} DATA=/eu/data/poseidon_ft/pre KF=20 STEP=2 SPLIT=$SP MODE=$MO \
    OUT=$OUT/${t}_${SP}_${MO}.json python -u eval_scot_ft15.py 2>&1 | grep -E "SCOT_FT15|lead (1|20):|Error|Traceback" | tee -a $OUT/score.log
done; done; done
echo PRE6_DONE
