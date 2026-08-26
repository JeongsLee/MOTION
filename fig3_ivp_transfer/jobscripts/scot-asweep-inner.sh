set -e
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled WANDB_DISABLED=true HF_HUB_OFFLINE=1
which git >/dev/null 2>&1 || (apt-get update -q && apt-get install -y -q git) >/dev/null 2>&1
cd /tmp && rm -rf po && git clone --depth 1 https://github.com/camlab-ethz/poseidon po 2>&1 | tail -1
cd /tmp/po
pip install -q -e . --no-deps 2>&1 | tail -1
pip install -q transformers==4.29.2 accelerate==0.31.0 "huggingface_hub<0.20" wandb einops timm netCDF4 h5py pandas pyyaml matplotlib safetensors psutil 2>&1 | tail -1
python - <<'P1'
import re
for pyf, oldmax, nmax in [("scOT/problems/fluids/incompressible.py","20000","2354"),
                          ("scOT/problems/reaction_diffusion/allen_cahn.py","15000","3750"),
                          ("scOT/problems/wave/acoustic.py","10512","3504")]:
    s = open(pyf).read()
    s = s.replace(f"{oldmax},", f"{nmax},")
    s = re.sub(r"self\.N_max\s*=\s*\d+", f"self.N_max = {nmax}", s)
    s = re.sub(r"self\.N_val\s*=\s*\d+", "self.N_val = 64", s)
    s = re.sub(r"self\.N_test\s*=\s*\d+", "self.N_test = 128", s)
    open(pyf, "w").write(s)
print("SPLIT PATCHES OK")
P1
cp /tmp/scot_ourmetric2.py .
B() { CJ=$(find "$1" -name config.json 2>/dev/null | grep -v checkpoint- | head -1); [ -z "$CJ" ] && CJ=$(find "$1" -path "*checkpoint-*/config.json" 2>/dev/null | sort -t- -k2 -n | tail -1); dirname "$CJ"; }
NS=$(B /eu/results/scot_ft_ckpt/NS-PwC_64_velx4); echo NS=$NS
WV=$(B /eu/results/scot_ft_ckpt/Wave-Layer_64_x4); echo WV=$WV
AC=$(B /eu/results/scot_ft_ckpt/ACE_64); echo AC=$AC
for T0 in 0 1 2 4 8; do
  echo "=== NS-x4 T0=$T0 ==="; DATA=/eu/data/poseidon/_assembled python -u scot_ourmetric2.py "$NS" fluids.incompressible.PiecewiseConstants $T0 20 0:2 time '{"just_velocities": true}' | tail -1
  echo "=== Wave-x4b T0=$T0 ==="; DATA=/eu/data/poseidon/_scot python -u scot_ourmetric2.py "$WV" wave.Layer $T0 20 0:1 time | tail -1
  echo "=== ACE@3200 T0=$T0 ==="; DATA=/eu/data/poseidon/_assembled python -u scot_ourmetric2.py "$AC" reaction_diffusion.AllenCahn $T0 19 0:1 time | tail -1
done
echo ASWEEP_DONE
