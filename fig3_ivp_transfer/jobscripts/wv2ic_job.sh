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
export LEAD_CAP=12
python - <<'P2IC'
p = "scOT/problems/wave/acoustic.py"
s = open(p).read()
a = 'self.input_dim = 2\n        self.label_description = "[u],[c]"'
s = s.replace(a, 'self.input_dim = 3\n        self.label_description = "[u],[c]"', 1)
old = """        inputs = (inputs - self.constants["mean"]) / self.constants["std"]
        inputs_c = (inputs_c - self.constants["mean_c"]) / self.constants["std_c"]
        labels = (labels - self.constants["mean"]) / self.constants["std"]

        inputs = torch.cat([inputs, inputs_c], dim=0)"""
new = """        t0v = t1 - 1 if t1 >= 1 else t1
        v0 = (
            torch.from_numpy(self.reader["solution"][i + self.start, t1])
            - torch.from_numpy(self.reader["solution"][i + self.start, t0v])
        ).type(torch.float32).reshape(1, self.resolution, self.resolution)
        inputs = (inputs - self.constants["mean"]) / self.constants["std"]
        v0 = v0 / self.constants["std"]
        inputs_c = (inputs_c - self.constants["mean_c"]) / self.constants["std_c"]
        labels = (labels - self.constants["mean"]) / self.constants["std"]

        inputs = torch.cat([inputs, v0, inputs_c], dim=0)"""
assert s.count(old) >= 1
s = s.replace(old, new, 1)
open(p, "w").write(s)
print("2IC PATCH OK")
P2IC
W2=$(B /eu/results/scot_ft_ckpt/Wave-Layer_64_2icx4); echo W2=$W2
export LEAD_CAP=12
for T0 in 0 1 2 4 8; do
  echo "=== Wave-2icx4 T0=$T0 ==="; DATA=/eu/data/poseidon/_scot python -u scot_ourmetric2.py "$W2" wave.Layer $T0 20 0:1 time | tail -1
done
echo WV2SWEEP_DONE
