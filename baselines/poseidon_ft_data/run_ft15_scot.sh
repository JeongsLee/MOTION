#!/bin/bash
# REVISION-R1 (user 2026-10-23): fine-tune Poseidon-T (scOT) on one FT15 task with the protocol of our earlier matched
# Poseidon runs (code/runners/_run_scot_ft.sh), on the SAME rows as MOTION-IVP-S: data = /eu/data/poseidon_ft/ft/<task>
# (rows = first 256 train + official val + official test); the dataset class is patched to N_max = rows, N_val/N_test =
# official, so scOT's own split arithmetic lands on those rows. 64 training trajectories, 3,200 optimizer steps x batch
# 40, lr 5e-5 cosine (scOT FT recipe), checkpoint selected on the official val split by scOT's trainer. Time config:
# all2all over leads up to the evaluated horizon (step 2 for even horizons, 1 otherwise). Scoring: eval_scot_ft15.py
# (MOTION's physical rel-L1 metric), direct and AR, on val and test.
# Env: TASK, SMOKE (1 = 2 steps, 4 eval trajectories, CPU), DATA (default /eu/data/poseidon_ft/ft).
set -e
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled WANDB_DISABLED=true HF_HUB_OFFLINE=1
CV=/code-vol; DATA=${DATA:-/eu/data/poseidon_ft/ft}; TASK=${TASK:?TASK}; N=64
# task -> scOT dataset id, problem file, original N_max in that file, rows, N_val, N_test, horizon, train step
case "$TASK" in
  NS-PwC)        DS=fluids.incompressible.PiecewiseConstants;        PYF=fluids/incompressible.py; OLD=20000; NV=120; NT=240; KF=20; ST=2;;
  NS-Tracer-PwC) DS=fluids.incompressible.PiecewiseConstants.tracer; PYF=fluids/incompressible.py; OLD=20000; NV=120; NT=240; KF=20; ST=2;;
  NS-SVS)        DS=fluids.incompressible.VortexSheet;               PYF=fluids/incompressible.py; OLD=20000; NV=120; NT=240; KF=20; ST=2;;
  NS-BB)         DS=fluids.incompressible.BrownianBridge;            PYF=fluids/incompressible.py; OLD=20000; NV=120; NT=240; KF=20; ST=2;;
  NS-SL)         DS=fluids.incompressible.ShearLayer;                PYF=fluids/incompressible.py; OLD=40000; NV=120; NT=240; KF=20; ST=2;;
  FNS-KF)        DS=fluids.incompressible.forcing.KolmogorovFlow;    PYF=fluids/incompressible.py; OLD=20000; NV=120; NT=240; KF=20; ST=2;;
  CE-RPUI)       DS=fluids.compressible.RiemannKelvinHelmholtz;      PYF=fluids/compressible.py;   OLD=10000; NV=120; NT=240; KF=20; ST=2;;
  CE-RM)         DS=fluids.compressible.RichtmyerMeshkov;            PYF=fluids/compressible.py;   OLD=1260;  NV=100; NT=130; KF=20; ST=2;;
  GCE-RT)        DS=fluids.compressible.gravity.RayleighTaylor;      PYF=fluids/compressible.py;   OLD=1260;  NV=100; NT=130; KF=10; ST=2;;
  Wave-Gauss)    DS=wave.Gaussians;                                  PYF=wave/acoustic.py;         OLD=10512; NV=60;  NT=240; KF=15; ST=1;;
  Wave-Layer)    DS=wave.Layer;                                      PYF=wave/acoustic.py;         OLD=10512; NV=60;  NT=240; KF=20; ST=2;;
  ACE)           DS=reaction_diffusion.AllenCahn;                    PYF=reaction_diffusion/allen_cahn.py; OLD=15000; NV=60; NT=240; KF=19; ST=1;;
  Poisson-Gauss) DS=elliptic.poisson.Gaussians.time;                 PYF=elliptic/poisson.py;      OLD=20000; NV=120; NT=240; KF=1; ST=1;;
  Helmholtz)     DS=elliptic.Helmholtz.time;                         PYF=elliptic/helmholtz.py;    OLD=19675; NV=128; NT=512; KF=1; ST=1;;
  SE-AF)         DS=fluids.compressible.steady.Airfoil.time;         PYF=fluids/compressible.py;   OLD=10869; NV=120; NT=240; KF=1; ST=1;;
  *) echo "unknown TASK $TASK"; exit 2;;
esac
FB=$TASK; [ "$TASK" = NS-Tracer-PwC ] && FB=NS-PwC          # NS-Tracer-PwC reads the NS-PwC file
ROWS=$(python3 -c "import json;print(json.load(open('$DATA/$FB.json'))['rows'])")
which git >/dev/null 2>&1 || (apt-get update -q && apt-get install -y -q git) >/dev/null 2>&1
cd /tmp && rm -rf po && git clone --depth 1 https://github.com/camlab-ethz/poseidon po 2>&1 | tail -1
cd /tmp/po && pip install -q -e . --no-deps 2>&1 | tail -1
pip install -q transformers==4.29.2 accelerate==0.31.0 "huggingface_hub<0.20" wandb einops timm netCDF4 h5py pandas \
  pyyaml matplotlib safetensors psutil 2>&1 | tail -2
python - "scOT/problems/$PYF" "$OLD" "$ROWS" "$NV" "$NT" <<'PYEOF'
import sys, re
pyf, old, rows, nv, nt = sys.argv[1:6]
s = open(pyf).read()
s = s.replace(f"{old},", f"{rows},")                        # positional N_max (incompressible subclasses)
s = re.sub(r"self\.N_max\s*=\s*\d+", f"self.N_max = {rows}", s)
s = re.sub(r"self\.N_val\s*=\s*\d+", f"self.N_val = {nv}", s)
s = re.sub(r"self\.N_test\s*=\s*\d+", f"self.N_test = {nt}", s)
open(pyf, "w").write(s); print(f"PATCHED {pyf}: N_max -> {rows}, N_val -> {nv}, N_test -> {nt}")
PYEOF
grep -nE "N_max|N_val|N_test" scOT/problems/$PYF | head -20
sed -i 's/report_to=[^,)]*/report_to="none"/' scOT/train.py
sed -i 's/evaluation_strategy="epoch"/evaluation_strategy="steps", eval_steps=200/' scOT/train.py
sed -i 's/save_strategy="epoch"/save_strategy="steps", save_steps=200/' scOT/train.py
sed -i 's/dataloader_num_workers=CPU_CORES/dataloader_num_workers=4/' scOT/train.py
# REVISION-R1 10-27: STRICT_T0=1 -> fine-tune on (t=0 -> t) pairs only (scOT's fix_input_to_time_step=0) instead of
# all2all, the strict-IVP training also run for MOTION on the second-order wave tasks
[ "${STRICT_T0:-0}" = 1 ] && sed -i 's/^    train_dataset = get_dataset(/    train_eval_set_kwargs["fix_input_to_time_step"] = 0\n    train_dataset = get_dataset(/' scOT/train.py && grep -n "fix_input_to_time_step" scOT/train.py
# REVISION-R1 10-27 (user): V0=1 -> closed wave IVP of the submitted version (code/runners/_run_scot_ft_2ic.sh): the
# observed du/dt = u(t1) - u(t1-1) (0 at t1 = 0, the zero-velocity initial condition) as a third input channel
# (scOT re-initializes the embedding for the new channel count); labels [u], [c] unchanged
if [ "${V0:-0}" = 1 ]; then
python - <<'P2IC'
p = "scOT/problems/wave/acoustic.py"
s = open(p).read()
a = 'self.input_dim = 2\n        self.label_description = "[u],[c]"'
n1 = s.count(a); assert n1 >= 1
s = s.replace(a, 'self.input_dim = 3\n        self.label_description = "[u],[c]"')
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
n2 = s.count(old); assert n2 >= 1
s = s.replace(old, new)
open(p, "w").write(s); print(f"V0 PATCH: input_dim x{n1}, getitem x{n2}")
P2IC
fi
sed -i 's/num_train_epochs=config\["num_epochs"\],/num_train_epochs=config["num_epochs"], max_steps=config["max_steps"],/' scOT/train.py
PM=${PMODEL:-T}   # REVISION-R1 10-29: PMODEL=T|B selects the Poseidon size (MODEL_MAP key + weights)
STEPS=3200; NEVA=0; [ "${SMOKE:-0}" = 1 ] && { STEPS=2; NEVA=4; }
CFG=$(python - "$DS" "$N" "$STEPS" "$PM" <<'PYEOF'
import json, sys
print(json.dumps({"dataset": sys.argv[1], "num_trajectories": int(sys.argv[2]), "model_name": sys.argv[4], "lr": 5e-5,
  "lr_embedding_recovery": 5e-4, "lr_time_embedding": 5e-4, "weight_decay": 1e-6, "lr_scheduler": "cosine",
  "warmup_ratio": 0.0, "early_stopping_patience": 1000, "num_epochs": 200, "max_steps": int(sys.argv[3]),
  "batch_size": 40, "max_grad_norm": 5.0}))
PYEOF
)
TIMEARGS=""; [ "$KF" -gt 1 ] && TIMEARGS="--max_num_train_time_steps $((KF / ST)) --train_time_step_size $ST"
CK=/eu/results/scot_ft15/Poseidon-$PM${CKTAG:+_$CKTAG}/$TASK${SMOKE:+_smoke}; OUT=/work/runs/ft15/Poseidon-$PM${CKTAG:+_$CKTAG}/$TASK${SMOKE:+_smoke}
[ "${EVAL_ONLY:-0}" = 1 ] || rm -rf "$CK"; mkdir -p "$CK" "$OUT" /tmp/scratch
echo "=== FT Poseidon-$PM on $TASK: DS=$DS rows=$ROWS N=$N steps=$STEPS time=[$TIMEARGS] ==="
[ "${EVAL_ONLY:-0}" = 1 ] || \
python -u scOT/train.py --json_config --config "$CFG" --finetune_from "$CV/models/Poseidon-$PM" --replace_embedding_recovery \
  $TIMEARGS --data_path "$DATA" --checkpoint_path "$CK" 2>&1 | tee "$OUT/train.log" | grep -vE "it/s\]|s/it\]" | tail -40
CJ=$(find "$CK" -name config.json | grep -v 'checkpoint-' | head -1); [ -z "$CJ" ] && CJ=$(find "$CK" -name config.json | sort | tail -1)
BEST=$(dirname "$CJ"); echo "=== model: $BEST ==="
python -c "import json;c=json.load(open(\"$CJ\"));print(\"=== config embed_dim\",c.get(\"embed_dim\"),\"depths\",c.get(\"depths\"))"
cp /work/runs/poseidon_ft_data/eval_scot_ft15.py /tmp/po/
# REVISION-R1 10-26: PAPER_ONLY=1 skips our-metric scoring (re-scoring the paper metric only)
[ "${PAPER_ONLY:-0}" = 1 ] || \
for SP in ${SPLITS:-val test}; do for MO in direct ar; do
  [ "$KF" = 1 ] && [ "$MO" = ar ] && continue
  # REVISION-R1 10-27: DUMP_N=k -> also save predictions / ground truth of the first k trajectories (downstream prediction panels)
  # (DUMP is set before OUT: prefix assignments expand left to right, so a later OUT would change $OUT)
  DUMP=${DUMP_N:+$OUT/dump_${SP}_${MO}.npz} DUMP_N=${DUMP_N:-16} \
  TASK=$TASK MODEL=$BEST DS=$DS DATA=$DATA KF=$KF STEP=$ST SPLIT=$SP MODE=$MO NEV=$NEVA OUT=$OUT/score_${SP}_${MO}.json \
    python -u eval_scot_ft15.py 2>&1 | grep -E "SCOT_FT15|lead|Error|error|debug|Traceback|dump" | tail -40 | tee -a $OUT/score.log
done; done
# REVISION-R1 10-27: DUMP_ONLY=1 (with EVAL_ONLY=1 SPLITS=test DUMP_N=k) -> stop after the dumps
[ "${DUMP_ONLY:-0}" = 1 ] && { echo "SCOT_DUMP_DONE $TASK" | tee $OUT/dump_done.txt; exit 0; }
# Poseidon-paper metric (final lead 14 / GCE-RT 7, functions of interest), both rollout modes, test only
for MO in direct ar; do
  [ "$KF" = 1 ] && [ "$MO" = ar ] && continue
  TASK=$TASK MODEL=$BEST DS=$DS DATA=$DATA KF=$KF STEP=$ST SPLIT=test MODE=$MO NEV=$NEVA PAPER=1 OUT=$OUT/score_test_${MO}_paper.json \
    python -u eval_scot_ft15.py 2>&1 | grep -E "SCOT_FT15|Error|Traceback" | tail -5 | tee -a $OUT/score.log
done
# REVISION-R1 10-26: the official metric is computed in scOT's normalized label space -> also score it there
for MO in direct ar; do
  [ "$KF" = 1 ] && [ "$MO" = ar ] && continue
  TASK=$TASK MODEL=$BEST DS=$DS DATA=$DATA KF=$KF STEP=$ST SPLIT=test MODE=$MO NEV=$NEVA PAPER=1 PAPER_NORM=1 \
    OUT=$OUT/score_test_${MO}_paper_norm.json python -u eval_scot_ft15.py 2>&1 | grep -E "SCOT_FT15|Error|Traceback" | tail -5 | tee -a $OUT/score.log
done
echo "SCOT_FT15_DONE $TASK" | tee $OUT/done.txt
