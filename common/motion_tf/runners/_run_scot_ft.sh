#!/bin/bash
# Poseidon-B (scOT) DOWNSTREAM FINETUNE on the SAME /eu subset + SAME split as our model, for the
# head-to-head. Patches the scOT dataset class N_max/N_val/N_test to our subset (train=first N, test=
# last 128) so scOT trains/tests on IDENTICAL trajectories to our finetune. Then scOT.inference eval
# (median rel-L1 @final). env: FT_TASK (NS-PwC|CE-RM|ACE), FT_NSHOT.
#   image: a PyTorch H100(sm_90)-capable image; install scOT --no-deps to keep it.
set -e
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled WANDB_DISABLED=true HF_HUB_OFFLINE=1
CV="${CODE_VOL:-/code-vol}"; EU="${EU_VOL:-/eu}"
# SCOT_DATA override: Wave-Layer needs the scOT-format file (separate solution+c vars) which lives in a
# SEPARATE dir (_scot) so it doesn't clobber our merged Wave-Layer.nc. Other tasks use _assembled directly.
DATA="${SCOT_DATA:-$EU/data/poseidon/_assembled}"
FT_TASK="${FT_TASK:-NS-PwC}"; N="${FT_NSHOT:-128}"

# task -> (scOT dataset string, subset N_max, final_time index)
case "$FT_TASK" in
  NS-PwC) DS="fluids.incompressible.PiecewiseConstants.tracer"; NMAX=2354; FT=20; PYF="scOT/problems/fluids/incompressible.py"; OLDMAX=20000;;
  CE-RM)  DS="fluids.compressible.RichtmyerMeshkov";     NMAX=1260; FT=20; PYF="scOT/problems/fluids/compressible.py"; OLDMAX=1260;;
  ACE)    DS="reaction_diffusion.AllenCahn";             NMAX=3750; FT=19; PYF="scOT/problems/reaction_diffusion/allen_cahn.py"; OLDMAX=15000;;
  Wave-Layer) DS="wave.Layer";                          NMAX=3504; FT=20; PYF="scOT/problems/wave/acoustic.py"; OLDMAX=10512;;
  Poisson-Gauss) DS="elliptic.poisson.Gaussians";       NMAX=20000; FT=1; PYF="scOT/problems/elliptic/poisson.py"; OLDMAX=20000;;  # STEADY: use SCOT_NATIVE_EVAL=1 (no time sweep)
  *) echo "unknown FT_TASK $FT_TASK"; exit 2;;
esac
# VELOCITY-ONLY: strip .tracer from DS BEFORE the CFG json is built (so the model is pure [u,v], the paper's
# NS-PwC velocity task). The --just_velocities flag + EVALJV are added later where TIMEARGS is assembled.
[ -n "$SCOT_JUST_VEL" ] && DS="${DS%.tracer}"

which git >/dev/null 2>&1 || (apt-get update -q && apt-get install -y -q git) >/dev/null 2>&1
cd /tmp && rm -rf po && git clone --depth 1 https://github.com/camlab-ethz/poseidon po 2>&1 | tail -1
cd /tmp/po
python -c "import torch;print('torch',torch.__version__,'cap',torch.cuda.get_device_capability())"
pip install -q -e . --no-deps 2>&1 | tail -1
pip install -q transformers==4.29.2 accelerate==0.31.0 "huggingface_hub<0.20" wandb einops timm netCDF4 h5py pandas pyyaml matplotlib safetensors 2>&1 | tail -2

# ---- patch the dataset class to our subset split (N_max=subset, N_val=64, N_test=128) ----
python - "$PYF" "$OLDMAX" "$NMAX" <<'PYEOF'
import sys, re
pyf, oldmax, nmax = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(pyf).read()
# N_max: replace the subset total
s = s.replace(f"{oldmax},", f"{nmax},")            # positional super().__init__(NMAX, ...)
s = re.sub(r"self\.N_max\s*=\s*\d+", f"self.N_max = {nmax}", s)
# val/test reserve -> match our last-128 test (val=64)
s = re.sub(r"self\.N_val\s*=\s*\d+",  "self.N_val = 64",  s)
s = re.sub(r"self\.N_test\s*=\s*\d+", "self.N_test = 128", s)
open(pyf, "w").write(s)
print(f"PATCHED {pyf}: N_max->{nmax} N_val->64 N_test->128")
PYEOF
grep -nE "N_max|N_val|N_test" "$PYF" | head

# scOT sets report_to=wandb in TrainingArguments -> WandbCallback crashes under WANDB_DISABLED. Force none.
sed -i 's/report_to=[^,)]*/report_to="none"/' scOT/train.py
# SPEED: scOT hardcodes per-EPOCH eval+save (600MB ckpt) + 16 dataloader workers respawned each epoch.
# Small N => ~4 steps/epoch so this routine dominates. Keep num_epochs (training amount) but make eval/save
# STEP-based (every 100 steps) and cut workers -> the heavy routine runs a few times, not 200x.
sed -i 's/evaluation_strategy="epoch"/evaluation_strategy="steps", eval_steps=100/' scOT/train.py
sed -i 's/save_strategy="epoch"/save_strategy="steps", save_steps=100/' scOT/train.py
sed -i 's/dataloader_num_workers=CPU_CORES/dataloader_num_workers=0/' scOT/train.py
# OPTIMIZER-STEP matching: drive training by max_steps (flat, like our step-stream) not num_epochs.
# max_steps>0 overrides num_train_epochs in HF -> one continuous loop, no 200x epoch machinery.
sed -i 's/num_train_epochs=config\["num_epochs"\],/num_train_epochs=config["num_epochs"], max_steps=config["max_steps"],/' scOT/train.py
grep -nE 'report_to=|evaluation_strategy|save_strategy|dataloader_num_workers|max_steps' scOT/train.py | head

# ---- finetune config (json bypasses wandb/yaml cleaning) ----
STEPS=${SCOT_STEPS:-$(( 50 * N ))}   # EPOCH MATCH: 2000 epochs at batch40; SCOT_STEPS overrides (budget-matched runs)
CFG=$(python - "$DS" "$N" "$STEPS" <<'PYEOF'
import json,sys
print(json.dumps({
  "dataset": sys.argv[1], "num_trajectories": int(sys.argv[2]), "model_name": "B",
  "lr": 5e-5, "lr_embedding_recovery": 5e-4, "lr_time_embedding": 5e-4,
  "weight_decay": 1e-6, "lr_scheduler": "cosine", "warmup_ratio": 0.0,
  "early_stopping_patience": 200, "num_epochs": 200, "max_steps": int(sys.argv[3]),
  "batch_size": 40, "max_grad_norm": 5.0,
}))
PYEOF
)
echo "=== CFG: $CFG ==="
CKDIR="$EU/results/scot_ft_ckpt/${FT_TASK}_${N}${SCOT_RUN_TAG:+_$SCOT_RUN_TAG}"
# FULL-T: train to t=1.0 (step 20) instead of scOT's default t=0.7 (step 14). max_num_time_steps=10 x
# time_step_size=2 = all2all over snapshots 0,2,..,20 -> lead times up to 20 (our data has 21 frames).
# (Setting these also makes train.py skip its built-in do_test — fine, we run our own from-IC eval.)
TIMEARGS=""
[ -n "$SCOT_FULL_T" ] && TIMEARGS="--max_num_train_time_steps 10 --train_time_step_size 2"
# VELOCITY-ONLY (paper's NS-PwC = velocity task, Fig.7): drop tracer (.tracer suffix) + --just_velocities
# on BOTH train and eval, so the model is [u,v]-only (matches our uv_only). inference.py supports it.
EVALJV=""
[ -n "$SCOT_JUST_VEL" ] && { TIMEARGS="$TIMEARGS --just_velocities"; EVALJV="--just_velocities"; }   # DS already stripped above
set +e
if [ -n "$SCOT_CKPT" ]; then
  # ZERO-SHOT / explicit ckpt: eval an arbitrary model dir (e.g. the PUBLIC pre-finetune Poseidon-B).
  echo "=== EVAL explicit ckpt SCOT_CKPT=$SCOT_CKPT (no retrain) ==="
  BEST="$SCOT_CKPT"
elif [ -n "$SCOT_EVAL_ONLY" ]; then
  # EVAL-ONLY: reuse the EXISTING finetuned ckpt (do NOT rm/retrain). Prefer the final saved model dir
  # (config.json NOT under a checkpoint-* subdir); fall back to the latest checkpoint-N.
  echo "=== EVAL-ONLY: reuse existing ckpt under $CKDIR (no retrain) ==="
  CJ=$(find "$CKDIR" -name config.json 2>/dev/null | grep -v 'checkpoint-' | head -1)
  [ -z "$CJ" ] && CJ=$(find "$CKDIR" -path '*checkpoint-*/config.json' 2>/dev/null | sort -t- -k2 -n | tail -1)
  BEST=$(dirname "$CJ"); [ -z "$CJ" ] && BEST="$CKDIR"
else
  rm -rf "$CKDIR"; mkdir -p "$CKDIR"   # /eu = persistent (re-eval w/o retrain)
  echo "=== FINETUNE Poseidon-B on $FT_TASK N=$N (DS=$DS, data=$DATA) ==="
  mkdir -p /tmp/scratch
  python -u scOT/train.py --json_config --config "$CFG" \
    --finetune_from "$CV/models/Poseidon-B" --replace_embedding_recovery $TIMEARGS \
    --data_path "$DATA" --checkpoint_path "$CKDIR" --move_data /tmp/scratch 2>&1
  echo "=== train rc=$? ==="
  CJ=$(find "$CKDIR" -name config.json 2>/dev/null | head -1)
  BEST=$(dirname "$CJ"); [ -z "$CJ" ] && BEST="$CKDIR"
fi
echo "=== best ckpt: $BEST ==="; ls -la "$BEST" 2>/dev/null | head
# PER-RUN OUT subdir so finetuned/base/velT/etc. NEVER collide on the same CSV (a previous run's stale CSV
# on persistent /eu was being misread when an eval errored — e.g. base can't eval NS-PwC: channel mismatch).
OUT="$EU/results/scot_ft_fulltraj/${FT_TASK}_N${N}${SCOT_RUN_TAG:+_$SCOT_RUN_TAG}${SCOT_JUST_VEL:+_jv}"
rm -rf "$OUT"; mkdir -p "$OUT"   # fresh each run
# FULL-TRAJECTORY via scOT's native AR rollout: the model is trained all2all (multi-lead-time, step size 2),
# and predicts a trajectory AUTOREGRESSIVELY (3.03% @final = AR rollout, NOT a single +20 jump). The
# eval_accumulation_error mode rolls out AR and reports the rel-L1 at EACH step -> the per-frame trajectory
# error. ar_steps = FT/2 homogeneous +2 steps (matches the trainer's set_ar_steps(time_step_size//2)).
if [ -n "$SCOT_NATIVE_EVAL" ]; then
  # OPTION-1: reproduce the trainer's ~3% number — run scOT.inference --mode eval WITHOUT forcing
  # initial_time/final_time, so it uses the dataset's NATIVE default time config (max_num_time_steps=7,
  # time_step_size=2 = the mixed short-lead protocol the Poseidon PAPER reports, Fig.7 ~2% @64).
  echo "=== NATIVE EVAL (dataset default time config = paper protocol) on $BEST ==="
  python -u -m scOT.inference --mode eval --model_path "${BEST:-$CKDIR}" \
    --dataset "$DS" --data_path "$DATA" --file "$OUT/${FT_TASK}_N${N}_native.csv" \
    --ckpt_dir /tmp/ck2 $EVALJV 2>&1 | tail -10
  echo "=== RAW native CSV ==="
  python3 - "$OUT/${FT_TASK}_N${N}_native.csv" <<PYEOF
import csv,sys
for r in csv.DictReader(open(sys.argv[1])):
    print("init=%s final=%s ar=%s | mean_rel=%s%% | mean_over_median=%s%% | uv/med=%s%% | tracer/med=%s%%"%(
        r.get("initial_time"),r.get("final_time"),r.get("ar_steps"),r.get("mean_relative_l1_error"),
        r.get("mean_over_median_relative_l1_error"),r.get("uv/median_relative_l1_error"),r.get("tracer/median_relative_l1_error")))
PYEOF
  echo "=== SCOT_FT_DONE $FT_TASK N=$N rc=$? ==="
  exit 0
fi
# FROM-IC FULL TRAJECTORY (apples-to-apples with our single-shot): predict frame t DIRECTLY from frame 0
# (initial_time=0, final_time=t, ar_steps=1 = one lead-time-t forward, NO AR accumulation) for every t=1..FT.
# This is the exact analog of our model predicting all frames from the IC. (The trainer's 3.03% is a
# DIFFERENT protocol: mixed short lead-times 2..14 from varied start points — not comparable.)
echo "=== POSEIDON from-IC full-traj: IC->frame t, t=1..$FT, direct (ar_steps=1) ==="
for t in $(seq 1 "$FT"); do
  python -u -m scOT.inference --mode eval --model_path "${BEST:-$CKDIR}" \
    --dataset "$DS" --data_path "$DATA" --file "$OUT/${FT_TASK}_N${N}_fromIC_t${t}.csv" \
    --ckpt_dir /tmp/ck2 --initial_time 0 --final_time "$t" --ar_steps 1 $EVALJV 2>&1 | tail -1
done
python3 - "$OUT" "$FT_TASK" "$N" "$FT" <<PYEOF
import csv, sys, os, numpy as np
out, task, N, FT = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
# QOI key = the physical channel to score (NS velocity-only -> uv; Wave-Layer solution -> u; ACE -> joint).
QOI = os.environ.get("SCOT_QOI", "uv")
uv={}; jt={}; tr={}; pq={}
for t in range(1, FT+1):
    try: r=list(csv.DictReader(open("%s/%s_N%s_fromIC_t%d.csv"%(out,task,N,t))))[0]
    except Exception as e: print("skip t=%d: %s"%(t,e)); continue
    f=lambda k: float(r.get(k,"nan"))
    uv[t]=f("uv/median_relative_l1_error"); jt[t]=f("mean_relative_l1_error"); tr[t]=f("tracer/median_relative_l1_error")
    pq[t]=f("%s/median_relative_l1_error"%QOI)
    print("  IC->t%2d: %s=%.2f%%  uv=%.2f%%  tracer=%.2f%%  joint=%.2f%%"%(t,QOI,pq[t],uv[t],tr[t],jt[t]))
m=lambda d:np.nanmean(list(d.values()))
print("SCOT_FROMIC_QOI(%s) %s N=%s: QOI_mean=%.3f%%  QOI_final(t=%d)=%.3f%%"%(QOI,task,N,m(pq),FT,pq.get(FT,float("nan"))))
print("SCOT_FROMIC_FULLTRAJ %s N=%s: UV_mean=%.3f%%  TRACER_mean=%.3f%%  JOINT_mean=%.3f%%"%(task,N,m(uv),m(tr),m(jt)))
print("SCOT_FROMIC_FINAL    %s N=%s (t=%d): uv=%.3f%%  joint=%.3f%%"%(task,N,FT,uv.get(FT,float("nan")),jt.get(FT,float("nan"))))
PYEOF
# AR full-traj (Poseidon native autoregressive rollout): per-AR-step rel-L1, mean over steps = full-traj AR.
if [ -n "$SCOT_AR_EVAL" ]; then
  ARS="${SCOT_AR_STEPS:-$(( FT / 2 ))}"
  echo "=== POSEIDON AR full-traj: eval_accumulation_error final_time=$FT ar_steps=$ARS ==="
  python -u -m scOT.inference --mode eval_accumulation_error --model_path "${BEST:-$CKDIR}" \
    --dataset "$DS" --data_path "$DATA" --file "$OUT/${FT_TASK}_N${N}_ar.csv" \
    --ckpt_dir /tmp/ck3 --initial_time 0 --final_time "$FT" --ar_steps "$ARS" $EVALJV 2>&1 | tail -3
  python3 - "$OUT/${FT_TASK}_N${N}_ar.csv" <<PYEOF
import csv, sys, numpy as np
rows = list(csv.DictReader(open(sys.argv[1])))
uv = [float(r.get("uv/median_relative_l1_error", "nan")) for r in rows]
print("SCOT_AR per-step uv:", [round(x, 2) for x in uv])
print("SCOT_AR_FULLTRAJ UV_mean=%.3f%% (over %d AR steps); UV_final=%.3f%%" % (
    np.nanmean(uv), len(uv), uv[-1] if uv else float("nan")))
PYEOF
fi
echo "=== SCOT_FT_DONE $FT_TASK N=$N rc=$? ==="
