#!/bin/bash
# PROSE-FD baseline on OUR data (same families/split/horizon), PROSE's training conditions.
# Runs the upstream felix-lyx/prose PyTorch model via torchrun, reading our volume's raw PDEBench/
# PDEArena h5 (which is exactly the format PROSE's native loaders expect). Env vars:
#   BS (batch_size, default 32), EP (max_epoch, default 40), NSTEPS (n_steps_per_epoch, default repo),
#   SMOKE=1 (tiny: 1 epoch, few steps, just verify data loads + trains).
set -e
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
CODE_VOL="${CODE_VOL:-/code-vol}"
DATA="$CODE_VOL/data/prose"
echo "=== deps ==="
which git >/dev/null 2>&1 || (apt-get update -q && apt-get install -y -q git) 2>&1 | tail -2
# torchdata==0.8.0 = last version with torchdata.datapipes (PROSE uses IterDataPipe) that is still
# compatible with torch 2.4 (newer torchdata removed datapipes; 0.7.1 would downgrade torch).
pip install -q --no-deps torchdata==0.8.0 2>&1 | tail -1
pip install -q h5py hydra-core omegaconf einops transformers rotary-embedding-torch tabulate scipy tqdm matplotlib wandb dadaptation 2>&1 | tail -3
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'ngpu', torch.cuda.device_count())"
echo "=== clone PROSE-FD ==="
cd /tmp && rm -rf pf && git clone --depth 1 https://github.com/felix-lyx/prose pf 2>&1 | tail -1
cd /tmp/pf/prose_fd
# we use prose_2to1, never the FNO baseline → stub the neuralop FNO3d import (avoids neuralop version hell;
# FNO3d is only referenced inside class FNO.__init__, never instantiated here).
sed -i 's/^from neuralop.models import FNO3d/FNO3d = None  # stubbed (FNO baseline unused)/' models/baselines.py
# our incom_ns raw folder has a few TRUNCATED/.part files (incomplete downloads) → PROSE's IncomNS2D
# reads the whole folder and crashes on the bad ones. Filter its file list to COMPLETE .h5 only (>8GB;
# good files are ~9.9GB). Line 350 is unique to IncomNS2D (arena/com use different listdir forms).
sed -i 's#self.data_files = sorted(os.listdir(self.folder))#self.data_files = sorted([f for f in os.listdir(self.folder) if f.endswith(".h5") and os.path.getsize(os.path.join(self.folder,f)) > 8000000000])#' data_utils/all_datasets.py
# same problem in OTHER families (com_ns CFD, pdearena arena) — a few 96-byte / .part incomplete files.
# .part is already excluded by the extension filters; add a >1MB size filter to skip the tiny corrupt
# .hdf5/.h5 (real files are 55GB/88GB/704MB → 1MB threshold is safe).
# com_ns folder is MIXED-resolution (128²-native "_128_" + 512²-native "_512_" files). PROSE reads the
# whole folder → batches mix 128² and 512². Our own preprocessing (prose.py) skips "512"-named files;
# mirror that here so com_ns = 128²-native only (matches our data exactly; x_num=128 → x_step=1 → 128²).
sed -i 's#if f.endswith(".hdf5")]#if f.endswith(".hdf5") and "512" not in f and os.path.getsize(os.path.join(folder,f)) > 1000000]#' data_utils/all_datasets.py
sed -i 's#and f.endswith(".h5")]#and f.endswith(".h5") and os.path.getsize(os.path.join(self.folder,f)) > 1000000]#g' data_utils/all_datasets.py
# DEBUG: print each family's loaded data shape (find the family whose data isn't 128²).
[ "${SHAPEDEBUG:-0}" = "1" ] && sed -i 's#d\["data"\] = data#d["data"] = data; print("SHAPETAG", self.type_label, tuple(data.shape), flush=True)#g' data_utils/all_datasets.py
NGPU=$(python -c "import torch;print(torch.cuda.device_count())")
OUT="$CODE_VOL/results/prose_fd_ours"; mkdir -p "$OUT"
# sync any existing ckpt back to local for resume (PROSE writes into hydra.run.dir = OUT directly)
EXTRA=""
if [ "${SMOKE:-0}" = "1" ]; then EXTRA="max_epoch=1 n_steps_per_epoch=${NSTEPS:-20} batch_size=${BS:-8}";
else EXTRA="max_epoch=${EP:-40} batch_size=${BS:-32}"; [ -n "$NSTEPS" ] && EXTRA="$EXTRA n_steps_per_epoch=$NSTEPS"; fi
echo "=== torchrun nproc=$NGPU | $EXTRA ==="
set +e
torchrun --standalone --nproc_per_node=$NGPU main.py \
  hydra.run.dir="$OUT" \
  model=prose_2to1 \
  use_wandb=0 \
  symbol.symbol_input=1 \
  data.t_num=20 input_len=10 data.x_num=128 \
  "data.types=[${TYPES:-shallow_water,com_ns,incom_ns,incom_ns_arena}]" \
  data.shallow_water.data_path="$DATA/pdebench/2D/shallow-water/2D_rdb_NA_NA.h5" \
  data.com_ns.folders.rand="$DATA/pdebench/2D/CFD/2D_Train_Rand" \
  data.com_ns.folders.turb="$DATA/pdebench/2D/CFD/2D_Train_Rand" \
  data.com_ns.type=rand \
  data.com_ns.x_num=128 \
  data.incom_ns.folder="$DATA/pdebench/2D/NS_incom" \
  data.incom_ns.x_num=512 \
  data.incom_ns_arena.folder="$DATA/pdearena/NavierStokes-2D-conditoned" \
  num_workers=0 num_workers_eval=0 \
  $EXTRA 2>&1
RC=$?
echo "=== done (rc=$RC) ==="; exit $RC
