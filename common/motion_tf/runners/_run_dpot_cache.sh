#!/bin/bash
# DPOT (HaoZhongkai/DPOT) on OUR dt=1 cache (converted to DPOT hdf5 by _convert_for_baselines.py TARGET=dpot).
# Same conditions as our ADA / PROSE: identical data, physical rel-L2 (DPOT normalize=False, masked rel-L2),
# ~122M (DPOT-M: width1024/depth12/mlp4). Joint 5-family training. env: EP, BS, GPU.
set -e
export PYTHONUNBUFFERED=1
CODE_VOL="${CODE_VOL:-/code-vol}"
H5="$CODE_VOL/data/prose/baseline_h5/dpot"
echo "=== deps ==="
which git >/dev/null 2>&1 || (apt-get update -q && apt-get install -y -q git) >/dev/null 2>&1
pip install -q h5py timm einops scipy pandas scikit-learn matplotlib tensorboard 2>&1 | tail -1
cd /tmp && rm -rf dp && git clone --depth 1 https://github.com/HaoZhongkai/DPOT dp 2>&1 | tail -1
cd /tmp/dp

# --- register our 5 families in the DATASET_DICT (train/test sizes match the 90/10 conversion split) ---
cat >> utils/make_master_file.py <<PYEOF

# ===== OUR dt=1 cache (converted) — 5 families, physical rel-L2, joint Stage-1 =====
_OURS = [
    ("shallow_water", 3600, 400, 1),
    ("com_ns",        5400, 600, 4),
    ("incom_ns",      3060, 340, 3),
    ("pdearena_ns",  10022, 1114, 3),
    ("diff_react",    3600, 400, 2),
    ("cfdbench",      8505, 945, 3),
]
for _nm, _ntr, _nte, _nc in _OURS:
    DATASET_DICT[_nm] = {
        "train_path": "$H5/%s_train.hdf5" % _nm,
        "test_path":  "$H5/%s_test.hdf5" % _nm,
        "train_size": _ntr, "test_size": _nte, "scatter_storage": False,
        "t_in": 10, "t_test": 10, "t_total": 20,
        "in_size": (128, 128), "n_channels": _nc, "downsample": (1, 1),
    }
PYEOF
echo "=== registered families ==="; python -c "from utils.make_master_file import DATASET_DICT as D; [print(k, D[k]['train_size'], D[k]['n_channels']) for k in ['shallow_water','com_ns','incom_ns','pdearena_ns','diff_react']]"

GPU=$(python -c "import torch;print(0 if torch.cuda.is_available() else -1)")
EP="${EP:-6}"; BS="${BS:-20}"; MODEL="${MODEL:-DPOT}"
WIDTH="${WIDTH:-1024}"; NLAYERS="${NLAYERS:-12}"; MLP="${MLP:-4}"; NBLOCKS="${NBLOCKS:-8}"
PATCH="${PATCH:-8}"; MODES="${MODES:-32}"   # FNO needs modes <= patched-grid//2 (res/PATCH); patch=8→grid16→modes<=8
echo "=== train $MODEL | 5-family joint | EP=$EP BS=$BS width=$WIDTH | physical rel-L2 ==="
set +e
python -u train_temporal.py --model "$MODEL" \
  --train_paths shallow_water com_ns incom_ns pdearena_ns cfdbench \
  --test_paths shallow_water com_ns incom_ns pdearena_ns cfdbench \
  --ntrain_list 3600 5400 3060 10022 8505 \
  --width "$WIDTH" --n_layers "$NLAYERS" --mlp_ratio "$MLP" --n_blocks "$NBLOCKS" --modes "$MODES" \
  --T_in 10 --T_ar 1 --T_bundle 1 --res 128 --patch_size "$PATCH" \
  --normalize 1 \
  --epochs "$EP" --warmup_epochs 1 --lr 1e-3 --batch_size "$BS" --gpu 0 2>&1
RC=$?
echo "=== done (rc=$RC) ==="; exit $RC
