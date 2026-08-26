#!/bin/bash
# MPP (PolymathicAI/multiple_physics_pretraining) on OUR dt=1 cache (converted to MPP sample-keyed hdf5 by
# _convert_for_baselines.py TARGET=mpp). Same conditions: identical data, built-in instance-norm + normalized
# rel-L2 loss (physical-comparable), 5-family joint via shared field vocab. env: EP, BS, EMBED.
set -e
export PYTHONUNBUFFERED=1
CODE_VOL="${CODE_VOL:-/code-vol}"
MPPH5="$CODE_VOL/data/prose/baseline_h5/mpp"
echo "=== deps ==="
which git >/dev/null 2>&1 || (apt-get update -q && apt-get install -y -q git) >/dev/null 2>&1
pip install -q h5py einops timm ruamel.yaml wandb tqdm dadaptation adan-pytorch torchinfo matplotlib scipy 2>&1 | tail -1
cd /tmp && rm -rf mpp && git clone --depth 1 https://github.com/PolymathicAI/multiple_physics_pretraining mpp 2>&1 | tail -1
cd /tmp/mpp

# data lives in per-family subdirs $MPPH5/<fam>/chunk_*.h5 (written chunked by _convert_for_baselines.py
# TARGET=mpp — avoids include_string collision + the ~40GB single-file FUSE corruption). MPP globs the dir.
for f in shallow_water com_ns incom_ns pdearena_ns cfdbench; do
  rm -f "$MPPH5/$f"/data.h5 "$MPPH5/$f"/data.hdf5   # remove stale symlinks from older runs (MPP globs *.h5/*.hdf5)
  echo "$f: $(ls "$MPPH5/$f"/chunk_*.h5 2>/dev/null | wc -l) chunks"
done

# --- custom sample-keyed dataset classes (mimic SWEDataset reader, per-family field_names) + register ---
cat >> data_utils/hdf5_datasets.py <<'PYEOF'

# OUR sample-keyed hdf5 (file[key]['data']=(T,H,W,C)); SWEDataset's reader/stats/bcs are generic.
# _specifics must stay @staticmethod (datasets.py calls it on the CLASS for the global field vocab).
class CachedSWE(SWEDataset):
    @staticmethod
    def _specifics(): return 0, None, ['h'], 'cached_swe', 'sample'
class CachedDR(SWEDataset):
    @staticmethod
    def _specifics(): return 0, None, ['activator', 'inhibitor'], 'cached_dr', 'sample'
class CachedCom(SWEDataset):
    @staticmethod
    def _specifics(): return 0, None, ['Vx', 'Vy', 'density', 'pressure'], 'cached_com', 'sample'
class CachedIncom(SWEDataset):
    @staticmethod
    def _specifics(): return 0, None, ['Vx', 'Vy', 'particles'], 'cached_incom', 'sample'
class CachedArena(SWEDataset):
    @staticmethod
    def _specifics(): return 0, None, ['Vx', 'Vy', 'u'], 'cached_arena', 'sample'
class CachedCFD(SWEDataset):
    @staticmethod
    def _specifics(): return 0, None, ['Vx', 'Vy', 'cfd_mask'], 'cached_cfd', 'sample'
PYEOF
# register in the name->object map (used by train_data_paths' 2nd element):
python - <<PYEOF
import re, io
p = "data_utils/datasets.py"
s = open(p).read()
inj = ("\nfrom data_utils.hdf5_datasets import CachedSWE, CachedDR, CachedCom, CachedIncom, CachedArena, CachedCFD\n"
       "DSET_NAME_TO_OBJECT['cached_swe']=CachedSWE; DSET_NAME_TO_OBJECT['cached_dr']=CachedDR\n"
       "DSET_NAME_TO_OBJECT['cached_com']=CachedCom; DSET_NAME_TO_OBJECT['cached_incom']=CachedIncom\n"
       "DSET_NAME_TO_OBJECT['cached_arena']=CachedArena; DSET_NAME_TO_OBJECT['cached_cfd']=CachedCFD\n")
# append after the DSET_NAME_TO_OBJECT dict definition (find first close brace after it)
i = s.index("DSET_NAME_TO_OBJECT")
j = s.index("}", i)
s = s[:j+1] + "\n" + inj + s[j+1:]
open(p,"w").write(s)
print("registered cached_* in DSET_NAME_TO_OBJECT")
PYEOF

# --- build a config YAML from the Ti template: our data paths, size, n_steps, no wandb ---
EMBED="${EMBED:-1024}"; HEADS="${HEADS:-16}"; BLK="${BLK:-12}"; NSTEPS="${NSTEPS:-10}"; BS="${BS:-16}"; EP="${EP:-40}"
python - <<PYEOF
from ruamel.yaml import YAML
yaml = YAML()
src = "config/mpp_avit_ti_config.yaml"
d = yaml.load(open(src))
# the top-level mapping has namespaces; patch every namespace that carries train_data_paths
def patch(cfg):
    cfg['train_data_paths'] = [["$MPPH5/shallow_water","cached_swe",""],["$MPPH5/com_ns","cached_com",""],
        ["$MPPH5/incom_ns","cached_incom",""],["$MPPH5/pdearena_ns","cached_arena",""],["$MPPH5/cfdbench","cached_cfd",""]]
    cfg['valid_data_paths'] = list(cfg['train_data_paths'])
    cfg['embed_dim']=$EMBED; cfg['num_heads']=$HEADS; cfg['processor_blocks']=$BLK
    cfg['n_states']=64   # field-embedding vocab must cover ALL registered datasets' fields (5-fluid + native MPP)
    cfg['n_steps']=$NSTEPS; cfg['batch_size']=$BS; cfg['epochs']=$EP if 'epochs' in cfg else cfg.get('epochs')
    if 'log_to_wandb' in cfg: cfg['log_to_wandb']=False
    if 'learning_rate' in cfg and cfg['learning_rate'] in (-1,-1.0): cfg['learning_rate']=1e-3
for k,v in d.items():
    if isinstance(v, dict) and ('train_data_paths' in v or 'embed_dim' in v):
        patch(v)
yaml.dump(d, open("config/ours.yaml","w"))
print("wrote config/ours.yaml")
PYEOF
NS=$(grep -m1 -oE "^[a-zA-Z_]+:" config/ours.yaml | head -1 | tr -d ':')
echo "=== train MPP | embed=$EMBED blk=$BLK n_steps=$NSTEPS | config ns=$NS ==="
set +e
python -u train_basic.py --run_name ours --yaml_config ./config/ours.yaml --config "${NS:-basic_config}" 2>&1
RC=$?
echo "=== done (rc=$RC) ==="; exit $RC
