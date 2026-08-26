#!/bin/bash
# BCAT (felix-lyx/bcat, arXiv 2501.18972) on OUR dt=1 cache — same 5 fluid families/data/physical-rel-L2 as
# the rest of the Stage-1 set. BCAT is data-only (NO symbol). Mirrors our PROSE-FD cache runner: inject a
# CachedNpy2D(myIterDp) reading our .npy cache, remap ALL_DATASETS to it, train model=bcat. env: EP, NSTEPS, BS, NGPU.
set -e
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
CODE_VOL="${CODE_VOL:-/code-vol}"
EU_VOL="${EU_VOL:-/eu}"                 # cluster volume: data + results (in-job writes persist; /code-vol object-vol does NOT)
PRE="${PRE:-$EU_VOL/data/prose/prebuilt}"
echo "=== deps ==="
which git >/dev/null 2>&1 || (apt-get update -q && apt-get install -y -q git) 2>&1 | tail -1
pip uninstall -y torchdata >/dev/null 2>&1 || true           # remove image's newer torchdata (no datapipes)
pip install -q --no-deps torchdata==0.7.1 2>&1 | tail -1     # dp.iter.Multiplexer API (BCAT pin)
python -c "import torch; print('torch', torch.__version__)"; python -c "import torchdata.datapipes" 2>&1 && echo "torchdata.datapipes OK" || echo "WARN torchdata.datapipes MISSING"
pip install -q "torch==2.7.1" "torchvision==0.22.1" torch-harmonics einops hydra-core omegaconf h5py tabulate sympy pandas \
  rotary-embedding-torch dadaptation torchinfo torchmetrics scipy scikit-learn tqdm wandb transformers \
  matplotlib seaborn neuraloperator accelerate torchtune==0.6.1 torchao==0.12.0 diffusers 2>&1 | tail -2
cd /tmp && rm -rf bc && git clone --depth 1 https://github.com/felix-lyx/bcat bc 2>&1 | tail -1
cd /tmp/bc/src && python /tmp/fix_eval3.py
# neuralop API changed (FNO2d removed); we use model=bcat so stub the broken import (mirrors PROSE-FD FNO3d stub)
grep -rlE "import (FNO2d|FNO3d)" --include=*.py . | while read f; do
  sed -i -E 's/.*from neuralop[^ ]* import.*FNO[23]d.*/FNO2d = None; FNO3d = None  # stub (model=bcat)/' "$f"; done
echo "stubbed FNO imports in: $(grep -rl 'FNO2d = None; FNO3d = None' --include=*.py . | tr '\n' ' ')"

# --- inject CachedNpy2D loader (reads our cache; data-only, no symbol/type key) ---
cat >> data_utils/all_datasets.py <<'PYEOF'

# ===== OUR-CACHE loader (data-only, mirrors PROSE-FD CachedNpy2D) =====
import glob as _glob
class CachedNpy2D(myIterDp):
    CACHE_TL = None
    def __init__(self, params, symbol_env, split="train", train=True):
        super().__init__(params, symbol_env, split, train)
        self.type_label = self.CACHE_TL
        cache = params.data[self.type_label].data_path
        self.files = (sorted(_glob.glob(os.path.join(cache, "shard_*.npy")))
                      if os.path.isdir(cache) else [cache])
        self._index = []
        for fi, f in enumerate(self.files):
            n = int(np.load(f, mmap_mode="r").shape[0])
            self._index += [(fi, i) for i in range(n)]
        self.fully_shuffled = True
        if not train:  # UNIFIED EVAL SUBSET = ADA's test-600 (first 100 rows of the last-10% region)
            if len(self.files) > 1:
                n_te = max(1, int(round(0.1 * len(self.files))))
                lo = sum(int(np.load(f, mmap_mode="r").shape[0]) for f in self.files[:-n_te])
                self._index = self._index[lo:lo + 100]
            else:
                c = len(self._index); n_te = max(1, int(round(0.1 * c)))
                self._index = self._index[c - n_te : c - n_te + 100]

    def __iter__(self):
        self.init_rng()
        rng = (np.arange(len(self._index)) if not self.train
               else self.get_iter_range(len(self._index))[self.local_rank :: self.n_gpu_per_node])  # UNIFIED: eval sees ALL 100/family
        if self.train:
            rng = self.rng.permutation(rng)
        _cf = {}
        for k in rng:
            fi, i = self._index[int(k)]
            arr = _cf.get(fi)
            if arr is None:
                arr = np.load(self.files[fi], mmap_mode="r"); _cf = {fi: arr}
            _vt = min(self.VALID_T, self.t_num) if getattr(self, "VALID_T", None) else self.t_num
            data = np.asarray(arr[i])[: _vt]                  # real frames only (uncond=14); collate re-pads to mixed_length
            _sh = int(os.environ.get("SHIFT", "0"))
            if _sh:
                data = np.roll(data, (-_sh, -_sh), axis=(1, 2)).copy()
            _fr = int(os.environ.get("FLIP_FR", "0")); _fc = int(os.environ.get("FLIP_FC", "0"))
            if _fr:
                data = np.flip(data, axis=1).copy()
                if data.shape[-1] >= 2: data[..., 0] = -data[..., 0]
            if _fc:
                data = np.flip(data, axis=2).copy()
                if data.shape[-1] >= 2: data[..., 1] = -data[..., 1]
            data = self.augment_data(data)
            data = torch.from_numpy(np.ascontiguousarray(data)).float()
            yield {"data": data}                              # BCAT: data-only (no 'type', no 'symbol_input')

class CachedShallow(CachedNpy2D): CACHE_TL = "shallow_water"
class CachedCom(CachedNpy2D): CACHE_TL = "com_ns"
class CachedIncom(CachedNpy2D): CACHE_TL = "incom_ns"
class CachedArena(CachedNpy2D): CACHE_TL = "incom_ns_arena"
class CachedCFD(CachedNpy2D): CACHE_TL = "cfdbench"
class CachedArenaU(CachedNpy2D): CACHE_TL = "incom_ns_arena_u"; VALID_T = 14   # uncond: 14 real frames -> mixed_length pads to t_num
PYEOF

# remap ALL_DATASETS (src/dataset.py) — reuse existing family names (already in DatasetIdx + configs)
python - <<'PYEOF'
p = "dataset.py"
s = open(p).read()
i = s.index("ALL_DATASETS"); j = s.index("}", i)
inj = ("\nALL_DATASETS['shallow_water']=ds.CachedShallow; ALL_DATASETS['com_ns']=ds.CachedCom\n"
       "ALL_DATASETS['incom_ns']=ds.CachedIncom; ALL_DATASETS['incom_ns_arena']=ds.CachedArena\n"
       "ALL_DATASETS['cfdbench']=ds.CachedCFD; ALL_DATASETS['incom_ns_arena_u']=ds.CachedArenaU\n")
s = s[:j+1] + "\n" + inj + s[j+1:]
open(p, "w").write(s)
print("remapped ALL_DATASETS -> CachedNpy2D")
PYEOF

NGPU=$(python -c "import torch;print(torch.cuda.device_count())")
EP="${EP:-125}"; NSTEPS="${NSTEPS:-250}"; BS="${BS:-32}"
# RE-PIN torchdata 0.7.1 LAST — the big deps install (torchtune/torchao/diffusers/accelerate) upgrades it,
# dropping torchdata.datapipes; reinstall it as the final torchdata so the runtime import works.
pip uninstall -y torchdata >/dev/null 2>&1 || true
pip install -q --no-deps torchdata==0.7.1 2>&1 | tail -1
python -c "import torchdata.datapipes; print('torchdata.datapipes OK (final)')" || echo "WARN torchdata.datapipes MISSING"
# --- crash-safe checkpointing (was BROKEN: trainer saves to {dump_path}/{exp_name}/{exp_id} with base
# dump_path defaulting to the RELATIVE "checkpoint" dir => EPHEMERAL container, lost on crash; exp_id RANDOM
# => no resume). Fix (verified via smoke test): absolute volume base + FIXED exp_id=v1 so checkpoint.pth
# (+ save_periodic=1 -> periodic-N.pth) land on the volume; reload_checkpoint=$CKDIR auto-resumes. ---
DUMP="${DUMP:-$EU_VOL/results}"; EXP="p1_bcat_sym"; CKDIR="$DUMP/$EXP/v1"; mkdir -p "$CKDIR"
NGPU=$(python -c "import torch;print(torch.cuda.device_count())")
for TAGSPEC in sh1:0:0 sh8:0:0; do
  TAG="${TAGSPEC%%:*}"; REST="${TAGSPEC#*:}"; FR="${REST%%:*}"; FC="${REST##*:}"
  echo "=== FLIP $TAG (FR=$FR FC=$FC) ==="
  set +e
  SHIFT=$([ "$TAG" = "sh1" ] && echo 1 || echo 8) FLIP_FR=0 FLIP_FC=0 FLIPTAG=$TAG torchrun --standalone --nnodes 1 --nproc_per_node "$NGPU" main.py \
    dump_path="$DUMP" exp_name="$EXP" exp_id=v1 save_periodic=0 eval_only=1 eval_from_exp=/eu/results/p1_bcat_fair/v1 \
    model=bcat use_wandb=0 symbol.symbol_input=0 data.tie_fields=1 \
    data.t_num=20 input_len=10 data.x_num=128 \
    "data.types=[shallow_water,com_ns,incom_ns,incom_ns_arena,incom_ns_arena_u]" \
    ++data.shallow_water.data_path="$PRE/shallow_water_n100000_t20.npy" \
    ++data.com_ns.data_path="$PRE/com_ns_n100000_t20_shards" \
    ++data.incom_ns.data_path="$PRE/incom_ns_n100000_t20_shards" \
    ++data.incom_ns_arena.data_path="$PRE/pdearena_ns_n100000_t20_shards" \
    ++data.cfdbench.data_path="$PRE/cfdbench_n100000_t20.npy" \
    ++data.incom_ns_arena_u.data_path="$PRE/pdearena_uncond_n100000_t20_shards" \
    data.mixed_length=20 \
    batch_size=32 2>&1 | grep -E "FULLTEST|DUMP_SAVED|Error|Traceback" | tail -20
  set -e
done
python - <<'CEOF'
import numpy as np, glob, os
for fam in ("shallow_water","com_ns","incom_ns","incom_ns_arena","incom_ns_arena_u"):
    f0 = f"/eu/_bcatsymE_id_{fam}.npz"
    if not os.path.exists(f0): continue
    z0 = np.load(f0); P0 = z0["pred"].astype(np.float64); GT = z0["gt"].astype(np.float64)
    dg = np.sqrt((GT ** 2).sum(axis=(2,3,4))) + 1e-12
    def err(P): return float((np.sqrt(((P - GT) ** 2).sum(axis=(2,3,4))) / dg).mean() * 100)
    out = [f"err_id={err(P0):.3f}%"]
    for tag, sh in (("sh1", 1), ("sh8", 8)):
        f = f"/eu/_bcatsymE_{tag}_{fam}.npz"
        if not os.path.exists(f): continue
        Pg = np.roll(np.load(f)["pred"].astype(np.float64), (sh, sh), axis=(2, 3))
        d0 = np.sqrt((P0 ** 2).sum(axis=(2,3,4))) + 1e-12
        eps = float((np.sqrt(((Pg - P0) ** 2).sum(axis=(2,3,4))) / d0).mean() * 100)
        out.append(f"{tag}: eps={eps:.3f}% err={err(Pg):.3f}%")
    print(f"[BCAT {fam} SHIFT] " + " | ".join(out), flush=True)
print("COMBINE_DONE", flush=True)
CEOF
echo "=== done ==="
