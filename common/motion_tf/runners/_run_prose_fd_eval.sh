#!/bin/bash
# EVAL the OFFICIAL pretrained PROSE-FD (HuggingFace felix-lyx/prose, prose_fd.pth) on OUR cached test
# data, PHYSICAL metric (PROSE's own) → directly comparable to our physical numbers + PROSE published.
set -e
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
CODE_VOL="${CODE_VOL:-/code-vol}"
PRE="${PRE:-$CODE_VOL/data/prose/prebuilt}"
echo "=== deps ==="
which git >/dev/null 2>&1 || (apt-get update -q && apt-get install -y -q git curl) 2>&1 | tail -1
which curl >/dev/null 2>&1 || (apt-get update -q && apt-get install -y -q curl) 2>&1 | tail -1
pip install -q --no-deps torchdata==0.8.0 2>&1 | tail -1
pip install -q h5py hydra-core omegaconf einops transformers rotary-embedding-torch tabulate scipy tqdm matplotlib wandb dadaptation 2>&1 | tail -2
cd /tmp && rm -rf pf && git clone --depth 1 https://github.com/felix-lyx/prose pf 2>&1 | tail -1
cd /tmp/pf/prose_fd
sed -i 's/^from neuralop.models import FNO3d/FNO3d = None/' models/baselines.py

# --- inject CachedNpy2D (reads our cache) + remap ALL_DATASETS (same as training cache runner) ---
cat >> data_utils/all_datasets.py <<'PYEOF'

import glob as _glob
class CachedNpy2D(myIterDp):
    CACHE_TL = None
    def __init__(self, params, symbol_env, split="train", train=True, file_idx=-1):
        super().__init__(params, symbol_env, split, train)
        self.type_label = self.CACHE_TL
        cache = params.data[self.type_label].data_path
        self.files = sorted(_glob.glob(os.path.join(cache, "shard_*.npy"))) if os.path.isdir(cache) else [cache]
        self._index = []
        for fi, f in enumerate(self.files):
            n = int(np.load(f, mmap_mode="r").shape[0]); self._index += [(fi, i) for i in range(n)]
        self.fully_shuffled = True
        if self.params.symbol.symbol_input:
            tree = self.symbol_env.generator.get_tree(self.type_label)
            self.symbol_ids = self.symbol_env.word_to_idx([self.symbol_env.equation_encoder.encode(tree)], float_input=False)[0]
    def __iter__(self):
        self.init_rng()
        rng = self.get_iter_range(len(self._index))[self.local_rank :: self.n_gpu_per_node]
        if self.train: rng = self.rng.permutation(rng)
        _cf = {}
        for k in rng:
            fi, i = self._index[int(k)]
            arr = _cf.get(fi)
            if arr is None: arr = np.load(self.files[fi], mmap_mode="r"); _cf = {fi: arr}
            data = self.augment_data(np.asarray(arr[i])[: self.t_num])
            data = torch.from_numpy(np.ascontiguousarray(data)).float()
            d = {"type": self.type_label, "data": data}
            if self.params.symbol.symbol_input: d["symbol_input"] = self.symbol_ids
            yield d
class CachedShallow(CachedNpy2D): CACHE_TL = "shallow_water"
class CachedComNS(CachedNpy2D): CACHE_TL = "com_ns"
class CachedIncomNS(CachedNpy2D): CACHE_TL = "incom_ns"
class CachedArena(CachedNpy2D): CACHE_TL = "incom_ns_arena"
class CachedCFD(CachedNpy2D): CACHE_TL = "cfdbench"
PYEOF
cat >> dataset.py <<'PYEOF'

ALL_DATASETS["shallow_water"] = ds.CachedShallow
ALL_DATASETS["com_ns"] = ds.CachedComNS
ALL_DATASETS["incom_ns"] = ds.CachedIncomNS
ALL_DATASETS["incom_ns_arena"] = ds.CachedArena
ALL_DATASETS["cfdbench"] = ds.CachedCFD
PYEOF

echo "=== download official PROSE-FD weights ==="
curl -L -s -o /tmp/prose_fd.pth https://huggingface.co/felix-lyx/prose/resolve/main/prose_fd.pth
ls -la /tmp/prose_fd.pth
NGPU=$(python -c "import torch;print(torch.cuda.device_count())")
OUT="$CODE_VOL/results/prose_fd_official_eval"; mkdir -p "$OUT"
echo "=== EVAL official PROSE-FD on our cache test (physical metric) ==="
set +e
torchrun --standalone --nproc_per_node=$NGPU main.py \
  hydra.run.dir="$OUT" model=prose_2to1 use_wandb=0 symbol.symbol_input=1 \
  eval_only=1 eval_from_exp=/tmp/prose_fd.pth \
  data.t_num=20 input_len=10 data.x_num=128 num_workers=0 num_workers_eval=0 \
  "data.types=[shallow_water,com_ns,incom_ns,incom_ns_arena,cfdbench]" \
  data.shallow_water.data_path="$PRE/shallow_water_n100000_t20.npy" \
  +data.com_ns.data_path="$PRE/com_ns_n100000_t20_shards" \
  +data.incom_ns.data_path="$PRE/incom_ns_n100000_t20_shards" \
  +data.incom_ns_arena.data_path="$PRE/pdearena_ns_n100000_t20_shards" \
  ++data.cfdbench.data_path="$PRE/cfdbench_n100000_t20.npy" \
  2>&1
echo "=== done (rc=$?) ==="
