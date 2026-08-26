#!/bin/bash
# Run the OFFICIAL pretrained PROSE-FD (HF felix-lyx/prose, prose_fd.pth) on OUR com_ns test cache and
# DUMP its physical-space predictions + targets (full channels) so we can measure PER-CHANNEL rel-L2 —
# i.e. does PROSE smooth compressible-NS velocity the same way ADA does, or capture it? Patches
# evaluate.py to np.savez the first com_ns batch (after denormalization, before the channel trim).
set -e
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
CODE_VOL="${CODE_VOL:-/code-vol}"
PRE="$CODE_VOL/data/prose/prebuilt"
which git >/dev/null 2>&1 || (apt-get update -q && apt-get install -y -q git curl) >/dev/null 2>&1
which curl >/dev/null 2>&1 || (apt-get update -q && apt-get install -y -q curl) >/dev/null 2>&1
pip install -q --no-deps torchdata==0.8.0 2>&1 | tail -1
pip install -q h5py hydra-core omegaconf einops transformers rotary-embedding-torch tabulate scipy tqdm matplotlib wandb dadaptation 2>&1 | tail -2
cd /tmp && rm -rf pf && git clone --depth 1 https://github.com/felix-lyx/prose pf 2>&1 | tail -1
cd /tmp/pf/prose_fd
sed -i 's/^from neuralop.models import FNO3d/FNO3d = None/' models/baselines.py

# --- CachedNpy2D loader (reads our cache), same as the eval runner ---
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
class CachedComNS(CachedNpy2D): CACHE_TL = "com_ns"
PYEOF
cat >> dataset.py <<'PYEOF'

ALL_DATASETS["com_ns"] = ds.CachedComNS
PYEOF

# --- patch evaluate.py: dump first com_ns batch (physical pred+gt, full channels) ---
python - <<'PYEOF'
f = "evaluate.py"; lines = open(f).read().split("\n"); out = []; done = False
for ln in lines:
    if (not done) and 'results["data_loss"].extend(data_loss)' in ln:
        ind = ln[:len(ln) - len(ln.lstrip())]
        out += [ind + 'if type == "com_ns" and idx == 0:',
                ind + '    import numpy as _np',
                ind + '    _np.savez("/code-vol/results/prose_comns_dump.npz", pred=data_output.float().cpu().numpy(), gt=d["data_label"].float().cpu().numpy())',
                ind + '    print("DUMPED prose com_ns batch", flush=True)']
        done = True
    out.append(ln)
open(f, "w").write("\n".join(out))
print("patched evaluate.py:", done)
PYEOF

echo "=== download official PROSE-FD weights ==="
curl -L -s -o /tmp/prose_fd.pth https://huggingface.co/felix-lyx/prose/resolve/main/prose_fd.pth
ls -la /tmp/prose_fd.pth
NGPU=$(python -c "import torch;print(torch.cuda.device_count())")
OUT="$CODE_VOL/results/prose_fd_official_eval"; mkdir -p "$OUT"
echo "=== EVAL official PROSE-FD on our com_ns cache (dump predictions) ==="
set +e
torchrun --standalone --nproc_per_node=$NGPU main.py \
  hydra.run.dir="$OUT" model=prose_2to1 use_wandb=0 symbol.symbol_input=1 \
  eval_only=1 eval_from_exp=/tmp/prose_fd.pth \
  data.t_num=20 input_len=10 data.x_num=128 num_workers=0 num_workers_eval=0 \
  "data.types=[com_ns]" \
  +data.com_ns.data_path="$PRE/com_ns_n100000_t20_shards" \
  2>&1
echo "=== done (rc=$?) ==="
ls -la "$CODE_VOL/results/prose_comns_dump.npz" 2>/dev/null
