#!/bin/bash
# PROSE-FD on OUR CACHED .npy (identical data to our models). Reads prebuilt cache (n,t_num,128,128,d
# native channels) via a CachedNpy2D loader injected into PROSE. PROSE's own training conditions.
# env: BS, EP, NSTEPS, SMOKE=1.
set -e
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
CODE_VOL="${CODE_VOL:-/code-vol}"
EU_VOL="${EU_VOL:-/eu}"                 # cluster volume: data + results live here (in-job writes persist; /code-vol object-vol does NOT)
PRE="${PRE:-$EU_VOL/data/prose/prebuilt}"
echo "=== deps ==="
which git >/dev/null 2>&1 || (apt-get update -q && apt-get install -y -q git) 2>&1 | tail -1
pip install -q --no-deps torchdata==0.8.0 2>&1 | tail -1
pip install -q h5py hydra-core omegaconf einops transformers==4.53.0 rotary-embedding-torch tabulate scipy tqdm matplotlib wandb dadaptation 2>&1 | tail -2
cd /tmp && rm -rf pf && git clone --depth 1 https://github.com/felix-lyx/prose pf 2>&1 | tail -1
cd /tmp/pf/prose_fd && python /tmp/fix_eval3.py
sed -i 's/^from neuralop.models import FNO3d/FNO3d = None/' models/baselines.py
# Report the eval metric in NORMALIZED space (same as our model), not physical. By default PROSE
# denormalizes (×std+mean) before compute_metrics → physical rel-L2 (flatters mean-dominated families
# like com_ns/SWE). Neutralize the two denorm lines so _l2_error is computed in normalized space.
# (Training LOSS is unchanged — still normalized; this only changes the reported metric.)
# DEFAULT = PHYSICAL metric (PROSE's native denorm) so it matches our model's physical eval for the
# Stage-1 comparison. Only neutralize denorm when METRIC=norm is explicitly requested.
if [ "${METRIC:-phys}" = "norm" ]; then
  echo "=== METRIC=norm: patching evaluate.py to report NORMALIZED rel-L2 ==="
  sed -i 's#data_output = data_output \* d\["std"\] + d\["mean"\]#data_output = data_output  # normalized metric#g' evaluate.py
  sed -i 's#d\["data_label"\] = d\["data_label"\] \* d\["std"\] + d\["mean"\]#d["data_label"] = d["data_label"]  # normalized metric#g' evaluate.py
else
  echo "=== METRIC=phys (default): PROSE native physical rel-L2 (matches our model eval) ==="
fi

# --- inject CachedNpy2D loader (reads our cache) + remap ALL_DATASETS ---
cat >> data_utils/all_datasets.py <<'PYEOF'

# ===== OUR-CACHE loader: train/eval PROSE on the exact .npy cache our models use =====
import glob as _glob
class CachedNpy2D(myIterDp):
    CACHE_TL = None
    VALID_T = None                                            # truncate to this many real frames (uncond=14);
                                                              # None => use full t_num. PROSE collate then pads
                                                              # back to mixed_length + sets data_mask_len (fair).
    def __init__(self, params, symbol_env, split="train", train=True, file_idx=-1):
        super().__init__(params, symbol_env, split, train)
        self.type_label = self.CACHE_TL
        cache = params.data[self.type_label].data_path
        if os.path.isdir(cache):
            self.files = sorted(_glob.glob(os.path.join(cache, "shard_*.npy")))
        else:
            self.files = [cache]
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

        if self.params.symbol.symbol_input:
            tree = self.symbol_env.generator.get_tree(self.type_label)
            enc = self.symbol_env.equation_encoder.encode(tree)
            self.symbol_ids = self.symbol_env.word_to_idx([enc], float_input=False)[0]
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
            _vt = min(self.VALID_T, self.t_num) if self.VALID_T else self.t_num
            data = np.asarray(arr[i])[: _vt]                  # real frames only (uncond=14); collate re-pads
            _fr = int(os.environ.get("FLIP_FR", "0")); _fc = int(os.environ.get("FLIP_FC", "0"))
            if _fr:
                data = np.flip(data, axis=1).copy()
                if data.shape[-1] >= 2: data[..., 0] = -data[..., 0]
            if _fc:
                data = np.flip(data, axis=2).copy()
                if data.shape[-1] >= 2: data[..., 1] = -data[..., 1]
            data = self.augment_data(data)
            data = torch.from_numpy(np.ascontiguousarray(data)).float()
            d = {"type": self.type_label, "data": data}
            if self.params.symbol.symbol_input:
                d["symbol_input"] = self.symbol_ids
            yield d

class CachedShallow(CachedNpy2D): CACHE_TL = "shallow_water"
class CachedComNS(CachedNpy2D): CACHE_TL = "com_ns"
class CachedIncomNS(CachedNpy2D): CACHE_TL = "incom_ns"
class CachedArena(CachedNpy2D): CACHE_TL = "incom_ns_arena"
class CachedArenaU(CachedNpy2D): CACHE_TL = "incom_ns_arena_u"; VALID_T = 14   # uncond: 14 real frames
class CachedReactDiff(CachedNpy2D): CACHE_TL = "react_diff"   # (react_diff symbol is a NotImpl stub in PROSE-FD — unused)
class CachedCFD(CachedNpy2D): CACHE_TL = "cfdbench"          # 5th fluid family (PROSE-FD native)
PYEOF

cat >> dataset.py <<'PYEOF'

# OUR-CACHE remap: 4 families read the prebuilt .npy cache (identical data to our models)
ALL_DATASETS["shallow_water"] = ds.CachedShallow
ALL_DATASETS["com_ns"] = ds.CachedComNS
ALL_DATASETS["incom_ns"] = ds.CachedIncomNS
ALL_DATASETS["incom_ns_arena"] = ds.CachedArena
ALL_DATASETS["incom_ns_arena_u"] = ds.CachedArenaU
ALL_DATASETS["react_diff"] = ds.CachedReactDiff
ALL_DATASETS["cfdbench"] = ds.CachedCFD
PYEOF

NGPU=$(python -c "import torch;print(torch.cuda.device_count())")
# --- crash-safe checkpointing (verified via smoke test). The trainer saves to {dump_path}/{exp_name}/{exp_id};
# the base dump_path defaults to the RELATIVE "checkpoint" dir (=> ephemeral container, lost on crash) and
# exp_id is RANDOM each run (=> resume can't find the prior ckpt). Fix: absolute volume base + FIXED exp_id=v1
# so checkpoint.pth (+ save_periodic=1 -> periodic-N.pth) land on the volume; reload_checkpoint=$CKDIR
# auto-resumes (model+optimizer+epoch+iter) if a prior checkpoint exists. (hydra.run.dir != trainer dump_path.) ---
DUMP="${DUMP:-$EU_VOL/results}"; EXP="p1_prose_sym"; CKDIR="$DUMP/$EXP/v1"; mkdir -p "$CKDIR"
NGPU=$(python -c "import torch;print(torch.cuda.device_count())")
for TAGSPEC in id:0:0 fr:1:0 fc:0:1 r180:1:1; do
  TAG="${TAGSPEC%%:*}"; REST="${TAGSPEC#*:}"; FR="${REST%%:*}"; FC="${REST##*:}"
  cp -f /eu/results/p1_prose_fair/v1/checkpoint.pth "$CKDIR/checkpoint.pth"
  echo "=== FLIP $TAG (FR=$FR FC=$FC) ==="
  set +e
  FLIP_FR=$FR FLIP_FC=$FC FLIPTAG=$TAG torchrun --standalone --nnodes 1 --nproc_per_node "$NGPU" main.py \
    dump_path="$DUMP" exp_name="$EXP" exp_id=v1 save_periodic=0 reload_checkpoint=$CKDIR \
    model=prose_2to1 use_wandb=0 symbol.symbol_input=1 \
    data.t_num=20 input_len=10 data.x_num=128 num_workers=0 num_workers_eval=0 \
    "data.types=[shallow_water,com_ns,incom_ns,incom_ns_arena,cfdbench,incom_ns_arena_u]" \
    data.shallow_water.data_path="$PRE/shallow_water_n100000_t20.npy" \
    +data.com_ns.data_path="$PRE/com_ns_n100000_t20_shards" \
    +data.incom_ns.data_path="$PRE/incom_ns_n100000_t20_shards" \
    +data.incom_ns_arena.data_path="$PRE/pdearena_ns_n100000_t20_shards" \
    +data.incom_ns_arena_u.data_path="$PRE/pdearena_uncond_n100000_t20_shards" \
    ++data.cfdbench.data_path="$PRE/cfdbench_n100000_t20.npy" \
    max_epoch=81 n_steps_per_epoch=2 batch_size=32 2>&1 | grep -E "FULLTEST|DUMP_SAVED|Error|Traceback" | tail -20
  set -e
done
python - <<'CEOF'
import numpy as np, glob, os
fams = sorted(set(os.path.basename(p).split("_", 2)[2][:-4] for p in glob.glob("/eu/_prosesym32_id_*.npz")))
print("families:", fams, flush=True)
def inv(p, tag):
    fr = tag in ("fr", "r180"); fc = tag in ("fc", "r180")
    if fr:
        p = np.flip(p, axis=2).copy()
        if p.shape[-1] >= 2: p[..., 0] = -p[..., 0]
    if fc:
        p = np.flip(p, axis=3).copy()
        if p.shape[-1] >= 2: p[..., 1] = -p[..., 1]
    return p
for fam in fams:
    d0 = np.load(f"/eu/_prosesym32_id_{fam}.npz"); gt = d0["gt"]
    preds = {}
    for tag in ("id", "fr", "fc", "r180"):
        f = f"/eu/_prosesym32_{tag}_{fam}.npz"
        if os.path.exists(f):
            preds[tag] = inv(np.load(f)["pred"], tag)
    def rel(p):
        num = np.sqrt(((p - gt) ** 2).sum((2, 3, 4))); den = np.sqrt((gt ** 2).sum((2, 3, 4))) + 1e-12
        return 100 * (num / den).mean()
    r = {t: rel(p) for t, p in preds.items()}
    base = r["id"]
    legal = [t for t in preds if r[t] < base * 1.15 or t == "id"]
    ens = np.mean([preds[t] for t in legal], axis=0)
    print(f"[PROSE {fam}] per-g:", {t: round(float(v), 3) for t, v in r.items()},
          f" legalK={len(legal)} single={base:.3f} ENS={rel(ens):.3f}", flush=True)
print("COMBINE_DONE", flush=True)
CEOF
echo "=== done ==="
