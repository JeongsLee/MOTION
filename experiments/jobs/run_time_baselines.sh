#!/bin/bash
# REVISION-R1 (10-22; R2-3 / R3-3 cost table): inference time per trajectory and peak GPU memory of the retrained
# baselines BCAT-fair (job-xcolmv996mh3, /eu/results/p1_bcat_fair/v1, model=bcat, data-only) and PROSE-FD-fair
# (job-1mzdbmiazal4, /eu/results/p1_prose_fair/v1, model=prose_2to1, symbol_input=1) on test-600
# (first 100 trajectories of the last-10% region, six families), 1 H100, batch 8.
# Reuses the eval-only path of the sym runners (bcat_sym_runnerE.sh / prose_sym_runnerE.sh) with the identity
# transform only (no FLIP, no dumps). Timing = the native evaluate() model call (BCAT: 10-frame KV-cached AR
# generate; PROSE-FD: one-shot generate of all output frames), upstream amp=1 (bf16 autocast), wrapped in
# torch.cuda.synchronize() timers. Per family: one untimed warm-up call on the first batch, then
# reset_peak_memory_stats(), then every batch (incl. the first) timed. Data loading / prepare_data excluded.
# Peak = torch.cuda.max_memory_allocated after the warm-up (weights included, as in the MOTION measurement).
set -o pipefail
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
PRE=/eu/data/prose/prebuilt
OUT=/eu/results/time_baselines; mkdir -p $OUT
nvidia-smi --query-gpu=name,memory.total --format=csv | tee $OUT/gpu.txt
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" | tee -a $OUT/gpu.txt
for d in /eu/results/p1_bcat_fair/v1 /eu/results/p1_prose_fair/v1; do echo "== $d"; ls -la $d/*.pth 2>&1 | tee -a $OUT/ckpt_ls.txt; done

echo "=== deps (BCAT-fair set; superset of PROSE-fair) ==="
which git >/dev/null 2>&1 || (apt-get update -q && apt-get install -y -q git) 2>&1 | tail -1
# 10-22 rerun: pin the image's torch and numpy<2 (first run: an unpinned dependency moved torch -> missing
# torch._inductor.kernel.flex_attention for BCAT; numpy 2 removed np.infty used by PROSE-FD)
TV=$(python -c "import torch; print(torch.__version__.split('+')[0])"); printf "torch==$TV\n" > /tmp/constr.txt   # numpy pinned <2 only before PROSE-FD (pyarrow for BCAT needs numpy 2)
echo "torch pinned to $TV"; export PIP_CONSTRAINT=/tmp/constr.txt
pip install -q torch-harmonics einops hydra-core omegaconf h5py tabulate sympy pandas \
  rotary-embedding-torch dadaptation torchinfo torchmetrics scipy scikit-learn tqdm wandb transformers \
  matplotlib seaborn neuraloperator accelerate torchtune==0.6.1 torchao==0.12.0 diffusers 2>&1 | tail -2
pip uninstall -y torchdata >/dev/null 2>&1 || true
pip install -q --no-deps torchdata==0.7.1 2>&1 | tail -1      # datapipes API (re-pinned LAST, as in the BCAT runner)
python -c "import torchdata.datapipes; print('torchdata.datapipes OK')" || echo "WARN torchdata.datapipes MISSING"
python -c "import torch, numpy; print('torch', torch.__version__, 'numpy', numpy.__version__)"

# ---- FULLTEST_RELNORM eval patch (fix_eval3, NO dumps): per-family phys rel-L2 as a checkpoint sanity check ----
cat > /tmp/fix_eval3.py <<'PYEOF'
import io
p = 'evaluate.py'
s = io.open(p).read()
if 'FULLTEST_RELNORM' not in s:
    anchor = '\n                # computing eval metrics'
    snippet = '''
                if True:  # FULLTEST_RELNORM: our-formula running mean over the FULL test set
                    import numpy as _np
                    if not hasattr(self, "_ft_acc"): self._ft_acc = {}
                    _dim = data_output.shape[-1]
                    _o = (data_output.float() * d["std"] + d["mean"]) if self.params.normalize else data_output.float()
                    _o = _o.cpu().numpy(); _t = samples["data"][..., :_dim].cpu().numpy()[:, -_o.shape[1]:]
                    _num = _np.sqrt(((_o - _t) ** 2).sum((2, 3, 4))); _den = _np.sqrt((_t ** 2).sum((2, 3, 4))) + 1e-12
                    _a = self._ft_acc.setdefault(type, [])
                    _a.append((_num / _den).mean(1))
                    _c = _np.concatenate(_a)
                    print(f"FULLTEST_RELNORM {type} n={len(_c)} mean={float(_c.mean())*100:.3f}%", flush=True)
'''
    ra = 'all_results = {}'
    k = s.index(ra)
    ls = s.rindex('\n', 0, k) + 1
    ind = s[ls:k]
    s = s[:k + len(ra)] + '\n' + ind + 'self._ft_acc = {}' + s[k + len(ra):]
    i = s.index(anchor)
    s = s[:i] + snippet + s[i:]
    io.open(p, 'w').write(s)
print('EVALPATCH3 OK')
PYEOF

# ---- timing patch: wrap the first model(...) call of evaluate() and report at the end of each family ----
cat > /tmp/time_patch.py <<'PYEOF'
import io, re
p = 'evaluate.py'
s = io.open(p).read()
m = re.search(r'\n(\s*)data_output = model\((mode|"generate"), \*\*model_input\)', s)
assert m, 'model call not found'
ind, arg = m.group(1), m.group(2)
s = s[:m.start()] + f'\n{ind}data_output = _TIMED(self, type, bs, lambda: model({arg}, **model_input))' + s[m.end():]
a = 'results["size"] = eval_size'
k = s.index(a); ls = s.rindex('\n', 0, k) + 1
s = s[:k + len(a)] + '\n' + s[ls:k] + '_TIME_REPORT(type)' + s[k + len(a):]
s += '''

# ===== REVISION-R1 timing helpers =====
import time as _tm
_TM = {}
def _TIMED(ev, fam, bs, fn):
    st = _TM.get(fam)
    if st is None:
        if "_params" not in _TM:
            _TM["_params"] = sum(q.numel() for q in ev.modules["model"].parameters())
            print(f"[time-params] {_TM['_params']/1e6:.1f}M parameters", flush=True)
        torch.cuda.synchronize(); _t0 = _tm.perf_counter()
        fn()                                           # warm-up batch (untimed)
        torch.cuda.synchronize()
        st = _TM[fam] = {"warm": _tm.perf_counter() - _t0, "n": 0, "t": 0.0, "nf": 0, "tf": 0.0,
                         "B": int(ev.params.batch_size_eval)}
        torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); _t0 = _tm.perf_counter()
    out = fn()
    torch.cuda.synchronize(); dt = _tm.perf_counter() - _t0
    st["n"] += bs; st["t"] += dt
    if bs == st["B"]:
        st["nf"] += bs; st["tf"] += dt
    return out
def _TIME_REPORT(fam):
    st = _TM.get(fam)
    if not st: return
    pk = torch.cuda.max_memory_allocated() / 2**30; rs = torch.cuda.max_memory_reserved() / 2**30
    lab = __import__("os").environ.get("TIME_MODEL", "MODEL")
    print(f"[time {lab} {fam}] n={st['n']} batch {st['B']}: {st['t']/max(st['n'],1):.4f} s/traj (forward, warm) | "
          f"full batches {st['tf']/max(st['nf'],1):.4f} s/traj (n={st['nf']}) | warm-up batch {st['warm']:.2f} s | "
          f"peak {pk:.2f} GiB (reserved {rs:.2f})", flush=True)
'''
io.open(p, 'w').write(s)
print('TIMEPATCH OK')
PYEOF

FILT='^\[time|FULLTEST_RELNORM .* n=100 |Reloading|Error|Traceback|MEM:'
COMMON="data.t_num=20 input_len=10 data.x_num=128 batch_size=8 batch_size_eval=8 eval_only=1 save_periodic=0 use_wandb=0 dump_path=$OUT exp_id=v1"
TYPES="data.types=[shallow_water,com_ns,incom_ns,incom_ns_arena,cfdbench,incom_ns_arena_u]"

# =========================== BCAT-fair ===========================
echo "=== BCAT ===" | tee -a $OUT/time.log
cd /tmp && rm -rf bc && git clone --depth 1 https://github.com/felix-lyx/bcat bc 2>&1 | tail -1
cd /tmp/bc/src && git log -1 --format='bcat %h %cd' | tee -a $OUT/time.log
python /tmp/fix_eval3.py && python /tmp/time_patch.py
grep -rlE "import (FNO2d|FNO3d)" --include=*.py . | while read f; do
  sed -i -E 's/.*from neuralop[^ ]* import.*FNO[23]d.*/FNO2d = None; FNO3d = None  # stub (model=bcat)/' "$f"; done
cat >> data_utils/all_datasets.py <<'PYEOF'

# ===== OUR-CACHE loader (data-only; = bcat_sym_runnerE.sh without FLIP) =====
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
        if not train:  # test-600 (first 100 rows of the last-10% region)
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
               else self.get_iter_range(len(self._index))[self.local_rank :: self.n_gpu_per_node])
        if self.train:
            rng = self.rng.permutation(rng)
        _cf = {}
        for k in rng:
            fi, i = self._index[int(k)]
            arr = _cf.get(fi)
            if arr is None:
                arr = np.load(self.files[fi], mmap_mode="r"); _cf = {fi: arr}
            _vt = min(self.VALID_T, self.t_num) if getattr(self, "VALID_T", None) else self.t_num
            data = np.asarray(arr[i])[: _vt]
            data = self.augment_data(data)
            data = torch.from_numpy(np.ascontiguousarray(data)).float()
            yield {"data": data}
class CachedShallow(CachedNpy2D): CACHE_TL = "shallow_water"
class CachedCom(CachedNpy2D): CACHE_TL = "com_ns"
class CachedIncom(CachedNpy2D): CACHE_TL = "incom_ns"
class CachedArena(CachedNpy2D): CACHE_TL = "incom_ns_arena"
class CachedCFD(CachedNpy2D): CACHE_TL = "cfdbench"
class CachedArenaU(CachedNpy2D): CACHE_TL = "incom_ns_arena_u"; VALID_T = 14
PYEOF
python - <<'PYEOF'
p = "dataset.py"; s = open(p).read()
i = s.index("ALL_DATASETS"); j = s.index("}", i)
inj = ("\nALL_DATASETS['shallow_water']=ds.CachedShallow; ALL_DATASETS['com_ns']=ds.CachedCom\n"
       "ALL_DATASETS['incom_ns']=ds.CachedIncom; ALL_DATASETS['incom_ns_arena']=ds.CachedArena\n"
       "ALL_DATASETS['cfdbench']=ds.CachedCFD; ALL_DATASETS['incom_ns_arena_u']=ds.CachedArenaU\n")
open(p, "w").write(s[:j+1] + "\n" + inj + s[j+1:]); print("remapped ALL_DATASETS (BCAT)")
PYEOF
( TIME_MODEL=BCAT torchrun --standalone --nnodes 1 --nproc_per_node 1 main.py $COMMON \
    exp_name=p1_time_bcat eval_from_exp=/eu/results/p1_bcat_fair/v1 \
    model=bcat symbol.symbol_input=0 data.tie_fields=1 "$TYPES" \
    ++data.shallow_water.data_path="$PRE/shallow_water_n100000_t20.npy" \
    ++data.com_ns.data_path="$PRE/com_ns_n100000_t20_shards" \
    ++data.incom_ns.data_path="$PRE/incom_ns_n100000_t20_shards" \
    ++data.incom_ns_arena.data_path="$PRE/pdearena_ns_n100000_t20_shards" \
    ++data.cfdbench.data_path="$PRE/cfdbench_n100000_t20.npy" \
    ++data.incom_ns_arena_u.data_path="$PRE/pdearena_uncond_n100000_t20_shards" \
    data.mixed_length=20 2>&1 | grep --line-buffered -E "$FILT" | tee -a $OUT/time.log ) || true

# =========================== PROSE-FD-fair ===========================
echo "=== PROSE-FD ===" | tee -a $OUT/time.log
if [ "${SKIP_PROSE:-0}" = 1 ]; then echo "PROSE skipped"; echo TIME_BASELINES_DONE | tee $OUT/done.txt; exit 0; fi
pip install -q "numpy<2" 2>&1 | tail -1    # PROSE-FD uses np.infty
cd /tmp && rm -rf pf && git clone --depth 1 https://github.com/felix-lyx/prose pf 2>&1 | tail -1
cd /tmp/pf/prose_fd && git log -1 --format='prose %h %cd' | tee -a $OUT/time.log
python /tmp/fix_eval3.py && python /tmp/time_patch.py
sed -i 's/^from neuralop.models import FNO3d/FNO3d = None/' models/baselines.py
# torch>=2.6 defaults torch.load(weights_only=True); PROSE-fair trained on torch 2.4 -> restore the old default
sed -i 's/reloaded = torch.load(params.reload_model)/reloaded = torch.load(params.reload_model, weights_only=False)/' models/build_model.py
grep -n "weights_only=False" models/build_model.py || echo "WARN weights_only patch not applied"
cat >> data_utils/all_datasets.py <<'PYEOF'

# ===== OUR-CACHE loader (= prose_sym_runnerE.sh without FLIP) =====
import glob as _glob
class CachedNpy2D(myIterDp):
    CACHE_TL = None
    VALID_T = None
    def __init__(self, params, symbol_env, split="train", train=True, file_idx=-1):
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
        if not train:  # test-600 (first 100 rows of the last-10% region)
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
               else self.get_iter_range(len(self._index))[self.local_rank :: self.n_gpu_per_node])
        if self.train:
            rng = self.rng.permutation(rng)
        _cf = {}
        for k in rng:
            fi, i = self._index[int(k)]
            arr = _cf.get(fi)
            if arr is None:
                arr = np.load(self.files[fi], mmap_mode="r"); _cf = {fi: arr}
            _vt = min(self.VALID_T, self.t_num) if self.VALID_T else self.t_num
            data = np.asarray(arr[i])[: _vt]
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
class CachedArenaU(CachedNpy2D): CACHE_TL = "incom_ns_arena_u"; VALID_T = 14
class CachedCFD(CachedNpy2D): CACHE_TL = "cfdbench"
PYEOF
cat >> dataset.py <<'PYEOF'

ALL_DATASETS["shallow_water"] = ds.CachedShallow
ALL_DATASETS["com_ns"] = ds.CachedComNS
ALL_DATASETS["incom_ns"] = ds.CachedIncomNS
ALL_DATASETS["incom_ns_arena"] = ds.CachedArena
ALL_DATASETS["incom_ns_arena_u"] = ds.CachedArenaU
ALL_DATASETS["cfdbench"] = ds.CachedCFD
PYEOF
( TIME_MODEL=PROSE-FD torchrun --standalone --nnodes 1 --nproc_per_node 1 main.py $COMMON \
    exp_name=p1_time_prose eval_from_exp=/eu/results/p1_prose_fair/v1 \
    model=prose_2to1 symbol.symbol_input=1 num_workers=0 num_workers_eval=0 "$TYPES" \
    data.shallow_water.data_path="$PRE/shallow_water_n100000_t20.npy" \
    +data.com_ns.data_path="$PRE/com_ns_n100000_t20_shards" \
    +data.incom_ns.data_path="$PRE/incom_ns_n100000_t20_shards" \
    +data.incom_ns_arena.data_path="$PRE/pdearena_ns_n100000_t20_shards" \
    +data.incom_ns_arena_u.data_path="$PRE/pdearena_uncond_n100000_t20_shards" \
    ++data.cfdbench.data_path="$PRE/cfdbench_n100000_t20.npy" \
    2>&1 | grep --line-buffered -E "$FILT" | tee -a $OUT/time.log ) || true
echo TIME_BASELINES_DONE | tee $OUT/done.txt
