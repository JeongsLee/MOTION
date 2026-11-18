"""Feature-group knockout evaluation of the B1/P1 feature baselines (2026-11-05).

  python make_ko_jobs.py   -> bcat-feat-ko.json, prose-feat-ko.json (next to this file)

Each job: same image, deps and patch steps as the training runner (prefix of {kind}_feat_runner.sh up to the
checkpoint block), then eval_only on the FINAL checkpoint (/eu/results/p1_{kind}_feat/v1/checkpoint.pth, no selection,
as in the benchmark) six times: no knockout, each feature group (grad, lap, adv, div) and all four groups.
Knockout = zeroing the group's channels inside basefeat.features (BASEFEAT_KO); raw input channels are untouched.
Batch size 4 (batch_size_eval 6), as in training, so the no-knockout run must reproduce the last in-training FULLTEST.
Output: /eu/results/p1_{kind}_feat/ko/ko_<groups>.log; compact SUMMARY lines (cumulative per-family FULLTEST means)
are echoed to stdout because job logs keep only the last 1000 lines.
"""
import base64
import json
import os

D = os.path.dirname(os.path.abspath(__file__))
ARGS = {
    "prose": ('model=prose_2to1 use_wandb=0 symbol.symbol_input=1 data.t_num=20 input_len=10 data.x_num=128 '
              'num_workers=0 num_workers_eval=0 "data.types=[$DTYPES]" '
              'data.shallow_water.data_path="$PRE/shallow_water_n100000_t20.npy" '
              '+data.com_ns.data_path="$PRE/com_ns_n100000_t20_shards" '
              '+data.incom_ns.data_path="$PRE/incom_ns_n100000_t20_shards" '
              '+data.incom_ns_arena.data_path="$PRE/pdearena_ns_n100000_t20_shards" '
              '+data.incom_ns_arena_u.data_path="$PRE/pdearena_uncond_n100000_t20_shards" '
              '++data.cfdbench.data_path="$PRE/cfdbench_n100000_t20.npy" batch_size=4'),
    "bcat": ('model=bcat use_wandb=0 symbol.symbol_input=0 data.tie_fields=1 data.t_num=20 input_len=10 data.x_num=128 '
             '"data.types=[$DTYPES]" '
             '++data.shallow_water.data_path="$PRE/shallow_water_n100000_t20.npy" '
             '++data.com_ns.data_path="$PRE/com_ns_n100000_t20_shards" '
             '++data.incom_ns.data_path="$PRE/incom_ns_n100000_t20_shards" '
             '++data.incom_ns_arena.data_path="$PRE/pdearena_ns_n100000_t20_shards" '
             '++data.cfdbench.data_path="$PRE/cfdbench_n100000_t20.npy" '
             '++data.incom_ns_arena_u.data_path="$PRE/pdearena_uncond_n100000_t20_shards" '
             'data.mixed_length=20 batch_size=4'),
}
EVAL = r'''
# ===== feature-group knockout evaluation (make_ko_jobs.py) =====
CK="$EU_VOL/results/$EXP/v1/checkpoint.pth"; OUT="$EU_VOL/results/$EXP/ko"; mkdir -p "$OUT"
ls -la "$CK" || { echo "NO CHECKPOINT"; exit 3; }
python - "$CK" <<'PY'
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
for k, v in sd.items():
    if isinstance(v, dict):
        for n, t in v.items():
            if hasattr(t, "dim") and t.dim() == 4 and ("embed" in n or "patch" in n):
                print("CKPT_CONV", k, n, tuple(t.shape))
PY
for KO in none grad lap adv div grad,lap,adv,div; do
  TAG=${KO//,/_}
  if [ "$KO" = none ]; then unset BASEFEAT_KO; else export BASEFEAT_KO=$KO; fi
  torchrun --standalone --nnodes 1 --nproc_per_node 1 main.py dump_path="$OUT" exp_name="ko_$TAG" exp_id=v1 \
    eval_only=1 eval_from_exp="$CK" __ARGS__ < /dev/null > "$OUT/ko_$TAG.log" 2>&1
  echo "=== KO=$KO rc=$? ==="
  grep -E "Eval \||Error|Traceback" "$OUT/ko_$TAG.log" | tail -3
  grep FULLTEST_RELNORM "$OUT/ko_$TAG.log" | awk -v t="$TAG" '{f=$2; m=$4; sub("mean=","",m); sub("%","",m); n=$3; sub("n=","",n);
     s[f]=s[f] n ":" m ","} END {for (f in s) print "SUMMARY", t, f, s[f]}'
done
echo "=== ko eval done ==="
'''

for kind in ("prose", "bcat"):
    r = open(f"{D}/{kind}_feat_runner.sh").read()
    r = r[: r.index('DUMP="${DUMP:-')] + EVAL.replace("__ARGS__", ARGS[kind])
    j = json.load(open(f"{D}/{kind}-feat.json"))
    fix = j["command"].split(" | base64 -d > /tmp/fix_eval3.py")[0]          # same eval patch as training
    j["name"] = f"{kind}-feat-ko"
    j["resourceSpecSlug"] = "resourcespec-ch100x1"
    j["command"] = (fix + " | base64 -d > /tmp/fix_eval3.py && echo " + base64.b64encode(r.encode()).decode()
                    + " | base64 -d | CODE_VOL=/eu/code_mirror EU_VOL=/eu BASEFEAT_DIR=/work/basefeat "
                    "DTYPES=shallow_water,com_ns,incom_ns,incom_ns_arena,cfdbench,incom_ns_arena_u "
                    f"EXP=p1_{kind}_feat bash")
    open(f"{D}/{kind}_ko_runner.sh", "w").write(r)
    json.dump(j, open(f"{D}/{kind}-feat-ko.json", "w"), indent=1)
    print(kind, "ok", len(r))
