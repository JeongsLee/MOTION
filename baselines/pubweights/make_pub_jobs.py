"""REVISION-R1 11-06: evaluate the RELEASED BCAT (HF felix-lyx/bcat, bcat.pth) and PROSE-FD (HF felix-lyx/prose,
prose_fd.pth) weights on our test-600 (same loader, same symmetry members as the retrained baselines of Fig. 4a,d /
Supplementary Note 3.2). Derived from the release runners bcat_sym_runnerE.sh / symk8_prose_shift.sh: same deps, cache
loader and eval patch; changes: eval_from_exp = released .pth, dump prefix _{bcat,prose}pubE_, general member
transform (transpose -> roll -> flips, FLIP_SIGN), member loop below, per-trajectory metrics at the end
(baseline_pertraj_pub.py).  python make_pub_jobs.py -> {bcat,prose}_pub_runner.sh, {bcat,prose}-pub.json"""
import base64, json, os, re
D = os.path.dirname(os.path.abspath(__file__))
J = "baselines/symmetry_dumps"
ALL = "shallow_water,com_ns,incom_ns,incom_ns_arena,cfdbench,incom_ns_arena_u"
P3 = "shallow_water,com_ns,incom_ns"
P5 = "shallow_water,com_ns,incom_ns,incom_ns_arena,incom_ns_arena_u"
G = {"id": (0, 0, 0), "fr": (0, 1, 0), "fc": (0, 0, 1), "r180": (0, 1, 1), "d1": (1, 0, 0), "r90": (1, 1, 0), "r270": (1, 0, 1), "d2": (1, 1, 1)}
M = []   # TAG T FR FC SR SC SIGN DTYPES
for g, (t, fr, fc) in G.items():
    M.append((g, t, fr, fc, 0, 0, 1, ALL if g == "id" else P5 if g == "fr" else P3))
for s in (1, 8):
    for g, (t, fr, fc) in G.items():
        M.append(("sh%d" % s if g == "id" else "%s_s%d" % (g, s), t, fr, fc, s, s, 1, P3))
M.append(("frns", 0, 1, 0, 0, 0, 0, "cfdbench"))
SPECS = " ".join(":".join(map(str, m)) for m in M)
TRANSFORM = '''            _ft = int(os.environ.get("FLIP_T", "0"))
            if _ft:
                data = np.swapaxes(data, 1, 2).copy()
                if data.shape[-1] >= 2:
                    _sw = list(range(data.shape[-1])); _sw[0], _sw[1] = 1, 0
                    data = data[..., _sw]
            _fr = int(os.environ.get("FLIP_FR", "0")); _fc = int(os.environ.get("FLIP_FC", "0"))
            _sr = int(os.environ.get("SHIFT_R", "0")); _sc = int(os.environ.get("SHIFT_C", "0"))
            if _sr or _sc:
                data = np.roll(data, (_sr, _sc), axis=(1, 2)).copy()
            _sg = int(os.environ.get("FLIP_SIGN", "1"))
            if _fr:
                data = np.flip(data, axis=1).copy()
                if _sg and data.shape[-1] >= 2: data[..., 0] = -data[..., 0]
            if _fc:
                data = np.flip(data, axis=2).copy()
                if _sg and data.shape[-1] >= 2: data[..., 1] = -data[..., 1]
'''
for kind in ("bcat", "prose"):
    if kind == "bcat":
        r = open(f"{J}/bcat_sym_runnerE.sh").read()
        a = r.index('            _fr = int(os.environ.get("FLIP_FR"'); b = r.index("            data = self.augment_data(data)")
        r = r[:a] + TRANSFORM + r[b:]
        # 11-06: same dependency pins as the feature-baseline runner (unpinned deps upgrade torch; flex_attention error)
        i0 = r.index("pip install -q torch-harmonics"); i1 = r.index("\n", r.index("diffusers", i0))
        r = (r[:i0] + "printf 'torch==2.7.1\\ntorchvision==0.22.1\\ntorchaudio==2.7.1\\n' > /tmp/pin_torch.txt\n"
             "pip install -q -c /tmp/pin_torch.txt torch-harmonics einops hydra-core omegaconf h5py tabulate sympy pandas \\\n"
             "  rotary-embedding-torch dadaptation torchinfo torchmetrics scipy scikit-learn tqdm wandb transformers==4.46.3 \\\n"
             "  matplotlib seaborn neuraloperator accelerate torchtune==0.6.1 torchao==0.11.0 diffusers 2>&1 | tail -2" + r[i1:])
        head = r[: r.index("for TAGSPEC in")]
        args = ('model=bcat use_wandb=0 symbol.symbol_input=0 data.tie_fields=1 data.t_num=20 input_len=10 data.x_num=128 '
                '"data.types=[$DT]" ++data.shallow_water.data_path="$PRE/shallow_water_n100000_t20.npy" '
                '++data.com_ns.data_path="$PRE/com_ns_n100000_t20_shards" ++data.incom_ns.data_path="$PRE/incom_ns_n100000_t20_shards" '
                '++data.incom_ns_arena.data_path="$PRE/pdearena_ns_n100000_t20_shards" ++data.cfdbench.data_path="$PRE/cfdbench_n100000_t20.npy" '
                '++data.incom_ns_arena_u.data_path="$PRE/pdearena_uncond_n100000_t20_shards" data.mixed_length=20 batch_size=32')
        fix = open(f"{J}/bcat_fix_eval_symE.py").read().replace("/eu/_bcatsymE_", "/eu/_bcatpubE_")
        img = "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime"
    else:
        r = open(f"{J}/symk8_prose_shift.sh").read()
        head = r[: r.index("for TAGSPEC in")]
        args = ('model=prose_2to1 use_wandb=0 symbol.symbol_input=1 data.t_num=20 input_len=10 data.x_num=128 num_workers=0 num_workers_eval=0 '
                '"data.types=[$DT]" data.shallow_water.data_path="$PRE/shallow_water_n100000_t20.npy" '
                '+data.com_ns.data_path="$PRE/com_ns_n100000_t20_shards" +data.incom_ns.data_path="$PRE/incom_ns_n100000_t20_shards" '
                '+data.incom_ns_arena.data_path="$PRE/pdearena_ns_n100000_t20_shards" +data.incom_ns_arena_u.data_path="$PRE/pdearena_uncond_n100000_t20_shards" '
                '++data.cfdbench.data_path="$PRE/cfdbench_n100000_t20.npy" batch_size=32')
        fix = open(f"{J}/prose_fix_eval_symE.py").read().replace("/eu/_prosesymE_", "/eu/_prosepubE_")
        img = "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime"
    assert "pubE_" in fix
    head = head.replace('EXP="p1_bcat_sym"', 'EXP="p1_bcat_pub"').replace('EXP="p1_prose_sym"', 'EXP="p1_prose_pub"')
    loop = f'''
CK=/work/pub_weights/{"bcat.pth" if kind == "bcat" else "prose_fd.pth"}; ls -la $CK
for SPEC in {SPECS}; do
  IFS=: read -r TAG T FR FC SR SC SG DT <<< "$SPEC"
  echo "=== MEMBER $TAG (T=$T FR=$FR FC=$FC SR=$SR SC=$SC SIGN=$SG) [$DT] ==="
  set +e
  FLIP_T=$T FLIP_FR=$FR FLIP_FC=$FC SHIFT_R=$SR SHIFT_C=$SC FLIP_SIGN=$SG FLIPTAG=$TAG DT=$DT torchrun --standalone --nnodes 1 --nproc_per_node "$NGPU" main.py \\
    dump_path="$DUMP" exp_name="$EXP" exp_id=v1 save_periodic=0 eval_only=1 eval_from_exp=$CK {args.replace("$DT", "$DT")} \\
    < /dev/null 2>&1 | grep -E "FULLTEST|DUMP_SAVED|Error|Traceback|Reloading|missing|Unexpected" | tail -12
  set -e
done
cd /tmp && pip install -q "numpy<2" scipy 2>&1 | tail -1
python /work/pub_weights/baseline_pertraj_pub.py /eu /work/pub_weights/{kind}_pub_pertraj.json {kind}
echo PUB_EVAL_DONE
'''
    r = head + loop
    open(f"{D}/{kind}_pub_runner.sh", "w").write(r)
    cmd = ("echo " + base64.b64encode(fix.encode()).decode() + " | base64 -d > /tmp/fix_eval3.py && echo "
           + base64.b64encode(r.encode()).decode() + " | base64 -d | CODE_VOL=/code-vol EU_VOL=/eu bash")
    job = {"name": f"{kind}-pub-eval", "imageUri": img, "resourceSpecSlug": "resourcespec-ch100x1",
           "clusterSlug": "cluster-capella", "clusterVolumes": [{"clusterVolumeSlug": "<cluster-volume>", "mountPath": "/eu"}],
           "objectVolumes": [{"mountPath": "/code-vol", "objectVolumeSlug": "<object-volume>"}],
           "imagePullPolicy": "Always", "organizationName": "<organization>", "teamName": "Default", "command": cmd}
    json.dump(job, open(f"{D}/{kind}-pub.json", "w"), indent=1)
    print(kind, "members", len(M), "runner", len(r))
