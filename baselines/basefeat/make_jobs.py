"""Build the B1/P1 training job files from the feature runners (2026-11-05, pinned dependencies).

  python make_jobs.py   -> bcat-feat.json, prose-feat.json (next to this file)
Same image, resource spec, budget and EXP names as the first attempt (2026-11-04); only the runner text changed.
"""
import base64
import json
import os

D = os.path.dirname(os.path.abspath(__file__))
J = "baselines/retrained_bcat_prose"
fix = base64.b64encode(open(f"{D}/fix_eval3.py", "rb").read()).decode()
for kind in ("bcat", "prose"):
    j = json.load(open(f"{J}/{kind}-fair.json"))
    run = base64.b64encode(open(f"{D}/{kind}_feat_runner.sh", "rb").read()).decode()
    j["name"] = f"{kind}-feat"
    j["command"] = (f"echo {fix} | base64 -d > /tmp/fix_eval3.py && echo {run} | base64 -d | "
                    "CODE_VOL=/eu/code_mirror EU_VOL=/eu BASEFEAT_DIR=/work/basefeat "
                    "DTYPES=shallow_water,com_ns,incom_ns,incom_ns_arena,cfdbench,incom_ns_arena_u EP=80 NSTEPS=2000 BS=4 "
                    f"EXP=p1_{kind}_feat SAVE_PERIODIC=0 bash")
    json.dump(j, open(f"{D}/{kind}-feat.json", "w"), indent=1)
    print(kind, j["resourceSpecSlug"], j["imageUri"])
