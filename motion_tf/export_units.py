"""Export the first N test units of selected families (active slots only) for running the latent scripts locally.
  python export_units.py OUT.pkl [--fams incom_ns,NS-Sines,com_ns] [--n 16]"""
import argparse, itertools, pickle
import numpy as np
from data import loader
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--fams", default="incom_ns,NS-Sines,com_ns")
ap.add_argument("--n", type=int, default=16); a = ap.parse_args()
res = {}
for fam in a.fams.split(","):
    us = []
    for u in itertools.islice(loader.units_for(fam, "test"), a.n):
        u = dict(u); f = np.asarray(u["fields"]); S = f.shape[-1]
        act = np.where(np.abs(f.reshape(-1, S)).max(0) > 0)[0]
        u["fields"] = np.ascontiguousarray(f[..., act]); u["_act"] = act; u["_S"] = S; us.append(u)
    res[fam] = us; print(fam, len(us), us[0]["fields"].shape, us[0]["_act"], flush=True)
pickle.dump(res, open(a.out, "wb"), protocol=4); print("EXPORT_OK")
