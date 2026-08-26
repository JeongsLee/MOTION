"""Poisson-Gauss scOT re-dump, split-aligned: scOT native test=240 (last 240); our shared
test = last 128 = scOT test indices 112..239. Dump preds for its indices 112..127
(= our test samples 0..15) + per-sample E over the aligned 128."""
import sys, os, numpy as np
sys.path.insert(0, os.getcwd())
from scOT.inference import get_trainer, rollout, get_test_set
model_path = sys.argv[1]; out = sys.argv[2]
DATA = os.environ["DATA"]
ts = get_test_set("elliptic.poisson.Gaussians", DATA)
trainer = get_trainer(model_path, 16, ts, workers=2)
o = rollout(trainer, ts, ar_steps=1)
c = ts.constants
m = np.array(float(c["mean_solution"])) if "mean_solution" in c else np.asarray(c["mean"]).reshape(-1)[0]
s = np.array(float(c["std_solution"])) if "mean_solution" in c else np.asarray(c["std"]).reshape(-1)[0]
p = o[0].astype(np.float64)[:, 0:1]*s + m; r = o[1].astype(np.float64)[:, 0:1]*s + m
E = np.abs(p-r).sum(axis=(1, 2, 3))/(np.abs(r).sum(axis=(1, 2, 3))+1e-12)
print("n_test:", len(E))
print(f"median over ALL {len(E)}: {np.median(E)*100:.3f}%")
print(f"median over aligned last-128 (idx 112..239): {np.median(E[112:240])*100:.3f}%")
np.savez_compressed(out, pred16=p[112:128, None].astype(np.float32),
                    ref16=r[112:128, None].astype(np.float32),
                    E=E[None, 112:240].astype(np.float32))
print("DUMP_DONE", out)
