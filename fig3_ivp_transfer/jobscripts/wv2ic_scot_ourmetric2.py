"""Generic fair-metric scorer for finetuned scOT ckpts: per-sample rel-L1 on the QoI channels in PHYSICAL
space, median over samples of mean over frames (our routine). Args: model_dir ds_string T0 FT qoi mode kw"""
import sys, os, json, numpy as np
sys.path.insert(0, os.getcwd())
from scOT.inference import get_trainer, rollout, get_test_set

model_path, ds, T0, FT, qoi, mode = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
KW = json.loads(sys.argv[7]) if len(sys.argv) > 7 else {}
DATA = os.environ["DATA"]
q0, q1 = [int(x) for x in qoi.split(":")]

def denorm(arr, ts):
    c = ts.constants
    if "mean_solution" in c:
        m = np.array(float(c["mean_solution"])); s = np.array(float(c["std_solution"]))
        return arr[:, q0:q1] * s + m
    m = np.asarray(c["mean"], dtype=np.float64).reshape(-1); s = np.asarray(c["std"], dtype=np.float64).reshape(-1)
    if m.size == 1:
        return arr[:, q0:q1] * s.item() + m.item()
    return arr[:, q0:q1] * s[q0:q1, None, None] + m[q0:q1, None, None]

def relL1(p, r):
    return np.abs(p - r).sum(axis=(1, 2, 3)) / (np.abs(r).sum(axis=(1, 2, 3)) + 1e-12)

trainer = None; E = []
if mode == "steady":
    ts = get_test_set(ds, DATA, dataset_kwargs=KW)
    trainer = get_trainer(model_path, 16, ts, workers=2)
    out = rollout(trainer, ts, ar_steps=1)
    e = relL1(denorm(out[0].astype(np.float64), ts), denorm(out[1].astype(np.float64), ts))
    E.append(e)
else:
    LCAP = int(os.environ.get("LEAD_CAP", "0"))
    tmax = min(FT, T0 + LCAP) if LCAP > 0 else FT
    for t in range(T0 + 1, tmax + 1):
        ts = get_test_set(ds, DATA, initial_time=T0, final_time=t, dataset_kwargs=KW)
        if trainer is None:
            trainer = get_trainer(model_path, 16, ts, workers=2)
        out = rollout(trainer, ts, ar_steps=1)
        e = relL1(denorm(out[0].astype(np.float64), ts), denorm(out[1].astype(np.float64), ts))
        E.append(e)
        print(f"  t={t}: median={np.median(e)*100:.3f}%", flush=True)
E = np.stack(E)
print(f"OURMETRIC2 ds={ds} model={model_path} T0={T0}: median_of_means={np.median(E.mean(axis=0))*100:.3f}%  "
      f"mean_of_medians={np.mean(np.median(E, axis=1))*100:.3f}%  final={np.median(E[-1])*100:.3f}%", flush=True)
