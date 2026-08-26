"""Fig3 per-family qual dump — scOT/Poseidon-B side. argv: model_dir ds T0 FT qoi mode out.npz [kw_json]
Extends the recovered scot_ourmetric2.py: same protocol, additionally dumps denormed
pred/ref trajectories for the first 16 test samples + per-sample metrics for all 128.
For NS also runs the native AR mode (+2 rollout: final_time=t with ar_steps=t/2)."""
import sys, os, json, numpy as np
sys.path.insert(0, os.getcwd())
from scOT.inference import get_trainer, rollout, get_test_set

model_path, ds, T0, FT, qoi, mode, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6], sys.argv[7]
KW = json.loads(sys.argv[8]) if len(sys.argv) > 8 else {}
DO_AR = int(os.environ.get("DO_AR", "0"))
DATA = os.environ["DATA"]
q0, q1 = [int(x) for x in qoi.split(":")]

def denorm(arr, ts):
    c = ts.constants
    if "mean_solution" in c:
        m = np.array(float(c["mean_solution"])); s = np.array(float(c["std_solution"]))
        return arr[:, q0:q1]*s + m
    m = np.asarray(c["mean"], dtype=np.float64).reshape(-1); s = np.asarray(c["std"], dtype=np.float64).reshape(-1)
    if m.size == 1:
        return arr[:, q0:q1]*s.item() + m.item()
    return arr[:, q0:q1]*s[q0:q1, None, None] + m[q0:q1, None, None]

def relL1(p, r):
    return np.abs(p-r).sum(axis=(1, 2, 3))/(np.abs(r).sum(axis=(1, 2, 3))+1e-12)

trainer = None; E = []; P16 = []; R16 = []
save = {}
if mode == "steady":
    ts = get_test_set(ds, DATA, dataset_kwargs=KW)
    trainer = get_trainer(model_path, 16, ts, workers=2)
    o = rollout(trainer, ts, ar_steps=1)
    p = denorm(o[0].astype(np.float64), ts); r = denorm(o[1].astype(np.float64), ts)
    E.append(relL1(p, r)); P16.append(p[:16]); R16.append(r[:16])
else:
    for t in range(T0+1, FT+1):
        ts = get_test_set(ds, DATA, initial_time=T0, final_time=t, dataset_kwargs=KW)
        if trainer is None:
            trainer = get_trainer(model_path, 16, ts, workers=2)
        o = rollout(trainer, ts, ar_steps=1)
        p = denorm(o[0].astype(np.float64), ts); r = denorm(o[1].astype(np.float64), ts)
        E.append(relL1(p, r)); P16.append(p[:16]); R16.append(r[:16])
        print(f"  t={t}: median={np.median(E[-1])*100:.3f}%", flush=True)
E = np.stack(E)
print(f"OURMETRIC2 ds={ds} T0={T0}: median_of_means={np.median(E.mean(axis=0))*100:.3f}%  "
      f"final={np.median(E[-1])*100:.3f}%", flush=True)
save.update(pred16=np.stack(P16, 1).astype(np.float32), ref16=np.stack(R16, 1).astype(np.float32),
            E=E.astype(np.float32))
if DO_AR:
    EA = []; PA16 = []
    for t in range(2, FT+1, 2):
        ts = get_test_set(ds, DATA, initial_time=0, final_time=t, dataset_kwargs=KW)
        o = rollout(trainer, ts, ar_steps=t//2)
        p = denorm(o[0].astype(np.float64), ts); r = denorm(o[1].astype(np.float64), ts)
        EA.append(relL1(p, r)); PA16.append(p[:16])
        print(f"  AR t={t} (steps={t//2}): median={np.median(EA[-1])*100:.3f}%", flush=True)
    EA = np.stack(EA)
    print(f"OURMETRIC2-AR ds={ds}: median_of_means={np.median(EA.mean(axis=0))*100:.3f}%", flush=True)
    save.update(predAR16=np.stack(PA16, 1).astype(np.float32), EAR=EA.astype(np.float32))
np.savez_compressed(out, **save)
print("DUMP_DONE", out, flush=True)
