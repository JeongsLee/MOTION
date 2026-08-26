"""scOT TTA extras: (a) NS AR(+2 rollout) x flip ensemble; (b) Wave 2icx4 repaired
anchor-swept (lead<=12) x flip ensemble. argv: mode model_dir out"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.getcwd())
from scOT.inference import get_trainer, rollout, get_test_set
mode, model_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
DATA = os.environ["DATA"]; VEL = int(os.environ.get("VEL", "0"))
GROUP = [(0,0),(1,0),(0,1),(1,1)]
def tg(x, g):
    fr, fc = g
    if fr:
        x = torch.flip(x, dims=[-2]).clone()
        if VEL: x[:, 0] = -x[:, 0]
    if fc:
        x = torch.flip(x, dims=[-1]).clone()
        if VEL: x[:, 1] = -x[:, 1]
    return x
class Wrap(torch.nn.Module):
    def __init__(self, mdl, g):
        super().__init__(); object.__setattr__(self, "_wrapped", mdl); self.m = mdl; self.g = g
    def __getattr__(self, name):
        try: return super().__getattr__(name)
        except AttributeError: return getattr(object.__getattribute__(self, "_wrapped"), name)
    def forward(self, pixel_values=None, time=None, **kw):
        o = self.m(pixel_values=tg(pixel_values, self.g), time=time, **kw)
        o.output = tg(o.output, self.g); return o
def denorm(arr, ts, q0, q1):
    c = ts.constants
    if "mean_solution" in c:
        return arr[:, q0:q1]*float(c["std_solution"]) + float(c["mean_solution"])
    mn = np.asarray(c["mean"], np.float64).reshape(-1); sd = np.asarray(c["std"], np.float64).reshape(-1)
    if mn.size == 1: return arr[:, q0:q1]*sd.item() + mn.item()
    return arr[:, q0:q1]*sd[q0:q1, None, None] + mn[q0:q1, None, None]
def relL1(p, r): return np.abs(p-r).sum(axis=(1,2,3))/(np.abs(r).sum(axis=(1,2,3))+1e-12)

if mode == "nsar":
    ds = "fluids.incompressible.PiecewiseConstants"; KW = {"just_velocities": True}; q0, q1 = 0, 2
    trainer = None; Eg = {g: [] for g in GROUP}; Rs = []
    for t in range(2, 21, 2):
        ts = get_test_set(ds, DATA, initial_time=0, final_time=t, dataset_kwargs=KW)
        if trainer is None:
            trainer = get_trainer(model_path, 16, ts, workers=2); base = trainer.model
        for g in GROUP:
            trainer.model = Wrap(base, g)
            o = rollout(trainer, ts, ar_steps=t//2)
            Eg[g].append(denorm(o[0].astype(np.float64), ts, q0, q1))
            if g == (0,0): Rs.append(denorm(o[1].astype(np.float64), ts, q0, q1))
        print(f"  t={t} done", flush=True)
    R = np.stack(Rs, 1); Pg = {g: np.stack(v, 1) for g, v in Eg.items()}
    ens = np.mean([Pg[g] for g in GROUP], axis=0)
    for arr, tag in [(Pg[(0,0)], "AR single"), (ens, "AR ENSEMBLE")]:
        e = np.stack([relL1(arr[:, k], R[:, k]) for k in range(R.shape[1])]).mean(axis=0)
        print(f"[scot NS {tag}] full-traj median={np.median(e)*100:.3f}%", flush=True)
    np.savez_compressed(out, predAR16=ens[:16].astype(np.float32), refAR16=R[:16].astype(np.float32))
else:  # wvrep: anchor sweep x flips, lead<=12
    ds = "wave.Layer"; q0, q1 = 0, 1
    trainer = None; meds_s, meds_e = [], []
    for T0 in (0, 1, 2, 4, 8):
        FT = min(T0+12, 20)
        Eg = {g: [] for g in GROUP}; Rs = []
        for t in range(T0+1, FT+1):
            ts = get_test_set(ds, DATA, initial_time=T0, final_time=t)
            if trainer is None:
                trainer = get_trainer(model_path, 16, ts, workers=2); base = trainer.model
            for g in GROUP:
                trainer.model = Wrap(base, g)
                o = rollout(trainer, ts, ar_steps=1)
                Eg[g].append(denorm(o[0].astype(np.float64), ts, q0, q1))
                if g == (0,0): Rs.append(denorm(o[1].astype(np.float64), ts, q0, q1))
        R = np.stack(Rs, 1); Pg = {g: np.stack(v, 1) for g, v in Eg.items()}
        ens = np.mean([Pg[g] for g in GROUP], axis=0)
        def med(arr):
            e = np.stack([relL1(arr[:, k], R[:, k]) for k in range(R.shape[1])]).mean(axis=0)
            return float(np.median(e))
        ms, me = med(Pg[(0,0)]), med(ens)
        meds_s.append(ms); meds_e.append(me)
        print(f"[A={T0}] single={ms*100:.3f}%  ens={me*100:.3f}%", flush=True)
    print(f"[scot WVREP SWEPT] single={np.mean(meds_s)*100:.3f}% (target 12.97)  ens={np.mean(meds_e)*100:.3f}%", flush=True)
    np.savez(out, single=np.array(meds_s), ens=np.array(meds_e))
print("DUMP_DONE", flush=True)
