"""Poisson scOT symmetry-TTA, split-aligned (last-128 = scOT test idx 112..239)."""
import sys, os, numpy as np, torch
sys.path.insert(0, os.getcwd())
from scOT.inference import get_trainer, rollout, get_test_set
model_path, out = sys.argv[1], sys.argv[2]
DATA = os.environ["DATA"]
GROUP = [(0,0),(1,0),(0,1),(1,1)]
def tg(x, g):
    fr, fc = g
    if fr: x = torch.flip(x, dims=[-2]).clone()
    if fc: x = torch.flip(x, dims=[-1]).clone()
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
ts = get_test_set("elliptic.poisson.Gaussians", DATA)
trainer = get_trainer(model_path, 16, ts, workers=2)
base = trainer.model
c = ts.constants
mn = float(c["mean_solution"]) if "mean_solution" in c else np.asarray(c["mean"]).reshape(-1)[0]
sd = float(c["std_solution"]) if "mean_solution" in c else np.asarray(c["std"]).reshape(-1)[0]
P = {}
for g in GROUP:
    trainer.model = Wrap(base, g)
    o = rollout(trainer, ts, ar_steps=1)
    P[g] = o[0].astype(np.float64)[:, 0:1]*sd + mn
    if g == (0,0): R = o[1].astype(np.float64)[:, 0:1]*sd + mn
ens = np.mean([P[g] for g in GROUP], axis=0)
def med(p, sl): 
    e = np.abs(p[sl]-R[sl]).sum(axis=(1,2,3))/(np.abs(R[sl]).sum(axis=(1,2,3))+1e-12)
    return np.median(e)*100
al = slice(112, 240)
print(f"[pois-scot] single aligned128={med(P[(0,0)], al):.3f}%  ens aligned128={med(ens, al):.3f}%", flush=True)
print(f"[pois-scot] single all240={med(P[(0,0)], slice(None)):.3f}%  ens all240={med(ens, slice(None)):.3f}%", flush=True)
np.savez_compressed(out, pred16=ens[112:128, None].astype(np.float32), ref16=R[112:128, None].astype(np.float32))
print("DUMP_DONE", out, flush=True)
