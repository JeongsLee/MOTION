"""scOT 2icx4 repaired TTA ens predictions at anchor 0 (t=1..20) — contour panel."""
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
trainer = None; Pg = {g: [] for g in GROUP}; Rs = []
for t in range(1, 21):
    ts = get_test_set("wave.Layer", DATA, initial_time=0, final_time=t)
    if trainer is None:
        trainer = get_trainer(model_path, 16, ts, workers=2); base = trainer.model
    c = ts.constants; mn = float(c["mean"]) if np.isscalar(c["mean"]) or np.asarray(c["mean"]).size==1 else None
    for g in GROUP:
        trainer.model = Wrap(base, g)
        o = rollout(trainer, ts, ar_steps=1)
        arr = o[0].astype(np.float64)[:16, 0:1]
        m0 = np.asarray(c["mean"], np.float64).reshape(-1)[0]; s0 = np.asarray(c["std"], np.float64).reshape(-1)[0]
        Pg[g].append(arr*s0 + m0)
        if g == (0,0): Rs.append(o[1].astype(np.float64)[:16, 0:1]*s0 + m0)
ens = np.mean([np.stack(Pg[g], 1) for g in GROUP], axis=0)
np.savez_compressed(out, pred16=ens.astype(np.float32), ref16=np.stack(Rs,1).astype(np.float32))
print("DUMP_DONE", out, flush=True)
