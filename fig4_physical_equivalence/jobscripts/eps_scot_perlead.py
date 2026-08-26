"""scOT symmetry-TTA control. argv: model_dir ds T0 FT qoi out [kw]
Same flip group {id,fx,fy,r180}; velocity sign handling for NS (channels [u,v]).
Rollout per g on transformed inputs is NOT possible through get_test_set (data fixed),
so we transform at the MODEL boundary: wrap the model with T_g -> forward -> T_g^{-1}."""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.getcwd())
from scOT.inference import get_trainer, rollout, get_test_set

model_path, ds, T0, FT, qoi, out = (sys.argv[1], sys.argv[2], int(sys.argv[3]),
                                    int(sys.argv[4]), sys.argv[5], sys.argv[6])
KW = json.loads(sys.argv[7]) if len(sys.argv) > 7 else {}
DATA = os.environ["DATA"]; VEL = int(os.environ.get("VEL", "0"))
q0, q1 = [int(x) for x in qoi.split(":")]
GROUP = [(t, fr, fc) for t in (0, 1) for fr in (0, 1) for fc in (0, 1)]

def tg(x, g, inv=False):
    t, fr, fc = g
    def _fl(x):
        if fr:
            x = torch.flip(x, dims=[-2]).clone()
            if VEL: x[:, 0] = -x[:, 0]
        if fc:
            x = torch.flip(x, dims=[-1]).clone()
            if VEL: x[:, 1] = -x[:, 1]
        return x
    def _tr(x):
        if t:
            x = x.transpose(-2, -1).contiguous()
            if VEL: x = x[:, [1, 0] + list(range(2, x.shape[1]))].clone()
        return x
    return _tr(_fl(x)) if inv else _fl(_tr(x))

class Wrap(torch.nn.Module):
    def __init__(self, mdl, g):
        super().__init__(); object.__setattr__(self, "_wrapped", mdl)
        self.m = mdl; self.g = g
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(object.__getattribute__(self, "_wrapped"), name)
    def forward(self, pixel_values=None, time=None, **kw):
        pv = tg(pixel_values, self.g)
        o = self.m(pixel_values=pv, time=time, **kw)
        o.output = tg(o.output, self.g, inv=True)
        return o

def denorm(arr, ts):
    c = ts.constants
    if "mean_solution" in c:
        mn = float(c["mean_solution"]); sd = float(c["std_solution"])
        return arr[:, q0:q1]*sd + mn
    mn = np.asarray(c["mean"], dtype=np.float64).reshape(-1); sd = np.asarray(c["std"], dtype=np.float64).reshape(-1)
    if mn.size == 1: return arr[:, q0:q1]*sd.item() + mn.item()
    return arr[:, q0:q1]*sd[q0:q1, None, None] + mn[q0:q1, None, None]

def relL1(p, r): return np.abs(p-r).sum(axis=(1,2,3))/(np.abs(r).sum(axis=(1,2,3))+1e-12)

ts0 = get_test_set(ds, DATA, initial_time=T0, final_time=min(T0+1, FT), dataset_kwargs=KW) \
    if FT > 1 else get_test_set(ds, DATA, dataset_kwargs=KW)
trainer = get_trainer(model_path, 16, ts0, workers=2)
base_model = trainer.model
E_g = {g: [] for g in GROUP}; P_g = {g: [] for g in GROUP}; R = []
times = range(T0+1, FT+1) if FT > 1 else [1]
for t in times:
    ts = get_test_set(ds, DATA, initial_time=T0, final_time=t, dataset_kwargs=KW) if FT > 1 \
        else get_test_set(ds, DATA, dataset_kwargs=KW)
    for g in GROUP:
        trainer.model = Wrap(base_model, g).to(trainer.model.device if hasattr(trainer.model,'device') else 'cuda' if torch.cuda.is_available() else 'cpu')
        o = rollout(trainer, ts, ar_steps=1)
        p = denorm(o[0].astype(np.float64), ts)
        E_g[g].append(p)
        if g == (0, 0, 0): R.append(denorm(o[1].astype(np.float64), ts))
    if t == times[0] or t == list(times)[-1]:
        print(f"  t={t} done", flush=True)
R = np.stack(R, 1)
per_g = {g: np.stack(v, 1) for g, v in E_g.items()}
# defect in scOT's own metric family: rel-L1 per (sample,time), time-mean, sample-median
_lb = {(0,1,0):"fr",(0,0,1):"fc",(0,1,1):"r180",(1,0,0):"d1",(1,1,0):"r90",(1,0,1):"r270",(1,1,1):"d2"}
def _eps_of(Pg, P0):
    num = np.abs(Pg - P0).sum(axis=(2, 3)); den = np.abs(P0).sum(axis=(2, 3)) + 1e-12
    return float(np.median((num / den).mean(axis=1)) * 100)
_P0 = per_g[(0, 0, 0)]
_eps = {_lb[g]: _eps_of(per_g[g], _P0) for g in GROUP if g != (0, 0, 0)}
print(f"[scot {ds}] EPS_EQUIV(%, task metric):",
      {k: round(v, 3) for k, v in sorted(_eps.items(), key=lambda kv: kv[1])}, flush=True)
ens4 = np.mean([per_g[g] for g in GROUP if g[0] == 0], axis=0)
ens8 = np.mean([per_g[g] for g in GROUP], axis=0)
# per-lead error of the identity pass, and the persistence baseline in the SAME metric:
# a family whose field changes by O(100%) over the horizon cannot be predicted to a few per
# cent at the long leads, so a flat curve would mean the task is not the long-lead one.
_P0 = per_g[(0, 0, 0)]
_perlead = [float(np.median(relL1(_P0[:, k], R[:, k])) * 100) if False else
            float(np.median(np.abs(_P0[:, k] - R[:, k]).sum(axis=(1, 2)) /
                            (np.abs(R[:, k]).sum(axis=(1, 2)) + 1e-12)) * 100)
            for k in range(R.shape[1])]
_pers = [float(np.median(np.abs(R[:, 0] - R[:, k]).sum(axis=(1, 2)) /
                         (np.abs(R[:, k]).sum(axis=(1, 2)) + 1e-12)) * 100)
         for k in range(R.shape[1])]
print("[scot {}] per-lead identity error (%):".format(ds),
      [round(v, 2) for v in _perlead], flush=True)
print("[scot {}] per-lead PERSISTENCE  (%):".format(ds),
      [round(v, 2) for v in _pers], flush=True)
for g in GROUP:
    e = np.stack([relL1(per_g[g][:, k], R[:, k]) for k in range(R.shape[1])]).mean(axis=0)
    print(f"[scot {ds} g={g}] full-traj median={np.median(e)*100:.3f}%", flush=True)
for arr, tag in [(per_g[(0,0,0)], "single"), (ens4, "K4-ENSEMBLE"), (ens8, "K8-ENSEMBLE")]:
    e = np.stack([relL1(arr[:, k], R[:, k]) for k in range(R.shape[1])]).mean(axis=0)
    print(f"[scot {ds} {tag}] full-traj median={np.median(e)*100:.3f}%", flush=True)
np.savez_compressed(out, pred16=ens8[:16].astype(np.float32), ref16=R[:16].astype(np.float32))
print("DUMP_DONE", out, flush=True)
