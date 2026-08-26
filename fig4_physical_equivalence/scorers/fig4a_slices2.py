"""Fig 4a materials at the VALIDATED readout (QGRID=32, the lattice the closure fix was built
for). Three real fields: the operator's answer from the benchmark's own copy, its answer from a
rotated copy (mapped back), and their difference -- so the panel's question is answered inside
the panel instead of assumed."""
import glob
import os

import numpy as np

D = os.environ.get("FDIR", "/corpus/scratch/fig4a32")
TAG = os.environ.get("TAG", "fa32_s0")
OUT = os.environ.get("OUT", "/code-vol/fig4a_out/fig4a_slices32.npz")
# the input frame the window starts from, for an amplitude-matched display
FAM = "pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08"
CH = 3                                   # density

z = np.load(glob.glob(f"{D}/fig1_ensv3_{TAG}.npz")[0])
print("keys:", [k for k in z.files], flush=True)
pid = z["perg_id_s0"]
n = round(pid.shape[1] ** (1 / 3))
K = int(os.environ.get("FRAME", pid.shape[0] - 1))   # display frame; eps uses the whole window
print(f"per-transform arrays {pid.shape} -> box {n}, scoring frame {K}", flush=True)


def mid(a):
    return np.asarray(a[K].reshape(n, n, n, -1)[:, :, n // 2, CH], np.float32)


out = {"pred_id": mid(pid)}
for t in ("r18", "fx"):
    k = f"perg_{t}_s0"
    if k not in z.files:
        print(f"  {t}: absent", flush=True)
        continue
    p = z[k]
    out[f"pred_{t}_back"] = mid(p)
    eps = 100 * float(np.linalg.norm(p - pid) / (np.linalg.norm(pid) + 1e-12))
    out[f"eps_{t}"] = np.float32(eps)
    print(f"  defect vs identity, {t}: {eps:.2f}%   (paper range 6.2-8.9)", flush=True)

gk = f"gt_{FAM}_s0"
if gk in z.files:
    out["gt"] = mid(z[gk])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez_compressed(OUT, **out)
print("wrote", OUT, round(os.path.getsize(OUT) / 1e3, 1), "kB", flush=True)
print("SLICES32_DONE", flush=True)
