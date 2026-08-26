"""Pull the three slices Fig 4a needs out of the per-transform prediction dump, so the
panel shows real answers rather than one answer copied twice.

perg_id_s0   : the operator's prediction from the benchmark's own copy
perg_r18_s0  : its prediction from the rotated copy, already mapped back to the original
               frame, so the two are directly comparable and their difference is the defect
"""
import glob
import os

import numpy as np

D = os.environ.get("FDIR", "/corpus/scratch/fig4a")
OUT = os.environ.get("OUT", "/code-vol/motion_fv2_phys/fig4a_slices.npz")
FAM = "pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08"

f = glob.glob(f"{D}/fig1_ensv3_fa_s0.npz")[0]
z = np.load(f)
print("keys:", [k for k in z.files][:12], flush=True)

pid, prot = z["perg_id_s0"], z["perg_r18_s0"]
n = round(pid.shape[1] ** (1 / 3))
print("per-transform arrays", pid.shape, "-> box", n, flush=True)
CH = 3                                   # density
K = pid.shape[0] - 1                     # last scored frame


def mid(a):
    v = a[K].reshape(n, n, n, -1)[:, :, n // 2, CH]
    return np.asarray(v, np.float32)


s_id, s_rot = mid(pid), mid(prot)
eps = 100 * float(np.linalg.norm(prot - pid) / (np.linalg.norm(pid) + 1e-12))
print(f"defect over the whole window: {eps:.3f}%", flush=True)
np.savez_compressed(OUT, pred_id=s_id, pred_rot_back=s_rot, eps=np.float32(eps))
print("wrote", OUT, os.path.getsize(OUT) / 1e3, "kB", flush=True)
