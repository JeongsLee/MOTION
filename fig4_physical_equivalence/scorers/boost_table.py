"""Corrected-sign Galilean boost table for MOTION, from the stored predictions.

Both the observed window and the scored future are boosted with rho(x-Vt), u+V, so the
target is an exact solution of the same equations and the error is read in the boosted
frame directly -- no un-boost, no interpolation.
"""
import glob
import os

import numpy as np

D = os.environ.get("BDIR", "/corpus/scratch/boostfix3d")
FAM = "pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08"
U_RMS = 0.82                      # cells/frame -> multiples of the family r.m.s. velocity


def rel(p, g):
    return 100 * np.linalg.norm(p - g) / (np.linalg.norm(g) + 1e-12)


def vrmse(p, g):
    return np.mean([np.mean([np.sqrt(((p[k, :, c] - g[k, :, c]) ** 2).mean()
                                     / (g[k, :, c].var() + 1e-7))
                             for c in range(p.shape[-1])]) for k in range(p.shape[0])])


print("=== MOTION 20.7M pretrained: Galilean boost, CORRECTED sign ===", flush=True)
print("  rho'(x,t)=rho(x-Vt,t), u'=u(x-Vt,t)+V   (frame t rolled FORWARD by c*t cells)",
      flush=True)
print(f"{'c':>3} {'V/u_rms':>8} {'n':>3} {'rel-L2':>8} {'median':>8} {'VRMSE':>7} "
      f"{'vs c=0':>9}", flush=True)
base = None
for c in (0, 1, 2, 4):
    rl, vr = [], []
    for s in range(8):
        f = glob.glob(f"{D}/fig1_ensv3_fb{c}_s{s}.npz")
        if not f:
            continue
        z = np.load(f[0])
        pk, gk = f"pred_{FAM}_s{s}", f"gt_{FAM}_s{s}"
        if pk not in z.files or gk not in z.files:
            continue
        p, g = z[pk], z[gk]
        ch = [k for k in range(g.shape[-1]) if g[0, :, k].std() > 0][:5]
        rl.append(rel(p[..., ch], g[..., ch]))
        vr.append(vrmse(p[..., ch], g[..., ch]))
    if not rl:
        print(f"{c:>3} {c*U_RMS:>8.2f}   (no files)", flush=True)
        continue
    m = float(np.mean(rl))
    if c == 0:
        base = m
    d = "" if c == 0 else f"{100*(m/base-1):+8.1f}%"
    print(f"{c:>3} {c*U_RMS:>8.2f} {len(rl):>3} {m:>8.2f} {np.median(rl):>8.2f} "
          f"{np.mean(vr):>7.3f} {d:>9}", flush=True)
print("BOOST_TABLE_DONE", flush=True)
