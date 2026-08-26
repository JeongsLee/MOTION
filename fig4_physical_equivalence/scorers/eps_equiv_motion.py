"""Equivariance defect of MOTION, from the per-transform predictions saved by
fig1_ens3d_v3 (SAVE_PERG). CPU only -- the defect compares two predictions on the same
lattice, so no ground truth and no model are needed.

  eps(T) = || T^-1 G(Tu) - G(u) || / || G(u) ||

The stored perg_* arrays are already inverse-transformed and in physical units, so the
defect is a direct norm ratio against perg_id. Boost defects are computed from the
separate BOOST runs by un-boosting their predictions.
"""
import glob
import os

import numpy as np

D = os.environ.get("REFDIR", "/corpus/scratch/refequiv")
TAG = os.environ.get("TAG", "ref_sym")
BTAG = os.environ.get("BTAG", "ref_b")
FAM = "pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08"
DT = 1.0 / 20.0
BOX = 32


def defect(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


print("=== MOTION equivariance defect (pretrained 20.7M, no downstream finetune) ===",
      flush=True)

# ---- symmetry group: perg_* arrays from the SAVE_PERG run -------------------------
tags = os.environ.get("TAGS", "fx,fy,fz,r18,r9,r44,r24").split(",")
acc = {t: [] for t in tags}
for s in range(8):
    f = glob.glob(f"{D}/fig1_ensv3_{TAG}_s{s}.npz")
    if not f:
        continue
    z = np.load(f[0])
    if f"perg_id_s{s}" not in z.files:
        print(f"  [s{s}] per-transform predictions absent", flush=True)
        continue
    ref = z[f"perg_id_s{s}"]
    for t in tags:
        k = f"perg_{t}_s{s}"
        if k in z.files:
            acc[t].append(defect(z[k], ref))
lab = {"s4": "shift 4", "s8": "shift 8", "s16": "shift 16", "fx": "mirror x", "fy": "mirror y", "fz": "mirror z", "r18": "rot 90 z",
       "r9": "rot 90 x", "r44": "rot 90 y", "r24": "cyclic xyz"}
for t in tags:
    v = acc[t]
    if v:
        print(f"  {lab[t]:11s}: eps = {np.mean(v):.4f} (median {np.median(v):.4f}, n={len(v)})",
              flush=True)

# ---- Galilean boosts: un-boost the stored prediction, compare with c=0 ------------
def load_boost(c, s):
    f = glob.glob(f"{D}/fig1_ensv3_{BTAG}{c}_s{s}.npz")
    if not f:
        return None
    z = np.load(f[0])
    k = f"pred_{FAM}_s{s}"
    return z[k] if k in z.files else None


for c in (1, 2, 4):
    ds = []
    for s in range(8):
        p0, pc = load_boost(0, s), load_boost(c, s)
        if p0 is None or pc is None:
            continue
        V = c * (1.0 / 128.0) / DT
        v = pc.reshape((10,) + (BOX,) * 3 + (-1,)).copy()
        for k in range(10):                     # frame 10+k was rolled forward by +c*(10+k)
            v[k] = np.roll(v[k], -((c * (10 + k)) // 4), axis=0)   # native -> box cells
        v = v.reshape(10, BOX ** 3, -1)
        v[..., 0] -= V                          # remove the frame velocity (slot 0 = vx)
        ds.append(defect(v, p0))
    if ds:
        print(f"  boost c={c} ({c*0.82:.2f} u_rms): eps = {np.mean(ds):.4f} "
              f"(median {np.median(ds):.4f}, n={len(ds)})", flush=True)
# ---- accuracy under each transform (the "rotated" column of Fig4d) ---------------
def rel(p_, g_):
    return 100 * np.linalg.norm(p_ - g_) / (np.linalg.norm(g_) + 1e-12)


def vrm(p_, g_):
    return np.mean([np.mean([np.sqrt(((p_[k, :, c] - g_[k, :, c]) ** 2).mean()
                                     / (g_[k, :, c].var() + 1e-7))
                             for c in range(p_.shape[-1])]) for k in range(p_.shape[0])])


print("--- accuracy under each transform (same weights, transformed test copy) ---",
      flush=True)
accs = {t: ([], []) for t in ["id"] + tags}
for s_ in range(8):
    f = glob.glob(f"{D}/fig1_ensv3_{TAG}_s{s_}.npz")
    if not f:
        continue
    z = np.load(f[0])
    gk = f"gt_{FAM}_s{s_}"
    if gk not in z.files:
        continue
    g = z[gk]
    ch = [c for c in range(g.shape[-1]) if g[0, :, c].std() > 0][:5]
    for t in ["id"] + tags:
        k = f"perg_{t}_s{s_}"
        if k in z.files:
            accs[t][0].append(rel(z[k][..., ch], g[..., ch]))
            accs[t][1].append(vrm(z[k][..., ch], g[..., ch]))
for t in ["id"] + tags:
    r_, v_ = accs[t]
    if r_:
        print(f"  {lab.get(t, t):11s}: rel-L2 {np.mean(r_):6.2f}  VRMSE {np.mean(v_):.3f}",
              flush=True)
print("EPS_MOTION_DONE", flush=True)
