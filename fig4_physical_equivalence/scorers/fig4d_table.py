"""Fig4d accuracy anchor, computed on CPU from stored predictions.

Three rows, one metric pair, one grid, one window (frames 10-19, 32^3 sites, physical
units): the released 1.3B specialist; the pretrained 20.7M generalist with only the
subsample ensemble (our baseline); and the same weights with the rotation group on top.
The lattice alignment is fixed on our side (predictions were read at the data sample
sites), so the three rows are directly comparable.

  python fig4d_table.py     env: REF, MOT_BASE, MOT_ROT
"""
import glob
import os

import numpy as np

REF = os.environ.get("REF", "/corpus/results/walrus_ref_cur")
BASE = os.environ.get("MOT_BASE", "/corpus/scratch/refequiv")     # ref_b0_s* = K8 only
ROT = os.environ.get("MOT_ROT", "/corpus/scratch/rot24acc")       # rot24_s*  = K8 x rot24
FAM = "pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08"
AX = np.arange(0, 128, 4)[:32]


def vrmse(p, g):
    return np.mean([np.mean([np.sqrt(((p[k, :, c] - g[k, :, c]) ** 2).mean()
                                     / (g[k, :, c].var() + 1e-7))
                             for c in range(p.shape[-1])]) for k in range(p.shape[0])])


def rel(p, g):                       # paper standard: whole-window joint norm ratio
    return 100 * np.linalg.norm(p - g) / (np.linalg.norm(g) + 1e-12)


def motion(dirpath, tag):
    rl, vr = [], []
    for s in range(8):
        f = sorted(glob.glob(f"{dirpath}/fig1_ensv3_{tag}_s{s}.npz"))
        if not f:
            continue
        z = np.load(f[0])
        p, g = z[f"pred_{FAM}_s{s}"], z[f"gt_{FAM}_s{s}"]
        ch = [c for c in range(g.shape[-1]) if g[0, :, c].std() > 0][:5]
        rl.append(rel(p[..., ch], g[..., ch]))
        vr.append(vrmse(p[..., ch], g[..., ch]))
    return rl, vr


def walrus():
    z = np.load(f"{REF}/walrus_tta_preds.npz")
    chm = list(np.asarray(z["chmap"]))
    rl, vr = [], []
    for key_p, key_g in (("pred_id", "gt"),):
        P, G = np.asarray(z[key_p], np.float32), np.asarray(z[key_g], np.float32)
        for n in range(P.shape[0]):
            p = P[n][:, AX][:, :, AX][:, :, :, AX]
            g = G[n][:, AX][:, :, AX][:, :, :, AX]
            p = p.reshape(p.shape[0], -1, p.shape[-1])[..., chm]
            g = g.reshape(g.shape[0], -1, g.shape[-1])[..., chm]
            rl.append(rel(p, g)); vr.append(vrmse(p, g))
    return rl, vr


rows = [("Walrus-FT 1.3B (family finetune)", *walrus()),
        ("MOTION 20.7M pretrained, K8", *motion(BASE, "ref_b0")),
        ("MOTION 20.7M pretrained, K8 x rot24", *motion(ROT, "rot24"))]
print("=== Fig4d anchor: same window, same 32^3 sites, physical units ===", flush=True)
for name, rl, vr in rows:
    if not rl:
        print(f"  {name:38s}  (no predictions on disk yet)", flush=True)
        continue
    print(f"  {name:38s}  rel-L2 {np.mean(rl):6.2f} / {np.median(rl):6.2f}   "
          f"VRMSE {np.mean(vr):.3f} / {np.median(vr):.3f}   (n={len(rl)})", flush=True)
if rows[1][1] and rows[2][1]:
    a, b = np.mean(rows[1][1]), np.mean(rows[2][1])
    av, bv = np.mean(rows[1][2]), np.mean(rows[2][2])
    print(f"  rotation-group gain: rel {100*(b/a-1):+.1f}%   VRMSE {100*(bv/av-1):+.1f}%",
          flush=True)
print("FIG4D_DONE", flush=True)
