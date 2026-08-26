"""Ensemble gain as a function of the averaged subgroup, for the released 1.3B model.

The per-rotation predictions are already stored inverse-transformed, so a subgroup average
is just a mean over the corresponding keys. Three nested subsets are scored: the identity
alone, the fourfold rotations about the vertical axis (the only subgroup this model closed),
and all twenty-four proper rotations. If the gain is a monotone function of subgroup size
the choice of deployed group is a protocol detail; if it changes sign, the choice is doing
the work and has to be declared in advance.
"""
import os

import numpy as np

REF = os.environ.get("REF", "/corpus/results/walrus_ref_cur")
AX = np.arange(0, 128, 4)[:32]
ROT24 = ["r0", "r3", "r5", "r6", "r9", "r10", "r12", "r15", "r17", "r18", "r20", "r23",
         "r24", "r27", "r29", "r30", "r32", "r35", "r37", "r38", "r41", "r42", "r44",
         "r47"]
C4Z = ["r0", "r6", "r18", "r20"]


def sub(a):
    v = a[:, AX][:, :, AX][:, :, :, AX]
    return v.reshape(v.shape[0], -1, v.shape[-1])


def rel(p, g):
    return 100 * np.linalg.norm(p - g) / (np.linalg.norm(g) + 1e-12)


z = np.load(f"{REF}/walrus_rot_preds.npz")
have = [k[5:] for k in z.files if k.startswith("pred_")]
print(f"stored per-rotation predictions: {len(have)}", flush=True)
chm = list(np.asarray(z["chmap"])) if "chmap" in z.files else None
G = np.asarray(z["gt"], np.float32) if "gt" in z.files else None
if G is None:
    G = np.asarray(np.load(f"{REF}/walrus_tta_preds.npz")["gt"], np.float32)

NS = G.shape[0]
gts = [sub(G[n])[..., chm] if chm else sub(G[n]) for n in range(NS)]

SETS = {"identity only": ["r0"] if "r0" in have else ["id"],
        "C4 about z (4)": [k for k in C4Z if k in have],
        "all proper rotations (24)": [k for k in ROT24 if k in have]}
base = None
for name, keys in SETS.items():
    if not keys:
        print(f"  {name:26s}: keys absent", flush=True)
        continue
    errs = []
    for n in range(NS):
        acc = None
        for k in keys:
            a = sub(np.asarray(z[f"pred_{k}"][n], np.float32))
            a = a[..., chm] if chm else a
            acc = a if acc is None else acc + a
        errs.append(rel(acc / len(keys), gts[n]))
    m = float(np.mean(errs))
    if base is None:
        base = m
    print(f"  {name:26s}: n={len(keys):2d}  rel-L2 {m:6.2f}  "
          f"gain {100*(m/base-1):+6.1f}%", flush=True)
print("GROUP_CURVE_DONE", flush=True)
