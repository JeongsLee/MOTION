"""Equivariance defect of the released Walrus checkpoint, from predictions already on
disk -- no GPU, no model.

  eps(T) = || T^-1 G(Tu) - G(u) || / || G(u) ||

is ground-truth free: it compares the operator's answers to two physically identical
descriptions of the same state, so it is unaffected by any offset between the scoring
lattice and the data lattice (both sides live on the same points).

Sources: walrus_rot_preds.npz stores the inverse-transformed per-rotation predictions
(SAVE_PERG); walrus_gal_c*_preds.npz store the boosted-frame predictions, which are
un-boosted here (roll back by c*k cells per frame, subtract V from the boost axis).
Arrays are (8, 10, 128, 128, 128, 5); they are subsampled to 32^3 immediately so the
whole audit fits in a small CPU box.
"""
import os

import numpy as np

REF = os.environ.get("REF", "/corpus/results/walrus_ref_cur")
AX = np.arange(0, 128, 4)[:32]
DT = 1.0 / 20.0


def sub(a):                      # (8,10,128^3,5) -> (8,10,32^3,5), one sample at a time
    out = np.empty((a.shape[0], a.shape[1], 32 ** 3, a.shape[-1]), np.float32)
    for n in range(a.shape[0]):
        v = a[n][:, AX][:, :, AX][:, :, :, AX]
        out[n] = v.reshape(v.shape[0], -1, v.shape[-1])
    return out


def load_sub(path, key):
    z = np.load(path, mmap_mode=None)
    if key not in z.files:
        return None
    arr = np.asarray(z[key], np.float32)
    s = sub(arr)
    del arr
    return s


def eps(a, b):                   # defect of a against the reference prediction b
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def per_sample_eps(a, b):
    return [float(np.linalg.norm(a[n] - b[n]) / (np.linalg.norm(b[n]) + 1e-12))
            for n in range(a.shape[0])]


print("=== Walrus equivariance defect (prediction vs prediction, no ground truth) ===",
      flush=True)

# ---- rotations (per-transform predictions were saved inverse-transformed) ---------
rot = f"{REF}/walrus_{os.environ.get('PERG_TAG', 'rot')}_preds.npz"
if os.path.exists(rot):
    ref = load_sub(rot, "pred_id")
    names = {k: k for k in os.environ.get("PERG_KEYS",
             "r18,r9,r44,r24").split(",")}
    for k, lab in names.items():
        a = load_sub(rot, f"pred_{k}")
        if a is None:
            print(f"  [{k}] not stored", flush=True)
            continue
        e = per_sample_eps(a, ref)
        print(f"  rotation {k:4s} ({lab:16s}): eps = {np.mean(e):.4f} "
              f"(median {np.median(e):.4f})", flush=True)
        del a
    del ref
else:
    print("  rotation file missing", flush=True)

# ---- Galilean boosts (stored in the boosted frame; un-boost before comparing) -----
base = load_sub(f"{REF}/walrus_tta_preds.npz", "pred_id")
for c in (1, 2, 4):
    p = f"{REF}/walrus_gal_c{c}_preds.npz"
    if not os.path.exists(p):
        print(f"  boost c={c}: file missing", flush=True)
        continue
    z = np.load(p)
    arr = np.asarray(z["pred_id"], np.float32)             # (8,10,128^3,5) boosted frame
    V = c * (1.0 / 128.0) / DT
    for n in range(arr.shape[0]):        # un-boost: forward rolled +c*t, so invert with -c*t
        for k in range(arr.shape[1]):
            arr[n, k] = np.roll(arr[n, k], -c * (10 + k), axis=0)
    s = sub(arr)
    del arr
    # the velocity channel of the boosted solution carries +V; remove it before comparing
    # (channel order is the walrus output order; the boost axis is x = channel index 2
    #  under the raw->out map recorded by the scorer, so subtract there)
    ch = int(os.environ.get("BOOST_CH", "2"))
    s[..., ch] -= V
    e = per_sample_eps(s, base)
    print(f"  boost c={c} (V={V:.4f}, {c*0.82:.2f} u_rms): eps = {np.mean(e):.4f} "
          f"(median {np.median(e):.4f})", flush=True)
    del s
print("EPS_WALRUS_DONE", flush=True)
