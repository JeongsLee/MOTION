"""Walrus boost table from the stored corrected-sign predictions, on both conventions:
the dense 128^3 grid the released model runs on, and the 32^3 subsample MOTION is scored
at, so the two models can be put in one column."""
import os

import numpy as np

REF = os.environ.get("REF", "/corpus/results/walrus_ref_cur")
AX = np.arange(0, 128, 4)[:32]


def rel(p, g):
    return 100 * np.linalg.norm(p - g) / (np.linalg.norm(g) + 1e-12)


def vrmse(p, g):
    return np.mean([np.mean([np.sqrt(((p[k, :, c] - g[k, :, c]) ** 2).mean()
                                     / (g[k, :, c].var() + 1e-7))
                             for c in range(p.shape[-1])]) for k in range(p.shape[0])])


def score(tag, sub):
    p = f"{REF}/walrus_{tag}_preds.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p)
    chm = list(np.asarray(z["chmap"]))
    P, G = z["pred_id"], z["gt"]
    rl, vr = [], []
    for n in range(P.shape[0]):
        a, b = np.asarray(P[n], np.float32), np.asarray(G[n], np.float32)
        if sub:
            a = a[:, AX][:, :, AX][:, :, :, AX]
            b = b[:, AX][:, :, AX][:, :, :, AX]
        a = a.reshape(a.shape[0], -1, a.shape[-1])[..., chm]
        b = b.reshape(b.shape[0], -1, b.shape[-1])[..., chm]
        rl.append(rel(a, b)); vr.append(vrmse(a, b))
    return float(np.mean(rl)), float(np.median(rl)), float(np.mean(vr))


for sub in (False, True):
    print(f"=== Walrus-FT 1.3B, corrected-sign boost "
          f"({'32^3 subsample (MOTION convention)' if sub else 'dense 128^3'}) ===",
          flush=True)
    print(f"{'c':>3} {'V/u_rms':>8} {'rel-L2':>8} {'median':>8} {'VRMSE':>7} {'vs c=0':>9}",
          flush=True)
    base = None
    for c in (0, 1, 2, 4):
        r = score(f"galfix_c{c}", sub)
        if r is None:
            print(f"{c:>3}   (missing)", flush=True)
            continue
        m, md, v = r
        if c == 0:
            base = m
        d = "" if c == 0 else f"{100*(m/base-1):+8.1f}%"
        print(f"{c:>3} {c*0.82:>8.2f} {m:>8.2f} {md:>8.2f} {v:>7.3f} {d:>9}", flush=True)
print("WALRUS_BOOST_TABLE_DONE", flush=True)
