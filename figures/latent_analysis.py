"""Latent semantics analysis (revision R1-3) on latent_dump.py output.
 1. Maps: input fields vs the first latent principal components vs the learned velocity vel(h) (one sample/family).
 2. Learned velocity vs true velocity: correlation of vel(h) components with v_x, v_y on the latent grid.
 3. Linear probes: ridge h -> physical quantities (v_x, v_y, scalar, vorticity, divergence), R^2 on held-out
    trajectories; trained model vs randomly initialised model.
 4. Intervention: mean change of the predicted v_x when h is shifted along the v_x probe direction.
  python latent_analysis.py LATENT.npz OUTDIR"""
import sys, os, json
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
Z = np.load(sys.argv[1]); OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
fams = json.loads(str(Z["__meta__"]))["fams"]
SCAL = {"shallow_water": (8, "height"), "CE-RP": (3, "density"), "com_ns": (3, "density"),
        "incom_ns": (6, "scalar"), "NS-Sines": (None, None)}

def pool(f, Hb, Wb):
    n, H, W = f.shape[:3]; return f.reshape(n, Hb, H // Hb, Wb, W // Wb, *f.shape[3:]).mean((2, 4))

def ridge_r2(X, Y, tr, te, lam=1e-1):
    Xm, Ym = X[tr].mean(0), Y[tr].mean(0)
    A = X[tr] - Xm; w = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(X.shape[1]), A.T @ (Y[tr] - Ym))
    P = (X[te] - Xm) @ w + Ym
    return 1 - ((Y[te] - P) ** 2).sum(0) / ((Y[te] - Y[te].mean(0)) ** 2).sum(0)

report = {}
for fam in fams:
    h = Z[f"{fam}|h"].astype(np.float32); hr = Z[f"{fam}|h_rand"].astype(np.float32); vel = Z[f"{fam}|vel"]
    x = Z[f"{fam}|x_in"]; n, Hb, Wb, d = h.shape
    xb = pool(x, Hb, Wb)                                         # input on the latent grid (normalized units)
    vx, vy = xb[..., 0], xb[..., 1]
    gx = lambda f: (np.roll(f, -1, 2) - np.roll(f, 1, 2)) / 2; gy = lambda f: (np.roll(f, -1, 1) - np.roll(f, 1, 1)) / 2
    vort, div = gx(vy) - gy(vx), gx(vx) + gy(vy)
    tgt = {"v_x": vx, "v_y": vy, "vorticity": vort, "divergence": div}
    si, sname = SCAL.get(fam, (None, None))
    if si is not None: tgt[sname] = xb[..., si]
    # 2. learned velocity vs true velocity (best matching component, sign-free)
    C = {k: [float(np.corrcoef(vel[..., c].ravel(), tgt[k].ravel())[0, 1]) for c in range(2)] for k in ("v_x", "v_y")}
    # 3. probes (hold out the last 2 trajectories)
    idx = np.arange(n * Hb * Wb).reshape(n, -1); tr = idx[:-2].ravel(); te = idx[-2:].ravel()
    Y = np.stack([tgt[k].reshape(-1) for k in tgt], 1)
    r2 = ridge_r2(h.reshape(-1, d), Y, tr, te); r2r = ridge_r2(hr.reshape(-1, d), Y, tr, te)
    report[fam] = {"vel_corr": C, "probe_r2": dict(zip(tgt, map(float, r2))), "probe_r2_random": dict(zip(tgt, map(float, r2r))),
                   "intervention": dict(zip(map(float, Z[f"{fam}|alphas"]), map(float, Z[f"{fam}|dvx"])))}
    print(f"== {fam}\n  vel(h) vs v: " + ", ".join(f"{k}: c0 {v[0]:+.2f} c1 {v[1]:+.2f}" for k, v in C.items()))
    print("  probe R^2 trained / random: " + ", ".join(f"{k} {a:.2f}/{b:.2f}" for k, a, b in zip(tgt, r2, r2r)))
    print("  intervention Δv_x(alpha): " + ", ".join(f"{a:+.0f}:{v:+.4f}" for a, v in report[fam]["intervention"].items()))
    # 1. maps for sample 0
    U, S_, Vt = np.linalg.svd(h[0].reshape(-1, d) - h[0].reshape(-1, d).mean(0), full_matrices=False)
    pcs = (U[:, :3] * S_[:3]).reshape(Hb, Wb, 3)
    panels = [("input $v_x$", vx[0]), ("input $v_y$", vy[0]), ("vorticity", vort[0])] + \
             ([(sname, tgt[sname][0])] if si is not None else []) + \
             [("latent PC1", pcs[..., 0]), ("latent PC2", pcs[..., 1]), ("latent PC3", pcs[..., 2]),
              ("learned vel$_0$", vel[0, ..., 0]), ("learned vel$_1$", vel[0, ..., 1])]
    fig, axs = plt.subplots(1, len(panels), figsize=(1.45 * len(panels), 1.7))
    for ax, (t, f) in zip(axs, panels):
        ax.imshow(f, cmap="RdBu_r", vmin=-np.abs(f).max(), vmax=np.abs(f).max()); ax.set_title(t, fontsize=6); ax.axis("off")
    fig.suptitle(fam, fontsize=7); fig.tight_layout(); fig.savefig(f"{OUT}/latent_maps_{fam}.png", dpi=170); plt.close(fig)
json.dump(report, open(f"{OUT}/latent_report.json", "w"), indent=1)
