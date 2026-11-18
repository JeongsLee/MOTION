"""Supplementary figure: specificity of the spatially varying advection commands (NS-Sines, held-out test trajectory 5,
MOTION-S). Velocity responses (commanded - base forecast) of the shear and vortex commands vs the NS-solver responses
(solver with the command - solver without), shown as vorticity of the response; r on the velocity response.
Data: motion_local/out_ctl3f (shear, 1 std) and out_ctl3v20 (vortex, R 20, 1 std; |vel / |dyn are stored at amplitude 1 only), keys |vel (model) and |dyn (solver)."""
import os
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
L = os.environ.get("MOTION_LOCAL", "motion_local"); OUT = "figures/out/figS_cmd_specificity"
a = np.load(f"{L}/out_ctl3f/control3f_m1f_NS-Sines.npz", allow_pickle=True); b = np.load(f"{L}/out_ctl3v20/control3v_m1f_NS-Sines.npz", allow_pickle=True)
M = {"shear": a["NS-Sines|shear|vel"][0], "vortex": b["NS-Sines|curl|vel"][0]}          # model response, steps (1, 10)
S = {"shear": a["NS-Sines|shear|dyn"], "vortex": b["NS-Sines|curl|dyn"]}                 # solver response, steps (1, 10)
r = lambda x, y: float(np.corrcoef(x.ravel(), y.ravel())[0, 1])
w = lambda F: (np.roll(F[..., 1], -1, 0) - np.roll(F[..., 1], 1, 0)) / 2 - (np.roll(F[..., 0], -1, 1) - np.roll(F[..., 0], 1, 1)) / 2
T = {}
for t, st in ((0, 1), (1, 10)):
    for c in ("shear", "vortex"):
        o = "vortex" if c == "shear" else "shear"
        T[f"{c}|{st}"] = dict(own=r(M[c][t], S[c][t]), other=r(M[c][t], S[o][t]), solver_solver=r(S["shear"][t], S["vortex"][t]))
plt.rcParams.update({"font.family": "sans-serif", "font.size": 7})
fig, ax = plt.subplots(2, 3, figsize=(180 / 25.4, 122 / 25.4))
t = 1
for i, c in enumerate(("shear", "vortex")):
    o = "vortex" if c == "shear" else "shear"
    for j, (F, ttl) in enumerate(((M[c][t], f"commanded response ({c})"), (S[c][t], f"solver response ({c})"), (S[o][t], f"solver response ({o})"))):
        W = w(F); vm = np.percentile(np.abs(w(S[c][t])), 99)
        a_ = ax[i, j]; a_.imshow(W.T, origin="lower", cmap="PuOr_r", vmin=-vm, vmax=vm, interpolation="bilinear")
        a_.set_xticks([]); a_.set_yticks([]); a_.set_title(ttl, fontsize=7)
        if j > 0: a_.text(0.97, 0.03, f"r {r(M[c][t], F):.2f}", transform=a_.transAxes, ha="right", fontsize=6.5, bbox=dict(fc="w", ec="none", alpha=0.8, pad=0.5))
    ax[i, 0].set_ylabel(f"{c} command", fontsize=7)
plt.tight_layout(); plt.savefig(OUT + ".pdf"); plt.savefig(OUT + ".png", dpi=200)
json.dump(T, open(OUT + ".json", "w"), indent=1); print(json.dumps(T, indent=1))
