"""Supplementary figure + table: persistent shear / vortex commands over 30 autoregressive steps (beyond the 10-step training
horizon), NS-Sines, MOTION-S. Data: motion_local/out_ctl3h5 (latent_control3.py --K 30 --horizon_all 1; 3 std; 4 held-out
test trajectories; fields of the first one every 5 steps)."""
import os
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
L = os.environ.get("MOTION_LOCAL", "motion_local") + "/out_ctl3h6"   # chain30: + flow-map edit of the base for every trajectory
OUT = "figures/out/figS_long_horizon"; OUT = "figures/out/figS_long_horizon"
w = lambda F: (np.roll(F[..., 1], -1, 0) - np.roll(F[..., 1], 1, 0)) / 2 - (np.roll(F[..., 0], -1, 1) - np.roll(F[..., 0], 1, 1)) / 2
plt.rcParams.update({"font.family": "sans-serif", "font.size": 7})
CM = [("shear", "control3h_shear_m1f_NS-Sines.npz", "shear"), ("curl", "control3h_curl_m1f_NS-Sines.npz", "vortex")]
fig = plt.figure(figsize=(180 / 25.4, 140 / 25.4)); G0 = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.15], hspace=0.30)
gT = G0[0].subgridspec(1, 2, wspace=0.12); gM = G0[1].subgridspec(3, 9, width_ratios=[1, 1, 1, 1, 0.15, 1, 1, 1, 1], hspace=0.06, wspace=0.05)
T = {}
for b, (kind, f, lab) in enumerate(CM):
    z = np.load(f"{L}/{f}", allow_pickle=True); C = json.loads(str(z["__meta__"]))["report"]["NS-Sines"]["conds"][f"{kind}3"]
    H = [x["horizon"] for x in C if x and "horizon" in x]; ks = np.arange(1, 31)
    g = lambda q: np.array([np.mean([h[k - 1][q] for h in H]) for k in ks])
    T[lab] = {q: g(q).round(3).tolist() for q in ("r_cmd_S1", "r_fm_S1", "r_base_S1", "r_base_S0", "r_S0_S1", "change_model", "change_solver", "r_resp", "r_fm_resp")}
    ax = fig.add_subplot(gT[b])
    lo = lambda q: np.array([np.min([h[k - 1][q] for h in H]) for k in ks]); hi = lambda q: np.array([np.max([h[k - 1][q] for h in H]) for k in ks])
    ax.fill_between(ks, lo("r_cmd_S1"), hi("r_cmd_S1"), color="#c0392b", alpha=0.15, lw=0); ax.fill_between(ks, lo("r_fm_S1"), hi("r_fm_S1"), color="#2471a3", alpha=0.15, lw=0)
    ax.plot(ks, g("r_cmd_S1"), "-", color="#c0392b", lw=1.2, label="latent command vs solver with command")
    ax.plot(ks, g("r_fm_S1"), "-", color="#2471a3", lw=1.2, label="flow-map edit of the base vs solver with command")
    ax.plot(ks, g("r_base_S1"), "--", color="0.45", lw=1.0, label="base vs solver with command")
    ax.axvline(10, color="0.75", lw=0.6, ls=":"); ax.set_ylim(0, 1.02); ax.set_xlim(1, 30); ax.set_xlabel("step"); ax.set_title(lab, fontsize=8)
    if b == 0: ax.set_ylabel("pattern correlation (velocity)"); ax.legend(frameon=False, fontsize=6, loc="lower left")
    else: ax.set_yticklabels([])
    for i, st in enumerate((10, 20, 30)):
        Bm, P, S0, S1, FM = z[f"NS-Sines|{kind}3|hfields{st}"]; vm = np.percentile(np.abs(w(S1)), 99)
        for c, (F, t) in enumerate(((Bm, "base"), (FM, "flow-map edit"), (P, "latent command"), (S1, "solver+cmd"))):
            a = fig.add_subplot(gM[i, 5 * b + c]); a.imshow(w(F).T, origin="lower", cmap="RdBu_r", vmin=-vm, vmax=vm, interpolation="bilinear")
            a.set_xticks([]); a.set_yticks([])
            if i == 0: a.set_title(t, fontsize=6)
            if c == 0 and b == 0: a.set_ylabel(f"step {st}", fontsize=7)
            if c in (0, 1, 2): a.text(0.97, 0.03, f"{np.corrcoef(F.ravel(), S1.ravel())[0, 1]:.2f}", transform=a.transAxes, ha="right", fontsize=5.5, bbox=dict(fc="w", ec="none", alpha=0.8, pad=0.3))
plt.savefig(OUT + ".pdf", bbox_inches="tight"); plt.savefig(OUT + ".png", dpi=200, bbox_inches="tight")
json.dump(T, open(OUT + ".json", "w")); print({k: {q: [v[q][s - 1] for s in (5, 10, 15, 20, 25, 30)] for q in v} for k, v in T.items()})
