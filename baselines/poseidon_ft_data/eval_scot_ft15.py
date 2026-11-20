"""REVISION-R1 (user 2026-10-23): score a fine-tuned Poseidon (scOT) model on an FT15 task with the SAME metric as
MOTION's score_ivp_l1.py: relative L1 in PHYSICAL units over the scored channels and all grid points, per trajectory
and lead; full-trajectory = mean over leads 1..KF (or the AR hop leads), final = last lead; median over trajectories.

Physical units: scOT normalizes labels per channel (label = (x - m) / s). Rather than reading the per-class constants
(named differently in every class), each scored raw channel is matched to the label channel it is an affine image of,
and (s, m) are fitted by least squares on the evaluated rows (R^2 is printed; it must be ~1).
Modes: direct (IC frame 0 -> lead t in one call, t = 1..KF) and ar (homogeneous steps of STEP frames, Poseidon's own
autoregressive rollout). Steady tasks: one prediction.
Env: TASK, MODEL, DS, DATA, KF, STEP, SPLIT (val|test), MODE (direct|ar), NEV (0 = all), OUT (json path).
"""
import os, json, numpy as np, h5py, torch
from scOT.inference import get_trainer, rollout
from scOT.problems.base import get_dataset

T = os.environ["TASK"]; MODEL = os.environ["MODEL"]; DS = os.environ["DS"]; DATA = os.environ["DATA"]
KF = int(os.environ.get("KF", "20")); STEP = int(os.environ.get("STEP", "1")); SPLIT = os.environ.get("SPLIT", "test")
MODE = os.environ.get("MODE", "direct"); NEV = int(os.environ.get("NEV", "0")); BS = int(os.environ.get("BS", "16"))
FB = "NS-PwC" if T == "NS-Tracer-PwC" else T                      # NS-Tracer-PwC reads the NS-PwC file
meta = json.load(open(f"{DATA}/{FB}.json")); NTR, NV, NT = meta["ntr"], meta["n_val"], meta["n_test"]
row0 = NTR if SPLIT == "val" else NTR + NV; nsp = NV if SPLIT == "val" else NT
n = min(nsp, NEV) if NEV else nsp
steady = T in ("Poisson-Gauss", "Helmholtz", "SE-AF")
# REVISION-R1 10-24: PAPER=1 -> Poseidon-paper metric (Herde et al. 2024, SM D.1, Table 6): relative L1 at the paper's
# final lead only, per function of interest (a vector field is one function), mean over functions; FUNCS index the
# channels of raw() below.
PAPER = os.environ.get("PAPER", "0") == "1"
PAPER_LEAD = {"GCE-RT": 7}.get(T, 14)
# REVISION-R1 10-26: PAPER_NORM=1 -> errors in scOT's normalized label space (the space of scOT's own
# relative_lp_error on label_ids, i.e. the official metric), no geometry mask; default = physical units
PAPER_NORM = os.environ.get("PAPER_NORM", "0") == "1"
FUNCS = {"NS-Tracer-PwC": [[0, 1], [2]], "CE-RPUI": [[0], [1, 2], [3]], "CE-RM": [[0], [1, 2], [3]],
         "GCE-RT": [[0], [1, 2], [3], [4]]}.get(T, [[0, 1]] if (T.startswith("NS-") or T == "FNS-KF") else [[0]])
# raw physical source of the scored quantities: (file, reader(row, t) -> (C, x, y))
FILE = f"{DATA}/{FB}.h5" if T == "Helmholtz" else f"{DATA}/{FB}.nc"
f = h5py.File(FILE, "r")
if T.startswith("NS-") or T == "FNS-KF":
    var = "velocity" if T.startswith("NS-") else "solution"; ch = [0, 1, 2] if T == "NS-Tracer-PwC" else [0, 1]
    raw = lambda r, t: f[var][r, t, ch]
elif T.startswith("CE-") or T == "GCE-RT":
    var = max((k for k in f if hasattr(f[k], "ndim")), key=lambda k: f[k].ndim)           # rho, u, v, p (+ g for GCE-RT)
    # REVISION-R1 10-26: the GCE-RT phi (Table 6's 5th function) is file channel 5, not 4
    _cc = [0, 1, 2, 3, 5] if (PAPER and T == "GCE-RT") else [0, 1, 2, 3]
    raw = lambda r, t: f[var][r, t, _cc]
elif T in ("Wave-Gauss", "Wave-Layer", "ACE"):
    raw = lambda r, t: f["solution"][r, t][None]
elif T == "Poisson-Gauss":
    raw = lambda r, t: f["solution"][r][None]
elif T == "Helmholtz":
    raw = lambda r, t: f[f"Sample_{r}"]["u"][()][None]
elif T == "SE-AF":
    raw = lambda r, t: f["solution"][r, 1][None]
leads = [1] if steady else (list(range(1, KF + 1)) if MODE == "direct" else list(range(STEP, KF + 1, STEP)))
if PAPER:
    leads = [1] if steady else [PAPER_LEAD]


def predict(t):
    kw = {} if steady else {"fix_input_to_time_step": 0, "time_step_size": t, "max_num_time_steps": 1}
    if os.environ.get("JUSTVEL") == "1":                     # velocity-only models (old *_jv runs)
        kw["just_velocities"] = True
    ds = get_dataset(dataset=DS, which=SPLIT, num_trajectories=NTR, data_path=DATA, move_to_local_scratch=None, **kw)
    if NEV:
        ds = torch.utils.data.Subset(ds, list(range(n)))
    tr = get_trainer(MODEL, BS, ds if not NEV else ds.dataset, workers=4)
    ar = 1 if (steady or MODE == "direct") else t // STEP
    out, lab, _ = rollout(tr, ds, ar_steps=ar)
    return np.asarray(out)[:n], np.asarray(lab)[:n]


ORI = (lambda a: a, lambda a: np.swapaxes(a, -1, -2))
err = np.zeros((n, len(leads))); fitlog = {}
# REVISION-R1 10-27 (downstream prediction panels): DUMP=<npz> -> physical predictions and ground truth of the first DUMP_N trajectories at
# every scored lead (same orientation as the raw file), with the per-trajectory errors
DUMP = os.environ.get("DUMP", ""); DN = min(n, int(os.environ.get("DUMP_N", "16"))); DP, DY = [], []
for j, t in enumerate(leads):
    out, lab = predict(t)
    y = np.stack([raw(row0 + k, t) for k in range(n)]).astype(np.float64)          # (n, C, x, y) physical
    # SE-AF: the airfoil interior (input mask == 1) is excluded from the metric for both models (scOT overwrites its
    # labels with 1 there, so the raw density is not its target); MOTION's scorer does the same (SCORE_EXCLUDE_GEOM=1)
    W = (np.stack([f["solution"][row0 + k, 0] for k in range(n)]) != 1).astype(np.float64)[:, None] if T == "SE-AF" else 1.0
    num = np.zeros(n); den = np.zeros(n); numc = np.zeros((n, y.shape[1])); denc = np.zeros((n, y.shape[1]))
    pc = np.zeros((DN,) + y.shape[1:], np.float32)
    for c in range(y.shape[1]):
        sel = (np.broadcast_to(W, y.shape)[:, c].ravel() > 0)
        yc = y[:, c].ravel()[sel]
        # scOT may transpose the spatial axes (incompressible classes, self.transpose): try both orientations
        # constant label channels (NS dummy rho / p) have a float32 std of ~1e-8, not 0, and give a NaN correlation,
        # which Python's max() can keep: require a real spread and map NaN to 0
        cand = [(float(np.nan_to_num(abs(np.corrcoef(o(lab[:, l]).ravel()[sel], yc)[0, 1]))) if lab[:, l].std() > 1e-6
                 else 0.0, l, oi) for l in range(lab.shape[1]) for oi, o in enumerate(ORI)]
        _, l, oi = max(cand); A = np.stack([ORI[oi](lab[:, l]).ravel()[sel], np.ones_like(yc)], 1)
        (s, m), res, *_ = np.linalg.lstsq(A, yc, rcond=None)
        r2 = 1 - float(np.sum((A @ [s, m] - yc) ** 2)) / float(np.sum((yc - yc.mean()) ** 2) + 1e-30)
        fitlog[f"t{t}_c{c}"] = {"label_ch": l, "transposed": bool(oi), "s": float(s), "m": float(m), "R2": r2}
        p = s * ORI[oi](out[:, l]) + m
        pc[:, c] = p[:DN]
        w = np.broadcast_to(W, y.shape)[:, c]
        if PAPER_NORM:                                       # normalized label space, unmasked
            numc[:, c] = np.abs(out[:, l] - lab[:, l]).reshape(n, -1).sum(1); denc[:, c] = np.abs(lab[:, l]).reshape(n, -1).sum(1)
        else:
            numc[:, c] = (w * np.abs(p - y[:, c])).reshape(n, -1).sum(1); denc[:, c] = (w * np.abs(y[:, c])).reshape(n, -1).sum(1)
        num += numc[:, c]; den += denc[:, c]
    DP.append(pc); DY.append(y[:DN].astype(np.float32))
    err[:, j] = (num / (den + 1e-12) if not PAPER else
                 np.mean([numc[:, g].sum(1) / (denc[:, g].sum(1) + 1e-12) for g in FUNCS], 0))
    print(f"[{T} {SPLIT} {MODE}] lead {t}: median {100 * np.median(err[:, j]):.3f}%  fit R2 min "
          f"{min(v['R2'] for k, v in fitlog.items() if k.startswith(f't{t}_')):.6f}", flush=True)
full = err.mean(1); res = {"task": T, "split": SPLIT, "mode": MODE, "n": n, "leads": leads,
                           "median_fulltraj_pct": 100 * float(np.median(full)), "median_final_pct": 100 * float(np.median(err[:, -1])),
                           "per_traj_fulltraj": full.tolist(), "fit": fitlog}
print(f"SCOT_FT15 {T} {SPLIT} {MODE}{' PAPER' if PAPER else ''}{' NORM' if PAPER_NORM else ''}: full-traj {res['median_fulltraj_pct']:.3f}%  final {res['median_final_pct']:.3f}%", flush=True)
res["metric"] = "paper (final lead, mean over functions of interest)" if PAPER else "ours (joint rel-L1, mean over leads)"
if DUMP:
    np.savez_compressed(DUMP, pred=np.stack(DP, 1), y=np.stack(DY, 1), leads=np.array(leads),
                        err=err[:DN], fulltraj=full[:DN], rows=np.arange(row0, row0 + DN))
    print(f"[dump] {DUMP}: pred {np.stack(DP, 1).shape}", flush=True)
json.dump(res, open(os.environ.get("OUT", f"/tmp/{T}_{SPLIT}_{MODE}{'_paper' if PAPER else ''}.json"), "w"))
