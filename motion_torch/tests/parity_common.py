"""Shared driver of the TF <-> PyTorch parity tests (torch env; the TF side runs in a subprocess).

For a configuration + checkpoint: (1) build test cases (synthetic samples as in motion_tf/tests/test_v3_smoke.py /
test_rh_mode.py, plus held-out real units from LOCAL_UNITS when present); (2) run the TF reference
(tests/tf_reference.py, TF 2.15, eager, TF32 off) which builds the examples with the REFERENCE make_example_ar;
(3) build the same examples with the port (must be bit-identical), run the torch model (TF32 off) and compare:
trunk tensors, the one-step prediction, the K-step rollout and the per-trajectory physical rel-L2.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PKG))

from motion_torch.config import MotionConfig  # noqa: E402
from motion_torch.convert import build_model  # noqa: E402
from motion_torch.data.example import (box_dims_for, load_local_units, make_example_ar, stack_batch,  # noqa: E402
                                       synthetic_sample)
from motion_torch.inference import predict, to_torch, trajectory_error  # noqa: E402

FV2 = os.environ.get("MOTION_FV2", os.path.join(os.path.dirname(PKG), "motion_tf"))
TF_PY = os.environ.get("TF_PYTHON", os.path.expanduser("~/anaconda3/envs/tf215/bin/python"))
LOCAL_UNITS = os.environ.get("LOCAL_UNITS", os.path.expanduser("checkpoints/units_nss_mtf.pkl"))
CACHE = os.environ.get("PARITY_CACHE", os.path.expanduser("~/.cache/motion_torch_parity"))

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def synthetic_cases(cfg, fams=("incom_ns", "CE-RP", "ACE", "pdebench3d_cns_Rand_M0.1_Eta1e-08_Zeta1e-08",
                               "Poisson-Gauss"), B=2, k_roll=10):
    rng = np.random.default_rng(1234)
    cases = []
    for fam in fams:
        from motion_torch.data.registry import FAMILIES
        K = FAMILIES[fam].K
        dims = (64, 64) if K == 2 else (16, 16, 16)
        samples = [synthetic_sample(fam, rng, dims=dims, T=20) for _ in range(B)]
        cases.append({"name": f"syn_{fam}", "samples": samples, "seeds": [11 + i for i in range(B)],
                      "t_in": cfg.t_in, "k_fut": 10, "k_relax": 4, "n_colloc": 512 if K == 3 else 1024,
                      "enc_n": 2048, "box_dims": (box_dims_for(fam, cfg) if K == 2 else   # 3-D synthetic: <= 16^3
                                                  tuple(min(v, 16) for v in box_dims_for(fam, cfg))),
                      "k_roll": None if fam == "Poisson-Gauss"
                      else k_roll, "light": False})
    return cases


def real_cases(cfg, fams, n_units=8, k_roll=10):
    if not os.path.exists(LOCAL_UNITS):
        print(f"[parity] LOCAL_UNITS {LOCAL_UNITS} not found: real-data comparison skipped")
        return []
    U = load_local_units(LOCAL_UNITS)
    cases = []
    for fam in fams:
        if fam not in U:
            continue
        for i, u in enumerate(U[fam][:n_units]):
            cases.append({"name": f"real_{fam}_{i}", "samples": [u], "unit_ref": [(LOCAL_UNITS, fam, i)],
                          "seeds": [1000 + i], "t_in": cfg.t_in,
                          "k_fut": 10, "k_relax": 4, "n_colloc": 2048, "enc_n": 8192,
                          "box_dims": box_dims_for(fam, cfg), "k_roll": k_roll, "light": True, "group": fam})
    return cases


def _run_tf_group(cfg, ckpt, cases, tag):
    tf_cases = [{k: v for k, v in c.items() if not (k == "samples" and "unit_ref" in c)} for c in cases]
    blob = {"cfg": {k: v for k, v in cfg.__dict__.items()}, "ckpt": ckpt, "fv2": FV2, "cases": tf_cases}
    h = hashlib.md5(pickle.dumps((blob["cfg"], ckpt, [(c["name"], c["seeds"], c["box_dims"], c["k_roll"],
                                                       c["t_in"], c["light"]) for c in cases]))).hexdigest()[:10]
    cpath = os.path.join(CACHE, f"cases_{tag}_{h}.pkl")
    opath = os.path.join(CACHE, f"tfref_{tag}_{h}.pkl")
    if not os.path.exists(opath):
        with open(cpath, "wb") as fh:
            pickle.dump(blob, fh, protocol=4)
        env = {k: v for k, v in os.environ.items() if not (k.startswith("MOTION_") or k == "NO_DEAD_REPAIR")}
        env.update(PYTHONDONTWRITEBYTECODE="1", TF_CPP_MIN_LOG_LEVEL="2", TF_FORCE_GPU_ALLOW_GROWTH="true")
        t0 = time.time()
        r = subprocess.run([TF_PY, "-u", os.path.join(HERE, "tf_reference.py"), cpath, opath], env=env)
        assert r.returncode == 0, "TF reference failed"
        print(f"[parity] TF reference {tag}: {time.time() - t0:.0f}s")
        os.remove(cpath)
    with open(opath, "rb") as fh:
        return pickle.load(fh)


def run_tf(cfg, ckpt, cases, tag):
    """Run (or reuse a cached) TF reference. One TF process per case group (synthetic / each real family), so the
    large-box families do not run out of GPU memory behind the others."""
    os.makedirs(CACHE, exist_ok=True)
    groups = {}
    for c in cases:
        groups.setdefault(c.get("group", "syn"), []).append(c)
    out = {"cases": {}}
    for g, cs in groups.items():
        r = _run_tf_group(cfg, ckpt, cs, f"{tag}_{g}")
        out["var_names"] = r["var_names"]
        out["cases"].update(r["cases"])
    return out


def _md5(v):
    return hashlib.md5(np.ascontiguousarray(v).tobytes()).hexdigest()


def diff(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    d = np.abs(a - b)
    return {"max_abs": float(d.max()), "max_ref": float(np.abs(b).max()),
            "rel_max": float(d.max() / max(np.abs(b).max(), 1e-30)),
            "rel_l2": float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))}


def compare_case(model, cfg, c, ref, device, model64=None):
    exs = [make_example_ar(s, c["t_in"], c["k_fut"], c["k_relax"], c["n_colloc"], c["enc_n"],
                           np.random.default_rng(sd), cfg, box_dims=c["box_dims"], global_norm=c.get("global_norm"))
           for s, sd in zip(c["samples"], c["seeds"])]
    b = stack_batch(exs)
    # ---- data parity: the port's example == the reference example (bit-identical)
    bad = []
    for k, v in b.items():
        if not isinstance(v, np.ndarray):
            continue
        rv = ref["batch"][k]
        same = (_md5(v) == rv) if isinstance(rv, str) else (v.shape == rv.shape and np.array_equal(v, rv))
        if not same:
            bad.append(k)
    assert b["mode"] == ref["mode"], (b["mode"], ref["mode"])
    tb = to_torch(b, device)
    out = {"name": c["name"], "mode": b["mode"], "data_mismatch": bad}
    with torch.no_grad():
        if ref.get("aux") is not None:
            h, W, e, _ = model.trunk(tb["coords_node"], tb["x_win"], tb["fmask"], tb["geom_node"], b["K"],
                                     list(b["roles"]), tb["cond"], tb["op_multihot"], tb["sdf_box"],
                                     tuple(b["box_dims"]), b["steady"])
            out["h"] = diff(h.cpu().numpy(), ref["aux"]["h"])
            out["W"] = diff(W.cpu().numpy(), ref["aux"]["W"])
        t0 = time.time()
        o = predict(model, tb, k_seg=c.get("k_roll"))
        if device == "cuda":
            torch.cuda.synchronize()
        out["seconds"] = time.time() - t0
    P = o["pred"].cpu().numpy()
    R = ref["pred"]
    if c.get("light"):
        P = P[:, [0, -1]]
    u0 = np.concatenate([b["x_win"][:, -1], b["ic_q"]], 1)
    out["step1"] = diff(P[:, 0], R[:, 0])
    out["step1_delta"] = diff(P[:, 0] - u0, R[:, 0] - u0)             # the increment the network produces
    out["last"] = diff(P[:, -1], R[:, -1])
    out["rollout"] = diff(P, R)
    if b["mode"] == "ss" and "pred_leads_1_3" in ref:
        with torch.no_grad():
            oL = predict(model, tb, k_seg=c.get("k_roll"), leads=[1, 3], with_metrics=False)["pred"].cpu().numpy()
        out["leads_1_3"] = diff(oL, ref["pred_leads_1_3"])
    if model64 is not None and not c.get("light"):
        # float64 run of the SAME port (Fourier phases stay float32, as in TF): how far each float32 implementation
        # is from the exact-arithmetic result = the float32 noise floor of this case
        tb64 = {k: (v.double() if torch.is_tensor(v) and v.dtype == torch.float32 else v) for k, v in tb.items()}
        with torch.no_grad():
            D = predict(model64, tb64, k_seg=c.get("k_roll"), with_metrics=False)["pred"].cpu().numpy()
        out["f64_vs_tf"] = diff(R, D)
        out["f64_vs_torch"] = diff(P, D)
        out["f64_vs_tf_step1"] = diff(R[:, 0], D[:, 0])
        out["f64_vs_torch_step1"] = diff(P[:, 0], D[:, 0])
    w = b["tstep_mask"] if b["mode"] != "steady" else None
    rp_t = o["rel_phys"].cpu().numpy()
    rp_f = ref["rel_phys"]
    if w is not None:
        wk = w[:, :rp_t.shape[1]]
        tr_t = (rp_t * wk).sum(-1) / np.maximum(wk.sum(-1), 1e-6)
        tr_f = (rp_f * wk).sum(-1) / np.maximum(wk.sum(-1), 1e-6)
    else:
        tr_t, tr_f = rp_t.mean(-1), rp_f.mean(-1)
    out["traj_phys_torch"] = tr_t.tolist()
    out["traj_phys_tf"] = tr_f.tolist()
    out["traj_phys_absdiff"] = float(np.abs(tr_t - tr_f).max())
    return out


def run_parity(tag, cfg: MotionConfig, ckpt, cases, device=None, tol=1e-5):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ref = run_tf(cfg, ckpt, cases, tag)
    model = build_model(cfg, ckpt, device=device)
    import copy
    model64 = copy.deepcopy(model).double()
    tmap = model.tf_name_map()
    missing = sorted(set(ref["var_names"]) - set(tmap.values()))
    extra = sorted(set(tmap.values()) - set(ref["var_names"]))
    print(f"[parity] {tag}: TF vars {len(ref['var_names'])} | torch params {len(tmap)} | TF-only {missing} | "
          f"torch-only {extra}")
    rows = []
    for c in cases:
        r = compare_case(model, cfg, c, ref["cases"][c["name"]], device, model64)
        rows.append(r)
        print(f"  {r['name']:42s} {r['mode']:6s} data {'OK' if not r['data_mismatch'] else r['data_mismatch']} | "
              + (f"h {r['h']['rel_max']:.1e} W {r['W']['rel_max']:.1e} | " if "h" in r else "")
              + f"step1 rel {r['step1']['rel_max']:.1e} (delta {r['step1_delta']['rel_max']:.1e}) | "
              f"rollout rel {r['rollout']['rel_max']:.1e} last {r['last']['rel_max']:.1e} | traj phys "
              f"tf {100 * np.mean(r['traj_phys_tf']):.4f}% torch {100 * np.mean(r['traj_phys_torch']):.4f}% "
              f"|d| {r['traj_phys_absdiff']:.1e}" + (f" | leads {r['leads_1_3']['rel_max']:.1e}"
                                                     if "leads_1_3" in r else "")
              + (f" | f64 ref step1/roll: tf {r['f64_vs_tf_step1']['rel_max']:.1e}/{r['f64_vs_tf']['rel_max']:.1e}"
                 f" torch {r['f64_vs_torch_step1']['rel_max']:.1e}/{r['f64_vs_torch']['rel_max']:.1e}"
                 if "f64_vs_tf" in r else ""), flush=True)
    summ = {"tag": tag, "ckpt": ckpt, "cfg": cfg.name, "env": cfg.env(), "tf_only_vars": missing,
            "torch_only_params": extra, "rows": rows}
    with open(os.path.join(HERE, f"parity_{tag}.json"), "w") as fh:
        json.dump(summ, fh, indent=1)
    # criterion: one-step and rollout difference <= tol or -- where the case amplifies float32 rounding itself --
    # not larger than 3x the distance of the float32 results (TF or torch) from a float64 run
    def _ok(r):
        def within(key, f64a, f64b):
            if r[key]["rel_max"] <= tol:
                return True
            # 11-09: the float32 noise floor is the distance of the PORT's float32 result from its own float64 run;
            # using max(TF, torch) was circular (a TF/port mismatch inflated the TF distance and passed itself)
            return f64b in r and r[key]["rel_max"] <= 3.0 * r[f64b]["rel_max"]
        return (within("step1", "f64_vs_tf_step1", "f64_vs_torch_step1")
                and within("rollout", "f64_vs_tf", "f64_vs_torch"))
    worst1 = max(r["step1"]["rel_max"] for r in rows)
    worstR = max(r["rollout"]["rel_max"] for r in rows)
    fails = [r["name"] for r in rows if not _ok(r)]
    ok = not fails and not missing and not extra and not any(r["data_mismatch"] for r in rows)
    summ.update(worst_step1=worst1, worst_rollout=worstR, fails=fails, ok=ok)
    with open(os.path.join(HERE, f"parity_{tag}.json"), "w") as fh:
        json.dump(summ, fh, indent=1)
    print(f"[parity] {tag}: worst rel. difference one-step {worst1:.2e}, rollout {worstR:.2e} (target {tol:.0e}; "
          f"float32-noise rule for rollouts) -> {'PASS' if ok else 'FAIL ' + str(fails)}")
    return summ, ok
