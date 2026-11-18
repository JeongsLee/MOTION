"""TensorFlow REFERENCE runner for the parity tests (run with the TF 2.15 interpreter; never imported by the port).

  python tests/tf_reference.py CASES.pkl OUT.pkl

CASES.pkl (written by tests/parity_common.py in the torch env): {"cfg": MotionConfig dict, "ckpt": npz path,
"fv2": path of the motion_tf reference code, "cases": [{name, samples, seeds, t_in, k_fut, k_relax, n_colloc, enc_n,
box_dims, k_roll, global_norm}]}. The MOTION_* environment of the configuration is set BEFORE the reference modules
are imported (they read it at import time). For each case the examples are built by the REFERENCE make_example_ar
from the given samples and seeds; outputs (eager TF, TF32 off): the batch, one-step aux tensors (h, W, z, delta,
u_next), the K-step rollout (or single-shot / receding-horizon output for those modes) and the per-frame metrics.
"""
import importlib.util
import os
import pickle
import sys
import time

cases_path, out_path = sys.argv[1], sys.argv[2]
with open(cases_path, "rb") as fh:
    C = pickle.load(fh)
_spec = importlib.util.spec_from_file_location(
    "mt_config", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.py"))
_cfgmod = importlib.util.module_from_spec(_spec)
sys.modules["mt_config"] = _cfgmod
_spec.loader.exec_module(_cfgmod)
cfg = _cfgmod.MotionConfig(**C["cfg"])
for k in list(os.environ):
    if k.startswith("MOTION_") or k == "NO_DEAD_REPAIR":
        del os.environ[k]
os.environ.update(cfg.env())
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, C["fv2"])
sys.dont_write_bytecode = True                                  # never touch the reference tree

import numpy as np
import tensorflow as tf

tf.config.experimental.enable_tensor_float_32_execution(False)
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)

from core.pretrain import load_ckpt
from data.registry import FAMILIES, NUM_SLOTS, SEMANTIC_SLOTS
from data.symbolic import OPERATOR_VOCAB
from v3 import data_ar as DA
from v3.data_ar import make_example_ar, stack_batch
from v3.model import V3Model, interp_box
from v3.train import _grid, _rel

print(f"[tf-ref] env {cfg.env()} | vocab {len(OPERATOR_VOCAB)}", flush=True)
model = V3Model(S=NUM_SLOTS, d=cfg.d, depth=cfg.depth, d_w=cfg.d_w, n_p=cfg.n_p, d_cond=cfg.d_cond,
                dims2=tuple(cfg.dims2), dims3=tuple(cfg.dims3), t_in=cfg.t_in, transport=cfg.transport,
                n_fams=len(FAMILIES), n_semantic=SEMANTIC_SLOTS, gate_cond=cfg.gate_cond, dict_decode=0,
                fam_head=cfg.fam_head)
model.warm_build()
V = model.trainable_variables
if C.get("ckpt"):
    load_ckpt(V, C["ckpt"], d_w=cfg.d_w, names=[v.name for v in V])
var_names = [v.name for v in V]
print(f"[tf-ref] {len(V)} vars, {sum(int(np.prod(v.shape)) for v in V) / 1e6:.2f}M", flush=True)
KEYS = ("coords_node", "x_win", "fmask", "geom_node", "y_node", "ic_q", "coords_q", "geom_q", "y_q", "tstep_mask",
        "cmask", "fbmask", "op_multihot", "cond", "fam_id", "sdf_box")


def fix_vocab(s):
    s = dict(s)
    mh = np.asarray(s["op_multihot"], np.float32)
    if len(mh) == len(OPERATOR_VOCAB) + 1:                     # LOCAL_UNITS restore (latent_control_k.py)
        assert float(mh[-1]) == 0.0
        s["op_multihot"] = mh[:-1]
    return s


STEP_GRAPH = tf.function(model.step, reduce_retracing=True)


def T(x):
    return tf.constant(x)


_UNITS = {}


def unit_from_ref(path, fam, i):
    """Held-out unit from a LOCAL_UNITS pickle (motion_tf/export_units.py), restored as latent_control_k.py does."""
    if path not in _UNITS:
        with open(path, "rb") as fh:
            _UNITS[path] = pickle.load(fh)
    u = dict(_UNITS[path][fam][i])
    f = u.pop("fields"); full = np.zeros(f.shape[:-1] + (int(u.pop("_S")),), f.dtype)
    full[..., u.pop("_act")] = f; u["fields"] = full
    return u


def run_case(c):
    if "unit_ref" in c:
        c = dict(c, samples=[unit_from_ref(*r) for r in c["unit_ref"]])
    if c.get("global_norm"):
        DA.GLOBAL_NORM.clear()
        DA.GLOBAL_NORM.update({f: tuple(np.asarray(v, np.float32) for v in mv) for f, mv in c["global_norm"].items()})
    exs = [make_example_ar(fix_vocab(s), c["t_in"], c["k_fut"], c["k_relax"], c["n_colloc"], c["enc_n"],
                           np.random.default_rng(sd), box_dims=c["box_dims"])
           for s, sd in zip(c["samples"], c["seeds"])]
    b = stack_batch(exs)
    ts = {k: T(b[k]) for k in KEYS}
    K, steady, dense2d = int(b["K"]), bool(b["steady"]), bool(b["dense2d"])
    roles = tuple(int(r) for r in b["roles"])
    bd = tuple(int(v) for v in b["box_dims"])
    gd = tuple(b.get("dims", ())) or None
    k_seg = int(c.get("k_roll") or b["k_seg"])
    cn, gn, xw, fm = ts["coords_node"], ts["geom_node"], ts["x_win"], ts["fmask"]
    qall = tf.concat([cn, ts["coords_q"]], 1)
    gall = tf.concat([gn, ts["geom_q"]], 1)
    u_prev = tf.concat([xw[:, -1], ts["ic_q"]], 1)
    sc = (np.concatenate([b["scale"], b["shift"]], -1) if cfg.input_zscore else b["scale"])
    res = {"batch": {k: v for k, v in b.items()}, "mode": b["mode"]}
    if not c.get("light"):
        # ---- trunk intermediates (same sequence as V3Model.step)
        feats = model.feats_from_win(xw, fm, gn)
        box, geom_box, _ = model.enc(cn, feats, dims=bd, roles=list(roles))
        box = box + model.sdf_gate(model.sdf_proj(ts["sdf_box"]))
        e = model.cond(ts["cond"])
        h = model.core(box, e, list(roles))
        geom_box = tf.concat([ts["sdf_box"], geom_box[..., 1:]], -1)
        W = model.banks(h, K=K, roles=list(roles), op_multihot=ts["op_multihot"], geom_box=geom_box, steady=steady,
                        e=e)
        res["aux"] = {"box": box.numpy(), "e": e.numpy(), "h": h.numpy(), "W": W.numpy()}
        del feats, box, geom_box, e, h, W
    # real-data cases run the step as a tf.function (graph mode, no XLA): eager TF runs out of memory on the
    # 128^2-box families on a 12 GB GPU
    step = STEP_GRAPH if c.get("light") else model.step
    t0 = time.time()
    if b["mode"] in ("ar", "steady"):
        prev_grid = prev2_grid = None
        if dense2d:
            prev_grid = _grid(xw[:, -1], gd, model.S)
            prev2_grid = _grid(xw[:, -2], gd, model.S) if int(xw.shape[1]) >= 2 else None
        win, fmask, up = xw, fm, u_prev
        Nn = int(cn.shape[1])
        preds, rn, rp, rf = [], [], [], []
        for k in range(k_seg):
            pred = step(cn, win, fmask, gn, K, list(roles), ts["cond"], ts["op_multihot"], qall, gall, up,
                        steady=steady, prev_grid=prev_grid, grid_dims=gd, fam_id=ts["fam_id"],
                        sdf_box=ts["sdf_box"], box_dims=bd, fbmask=ts["fbmask"], surf=False,
                        prev2_grid=prev2_grid)
            kt = 0 if steady else k
            y_all = tf.concat([ts["y_node"][:, kt], ts["y_q"][:, kt]], 1)
            rn.append(_rel(pred, y_all, ts["cmask"], False).numpy())
            rp.append(_rel(pred, y_all, ts["cmask"], False, T(sc)).numpy())
            rf.append(_rel(pred, y_all, ts["cmask"], True, fluct=True).numpy())
            preds.append(pred.numpy())
            up = pred
            pred_n = up[:, :Nn]
            win = tf.concat([win[:, 1:], pred_n[:, None]], 1)
            fmask = tf.concat([fmask[:, 1:], tf.ones_like(fmask[:, :1])], 1)
            if dense2d:
                prev2_grid, prev_grid = prev_grid, _grid(pred_n, gd, model.S)
        res["pred"] = np.stack(preds, 1)
    else:
        pg = _grid(xw[:, -1], gd, model.S) if (gd is not None and getattr(model, "ss_warp", False)) else None
        if b["mode"] == "ss":
            P = model.step_ss(cn, xw, fm, gn, K, list(roles), ts["cond"], ts["op_multihot"], qall, gall, u_prev,
                              kf=k_seg, fam_id=ts["fam_id"], sdf_box=ts["sdf_box"], box_dims=bd, prev_grid=pg,
                              grid_dims=gd)
            res["pred"] = P.numpy()
            res["pred_leads_1_3"] = model.step_ss(cn, xw, fm, gn, K, list(roles), ts["cond"], ts["op_multihot"],
                                                  qall, gall, u_prev, kf=k_seg, fam_id=ts["fam_id"],
                                                  sdf_box=ts["sdf_box"], box_dims=bd, prev_grid=pg, grid_dims=gd,
                                                  leads=[1, 3]).numpy()
            preds = [P[:, k] for k in range(k_seg)]
        else:                                                    # "rh": hop-1 AR of the single-shot operator
            win, fmask, up = xw, fm, u_prev
            Nn = int(cn.shape[1])
            preds = []
            for r in range(k_seg):
                p = model.step_ss(cn, win, fmask, gn, K, list(roles), ts["cond"], ts["op_multihot"], qall, gall, up,
                                  kf=k_seg, fam_id=ts["fam_id"], sdf_box=ts["sdf_box"], box_dims=bd, prev_grid=pg,
                                  grid_dims=gd, leads=[1])[:, 0]
                preds.append(p)
                up = p
                pred_n = up[:, :Nn]
                win = tf.concat([win[:, 1:], pred_n[:, None]], 1)
                fmask = tf.concat([fmask[:, 1:], tf.ones_like(fmask[:, :1])], 1)
                if pg is not None:
                    pg = _grid(pred_n, gd, model.S)
            res["pred"] = np.stack([p.numpy() for p in preds], 1)
        rn, rp, rf = [], [], []
        for k in range(k_seg):
            y_all = tf.concat([ts["y_node"][:, k], ts["y_q"][:, k]], 1)
            rn.append(_rel(preds[k], y_all, ts["cmask"], False).numpy())
            rp.append(_rel(preds[k], y_all, ts["cmask"], False, T(sc)).numpy())
            rf.append(_rel(preds[k], y_all, ts["cmask"], True, fluct=True).numpy())
    res["rel_norm"], res["rel_phys"], res["rel_fluct"] = (np.stack(rn, 1), np.stack(rp, 1), np.stack(rf, 1))
    res["seconds"] = time.time() - t0
    if c.get("light"):                                           # real-data cases: checksums + first/last frame only
        import hashlib
        res["batch"] = {k: (hashlib.md5(np.ascontiguousarray(v).tobytes()).hexdigest() if isinstance(v, np.ndarray)
                            else v) for k, v in b.items()}
        res["aux"] = None
        res["pred"] = res["pred"][:, [0, -1]]
    print(f"[tf-ref] {c['name']}: mode {b['mode']} K={K} B={len(exs)} N={b['coords_node'].shape[1]} "
          f"box {bd} k={k_seg}  phys rel {100 * res['rel_phys'].mean():.3f}%  ({res['seconds']:.1f}s)", flush=True)
    return res


out = {"var_names": var_names, "cases": {}}
for c in C["cases"]:
    out["cases"][c["name"]] = run_case(c)
with open(out_path, "wb") as fh:
    pickle.dump(out, fh, protocol=4)
print("TF_REF_OK", out_path, flush=True)
