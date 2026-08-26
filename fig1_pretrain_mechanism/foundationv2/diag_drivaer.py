"""Diagnose the drivaernet degradation: is it DC/scale, spatial structure, or the relaxation loop?

The family improves early then worsens monotonically in r2, r3 AND r4 — three runs, two different
loss functions — so it is not the per-channel loss and not the geometry starvation (r4 has the SDF
volume + recovered normals and still degrades). This dumps, for the SAME val designs at an EARLY
and a LATE checkpoint, the prediction statistics per relaxation step so the failure mode is visible
rather than inferred:

  * mean / std of prediction vs target  -> DC-offset or amplitude collapse
  * correlation of prediction with target -> spatial structure retained or lost
  * the same across the k_relax steps -> does the relaxation diverge or saturate

Run:  python diag_drivaer.py <early_ckpt> <late_ckpt>
"""
from __future__ import annotations

import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.pretrain import load_ckpt                       # noqa: E402
from data import loader                                   # noqa: E402
from data.registry import FAMILIES                        # noqa: E402
from v3.data_ar import make_example_ar, stack_batch       # noqa: E402
from v3.model import V3Model                              # noqa: E402

FAM = os.environ.get("DIAG_FAM", "drivaernet_pressure")
N = int(os.environ.get("DIAG_N", "6"))
KREL = 4


def build():
    m = V3Model(S=9, d=384, depth=8, d_w=32, n_p=8, d_cond=128, dims2=(64, 64),
                dims3=(32, 32, 32), t_in=10, transport=True,
                n_fams=len(FAMILIES), n_slices=64)
    m.warm_build()
    return m


def examples():
    rng = np.random.default_rng(1234)
    out = []
    for s in loader.units_for(FAM, "val"):
        out.append(make_example_ar(s, 10, 10, KREL, 2048, 8192, rng, box_dims=(32, 32, 32)))
        if len(out) >= N:
            break
    return out


def run(model, exs, tag):
    print(f"\n===== {tag} =====")
    print(f"{'design':>6} {'step':>4} {'pred_mean':>10} {'pred_std':>9} "
          f"{'tgt_mean':>9} {'tgt_std':>8} {'corr':>6} {'relL2':>7}")
    for di, ex in enumerate(exs):
        b = stack_batch([ex])
        cn = tf.constant(b["coords_node"]); gn = tf.constant(b["geom_node"])
        cq = tf.constant(b["coords_q"]); gq = tf.constant(b["geom_q"])
        cond = tf.constant(b["cond"]); op = tf.constant(b["op_multihot"])
        fid = tf.constant(b["fam_id"]); sdb = tf.constant(b["sdf_box"])
        cm = tf.constant(b["cmask"])
        qall = tf.concat([cn, cq], 1); gall = tf.concat([gn, gq], 1)
        win = tf.constant(b["x_win"]); fm = tf.constant(b["fmask"])
        u_prev = tf.concat([tf.constant(b["x_win"])[:, -1], tf.constant(b["ic_q"])], 1)
        y = tf.concat([tf.constant(b["y_node"])[:, 0], tf.constant(b["y_q"])[:, 0]], 1)
        slot = int(np.argmax(b["cmask"][0]))
        for k in range(KREL):
            pred = model.step(cn, win, fm, gn, 3, (0, 1, 2), cond, op, qall, gall, u_prev,
                              steady=True, prev_grid=None, grid_dims=None,
                              fam_id=fid, sdf_box=sdb)
            p = pred[0, :, slot].numpy(); t = y[0, :, slot].numpy()
            c = float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-9 else float("nan")
            rel = float(np.linalg.norm(p - t) / (np.linalg.norm(t) + 1e-12))
            print(f"{di:>6} {k:>4} {p.mean():>10.3f} {p.std():>9.3f} "
                  f"{t.mean():>9.3f} {t.std():>8.3f} {c:>6.3f} {rel:>7.3f}")
            u_prev = pred
            win = tf.concat([win[:, 1:], pred[:, :tf.shape(cn)[1]][:, None]], 1)
            fm = tf.concat([fm[:, 1:], tf.ones_like(fm[:, :1])], 1)


if __name__ == "__main__":
    exs = examples()
    print(f"[diag] {FAM}: {len(exs)} val designs, {exs[0]['coords_node'].shape[0]} nodes, "
          f"scale(std used to normalize) = {exs[0]['scale'][np.argmax(exs[0]['cmask'])]:.3f}")
    for ck in sys.argv[1:]:
        model = build()
        load_ckpt(model.trainable_variables, ck, d_w=32,
                  names=[v.name for v in model.trainable_variables])
        run(model, exs, os.path.basename(ck))
    print("DIAG_OK")
