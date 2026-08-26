"""Causal ablation of a single tendency bank at a fixed checkpoint.

Reports, per family, the eval metric with the bank ACTIVE and with its gate ZEROED — the gate is
the bank's only output path, so zeroing it removes exactly that mechanism and nothing else. Also
prints every gate's magnitude, which tells you whether a bank ever opened at all (a bank whose
gate is ~1e-4 cannot have contributed regardless of what the ablation shows).

  python ablate_bank.py <ckpt.npz> [bank_name ...]      # default: compressible
"""
from __future__ import annotations

import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.pretrain import load_ckpt                      # noqa: E402
from data import loader                                  # noqa: E402
from data.registry import FAMILIES, pretrain_families    # noqa: E402
from data.splits import load_manifest                    # noqa: E402
from v3.data_ar import make_example_ar, stack_batch      # noqa: E402
from v3.model import V3Model                             # noqa: E402
from v3.train import rollout_loss                        # noqa: E402

N_EVAL = int(os.environ.get("ABL_N", "8"))
KEYS = ("coords_node", "x_win", "fmask", "geom_node", "y_node", "ic_q", "coords_q",
        "geom_q", "y_q", "tstep_mask", "cmask", "op_multihot", "cond", "fam_id", "sdf_box")


def evaluate(model, fams):
    out = {}
    for fam in fams:
        rng = np.random.default_rng(1234)
        vals, n = [], 0
        try:
            for s in loader.units_for(fam, "val"):
                K = int(s["K"])
                ex = make_example_ar(s, 10, 10, 4, 2048, 8192, rng,
                                     box_dims=(64, 64) if K == 2 else (32, 32, 32))
                b = stack_batch([ex])
                ts = tuple(tf.constant(b[k]) for k in KEYS)
                gd = tuple(b.get("dims", ())) or None
                st = (K, bool(b["steady"]), bool(b["dense2d"]),
                      tuple(int(r) for r in b["roles"]), int(b["k_seg"]), gd)
                vals.append(float(rollout_loss(model, *ts, *st, clamp=1e9)))
                n += 1
                if n >= N_EVAL:
                    break
        except Exception as e:
            print(f"  [skip] {fam}: {type(e).__name__} {e}", flush=True)
        if vals:
            out[fam] = float(np.mean(vals))
    return out


def main():
    ckpt = sys.argv[1]
    banks = sys.argv[2:] or ["compressible"]
    fams = [f.name for f in pretrain_families()]
    fams = [f for f in fams if _has_manifest(f)]

    model = V3Model(S=9, d=384, depth=8, d_w=32, n_p=8, d_cond=128, dims2=(64, 64),
                    dims3=(32, 32, 32), t_in=10, transport=True,
                    n_fams=len(FAMILIES), n_slices=64)
    model.warm_build()
    V = model.trainable_variables
    load_ckpt(V, ckpt, d_w=32, names=[v.name for v in V])

    print("\n=== gate magnitudes (a gate near 0 means the bank never opened) ===")
    for v in sorted(V, key=lambda v: v.name):
        if "gate_" in v.name:
            a = float(tf.reduce_mean(tf.abs(v)))
            print(f"  {v.name.split('gate_')[-1].split(':')[0]:16s} {a:.5f}")

    base = evaluate(model, fams)
    print(f"\n=== ablation: zeroing {banks} ===")
    saved = {}
    for b in banks:
        g = model.banks.gates[b]
        saved[b] = g.numpy().copy()
        g.assign(tf.zeros_like(g))
    off = evaluate(model, fams)
    for b, w in saved.items():
        model.banks.gates[b].assign(w)

    print(f"\n{'family':40s} {'with':>8} {'without':>8} {'delta':>8}")
    tot_a = tot_b = 0.0
    for f in sorted(base, key=lambda f: (off.get(f, 0) - base[f])):
        if f not in off:
            continue
        d = off[f] - base[f]
        tot_a += base[f]; tot_b += off[f]
        flag = "  <-- bank HELPS" if d > 0.002 else ("  <-- bank HURTS" if d < -0.002 else "")
        print(f"{f:40s} {100*base[f]:8.2f} {100*off[f]:8.2f} {100*d:+8.2f}{flag}")
    n = len(base)
    print(f"\n{'CLASS-AVG':40s} {100*tot_a/n:8.2f} {100*tot_b/n:8.2f} {100*(tot_b-tot_a)/n:+8.2f}")
    print("(delta > 0 means removing the bank made it WORSE, i.e. the bank was helping)")
    print("ABLATE_OK")


def _has_manifest(f):
    try:
        load_manifest(f)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
