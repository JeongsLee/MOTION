"""(a) physical-vs-normalized scoring for one checkpoint, and (b) how much the model leans on
observed history. Evaluate one checkpoint at several
history lengths, from the full window down to a single frame (a true IVP).

Nothing in the architecture has to change: the observed window is a FIXED-width buffer that is
right-aligned and carries a per-frame validity mask, and the six steady families already train
with an all-zero mask. Shortening the history is therefore just a data-side choice — but for
transient families the short-mask pattern was never seen in training, so this measures
out-of-distribution robustness, not trained capability.

  python eval_tin.py <ckpt.npz> [t_in_list]      # default 10,4,2,1
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

N_EVAL = int(os.environ.get("TIN_N", "6"))
PHYS = bool(int(os.environ.get("TIN_PHYS", "0")))   # 1 = denormalize before scoring
ENC_N = int(os.environ.get("TIN_ENC", "2048"))    # smaller than training: this runs EAGER, and
N_COL = int(os.environ.get("TIN_COL", "512"))     # 512 tokens x 18k queries OOMs a dense 2D grid
T_IN_MAX = 10
KEYS = ("coords_node", "x_win", "fmask", "geom_node", "y_node", "ic_q", "coords_q",
        "geom_q", "y_q", "tstep_mask", "cmask", "op_multihot", "cond", "fam_id", "sdf_box")


def trim_history(ex, keep):
    """Keep only the newest `keep` observed frames; the buffer width is unchanged, the mask
    and the zero-padding do the work (exactly how a shorter history would arrive at inference)."""
    ex = dict(ex)
    xw = ex["x_win"].copy()
    fm = ex["fmask"].copy()
    live = int(fm.sum())
    if live > keep:
        drop = live - keep
        lo = T_IN_MAX - live
        xw[lo:lo + drop] = 0.0
        fm[lo:lo + drop] = 0.0
    ex["x_win"], ex["fmask"] = xw, fm
    return ex


def main():
    ckpt = sys.argv[1]
    tins = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["10", "4", "2", "1"])]
    fams = []
    for f in pretrain_families():
        try:
            load_manifest(f.name)
            fams.append(f.name)
        except Exception:
            pass

    model = V3Model(S=9, d=384, depth=8, d_w=32, n_p=8, d_cond=128, dims2=(64, 64),
                    dims3=(32, 32, 32), t_in=T_IN_MAX, transport=True,
                    n_fams=len(FAMILIES), n_slices=512)
    model.warm_build()
    V = model.trainable_variables
    load_ckpt(V, ckpt, d_w=32, names=[v.name for v in V])

    res = {t: {} for t in tins}
    for fam in fams:
        steady = FAMILIES[fam].time == "steady"
        rng = np.random.default_rng(1234)
        pool = []
        try:
            for s in loader.units_for(fam, "val"):
                K = int(s["K"])
                pool.append(make_example_ar(s, T_IN_MAX, 10, 4, N_COL, ENC_N, rng,
                                            box_dims=(64, 64) if K == 2 else (32, 32, 32)))
                if len(pool) >= N_EVAL:
                    break
        except Exception as e:
            print(f"  [skip] {fam}: {type(e).__name__}", flush=True)
            continue
        for t in tins:
            if steady and t != tins[0]:
                res[t][fam] = res[tins[0]][fam]        # steady has no history to trim
                continue
            vals = []
            for ex in pool:
                b = stack_batch([trim_history(ex, t)])
                ts = tuple(tf.constant(b[k]) for k in KEYS)
                gd = tuple(b.get("dims", ())) or None
                st = (int(b["K"]), bool(b["steady"]), bool(b["dense2d"]),
                      tuple(int(r) for r in b["roles"]), int(b["k_seg"]), gd)
                vals.append(float(rollout_loss(model, *ts, *st, clamp=1e9,
                                               scale=tf.constant(b["scale"]) if PHYS else None)))
            res[t][fam] = float(np.mean(vals))
        print(f"  {fam:40s} " + "  ".join(f"t{t}={100*res[t][fam]:6.2f}" for t in tins), flush=True)

    print(f"\n{'family':40s} " + "".join(f"{'t_in='+str(t):>10s}" for t in tins))
    for fam in sorted(res[tins[0]], key=lambda f: -(res[tins[-1]].get(f, 0) - res[tins[0]][f])):
        print(f"{fam:40s} " + "".join(f"{100*res[t][fam]:10.2f}" for t in tins))
    print(f"\n{'CLASS-AVG':40s} " +
          "".join(f"{100*np.mean(list(res[t].values())):10.2f}" for t in tins))
    tr = [f for f in res[tins[0]] if FAMILIES[f].time != "steady"]
    print(f"{'TRANSIENT-ONLY':40s} " +
          "".join(f"{100*np.mean([res[t][f] for f in tr]):10.2f}" for t in tins))
    print("EVALTIN_OK")


if __name__ == "__main__":
    main()
