"""Print the FamilyGate α_k per expert for each family from the latest checkpoint — the
interpretability fingerprint. softplus gate → α≥0, unnormalized (per-expert magnitude).

    python -m motion_tf.eval.gate_probe [config-dotted-path]
"""
from __future__ import annotations
import glob
import importlib
import os
import sys

import numpy as np
import tensorflow as tf

from ..model import PhysicsOperatorMixture
from ..utils import ckpt as ckptlib
from ..data.prose import SPEC


def main(cfg_path="motion_tf.train.configs.prose_multi_mgpu"):
    cfg = importlib.import_module(cfg_path).Cfg
    m = PhysicsOperatorMixture(cfg)
    sd = getattr(cfg, "save_dir", "results/prose_multi_mgpu")
    cks = sorted(glob.glob(os.path.join("/code-vol", sd, "ckpt_*.npz")),
                 key=lambda p: int(p.split("_")[-1].split(".")[0]))
    if not cks:
        print("no ckpt found under", os.path.join("/code-vol", sd)); return
    ckptlib.load(m, cks[-1])
    print("loaded", cks[-1], flush=True)
    # --- param breakdown (each expert is effectively only ~tot/n_experts) ---
    nn = lambda vs: int(sum(np.prod(v.shape) for v in vs))
    tot = nn(m.trainable_variables)
    geo = nn([v for l in m.geo for v in l.trainable_variables])
    gate = nn(m.gate.trainable_variables)
    print(f"PARAMS total={tot:,}  geo(shared)={geo:,}({100*geo//tot}%)  gate={gate:,}", flush=True)
    esum = 0
    for e in m.experts:
        en = nn(e.trainable_variables); esum += en
        print(f"  expert {e.name:12s} {en:,} ({100*en//tot}%)", flush=True)
    print(f"  basis/mix(shared) {tot-geo-gate-esum:,}", flush=True)
    print("experts:", m.expert_names, flush=True)
    for fam in cfg.datasets:
        desc = np.array(SPEC[fam]["desc"], np.float32)[None]
        w = m.gate_weights(desc)
        s = "  ".join(f"{k}={float(v[0]):.3f}" for k, v in w.items())
        top = max(w.items(), key=lambda kv: float(kv[1][0]))[0]
        print(f"{fam:16s} desc={SPEC[fam]['desc']}  ->top:{top:11s} | {s}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "motion_tf.train.configs.prose_multi_mgpu")
