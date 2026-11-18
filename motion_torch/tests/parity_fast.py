"""Accelerated trunk (fast.py) against the reference port on the released MOTION-NS weights, real units, CPU.
  python motion_torch/tests/parity_fast.py [ckpt] [units.pkl]"""
import os, sys, copy, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch
from motion_torch import PRESETS, build_model
from motion_torch.data.example import load_local_units, make_example_ar, stack_batch, box_dims_for
from motion_torch.inference import to_torch, predict, trajectory_error
from motion_torch.fast import accelerate
ck = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("checkpoints/zsc_ckpt_160000.npz")
up = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("checkpoints/units_nss_mtf.pkl")
torch.set_grad_enabled(False)
cfg = PRESETS["motion_ns"]
ref = build_model(cfg, ck, device="cpu").eval()
fast = copy.deepcopy(ref); print("accelerated modules:", accelerate(fast))
units = load_local_units(up)
worst = {}
for fam in units:
    for i, u in enumerate(units[fam][:2]):
        ex = make_example_ar(u, 10, 10, 4, 2048, 8192, np.random.default_rng(i), cfg, box_dims=box_dims_for(fam, cfg))
        b = to_torch(stack_batch([ex]), "cpu")
        t0 = time.time(); o1 = predict(ref, b); t1 = time.time(); o2 = predict(fast, b); t2 = time.time()
        p1, p2 = o1["pred"], o2["pred"]
        r1 = float((p1[:, 0] - p2[:, 0]).norm() / p1[:, 0].norm()); r10 = float((p1 - p2).norm() / p1.norm())
        e1, e2 = trajectory_error(o1), trajectory_error(o2)
        de = float(np.max(np.abs(np.asarray(e1) - np.asarray(e2))))
        worst[fam] = max(worst.get(fam, (0, 0, 0)), (r1, r10, de))
        print(f"{fam:18s} u{i} step1 {r1:.2e} 10-step {r10:.2e} traj-err diff {de:.2e} | cpu ref {t1-t0:.1f}s fast {t2-t1:.1f}s")
print("MAX", max(v[0] for v in worst.values()), max(v[1] for v in worst.values()), max(v[2] for v in worst.values()))
