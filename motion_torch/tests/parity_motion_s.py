"""TF <-> PyTorch parity for MOTION-S (paper m1f, 20.7M; MOTION_LEGACY_VOCAB=1 MOTION_VORTEX2D_LEGACY=1 NO_DEAD_REPAIR=1).

  ~/anaconda3/envs/torch/bin/python tests/parity_motion_s.py [--ckpt PATH] [--quick 1] [--real 8]
Synthetic 2-D AR (incom_ns, CE-RP with transport), 2-D single-shot (ACE), 3-D AR (pdebench3d M0.1), steady
(Poisson-Gauss), plus held-out real units (incom_ns, com_ns, shallow_water, pdearena_ns) from LOCAL_UNITS."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parity_common import real_cases, run_parity, synthetic_cases  # noqa: E402,F401

from motion_torch.config import PRESETS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=os.path.expanduser("checkpoints/m1f_ckpt_best.npz"))
ap.add_argument("--quick", type=int, default=0)
ap.add_argument("--real", type=int, default=8, help="real units per family (0 = synthetic only)")
a = ap.parse_args()
cfg = PRESETS["motion_s"]
cases = synthetic_cases(cfg)
if a.quick:
    cases = cases[:1]
elif a.real:
    cases += real_cases(cfg, ("incom_ns", "com_ns", "shallow_water", "pdearena_ns"), n_units=a.real)
_, ok = run_parity("motion_s" + ("_quick" if a.quick else ""), cfg, a.ckpt, cases)
sys.exit(0 if ok else 1)
