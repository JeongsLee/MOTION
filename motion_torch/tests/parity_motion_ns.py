"""TF <-> PyTorch parity for MOTION-NS (zsc, 21.8M; MOTION_WARP_LIVE=1 MOTION_BANK_NGENERIC=8
MOTION_BANK_GENERIC_MERGE=1 MOTION_BOX_UNIFORM=64 MOTION_INPUT_ZSCORE=1 MOTION_COND_STATS=1).

  ~/anaconda3/envs/torch/bin/python tests/parity_motion_ns.py [--ckpt PATH] [--real 8] [--sswarp 1]
Synthetic cases (all code paths, 64^2 / 64^3-box), held-out real units of the 6 joint-benchmark families from
LOCAL_UNITS (MOTION_SPLIT_MTF test split), and with --sswarp 1 a second configuration that exercises the
MOTION-IVP-S code paths with the same weights: MOTION_SS_WARP=blend single-shot (MOTION_FORCE_MODE=ss) and the
receding-horizon hop-1 rollout (MOTION_RH_FAMS)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parity_common import real_cases, run_parity, synthetic_cases  # noqa: E402

from motion_torch.config import PRESETS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=os.path.expanduser("checkpoints/zsc_ckpt_160000.npz"))
ap.add_argument("--real", type=int, default=8)
ap.add_argument("--sswarp", type=int, default=1)
a = ap.parse_args()
cfg = PRESETS["motion_ns"]
cases = synthetic_cases(cfg)
if a.real:
    cases += real_cases(cfg, ("incom_ns", "com_ns", "shallow_water", "pdearena_ns", "cfdbench"), n_units=a.real)
_, ok = run_parity("motion_ns", cfg, a.ckpt, cases)
if a.sswarp:
    # IVP-S code paths with the NS weights: every transient family single-shot with the blend transport carrier,
    # com_ns in receding-horizon mode (hop-1 AR of the single-shot operator)
    cfg2 = cfg.replace(name="motion_ns_sswarp", force_mode="ss", ss_warp="blend", rh_fams=("com_ns",))
    c2 = synthetic_cases(cfg2, fams=("incom_ns", "com_ns", "CE-RP"))
    if a.real:
        c2 += real_cases(cfg2, ("incom_ns", "com_ns"), n_units=2)
    _, ok2 = run_parity("motion_ns_sswarp", cfg2, a.ckpt, c2)
    ok = ok and ok2
sys.exit(0 if ok else 1)
