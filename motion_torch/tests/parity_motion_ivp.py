"""TF <-> PyTorch parity for the pretrained MOTION-IVP checkpoint (stage-3 option C, motion_ivps_gn3w/ckpt_best,
md5 3b780f3b; revision R1, 2026-11-09). Configuration of the FT15 / scoring runs (run_ft15_motion_v4.sh,
score_ivp_l1.py): t_in 10 window with one observed frame (MOTION_TI_FIXED=1), n_p 32, global normalisation,
single-shot with the blend transport carrier, NS-Sines in receding-horizon hop-1 mode; synthetic samples of
Poseidon pretraining families (statistics for the global normalisation from the synthetic samples themselves).

  python tests/parity_motion_ivp.py [--ckpt PATH]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parity_common import run_parity, synthetic_cases  # noqa: E402

from motion_torch.config import PRESETS  # noqa: E402
from motion_torch.data.example import compute_global_norm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/motion_ivp_ckpt_best.npz")
ap.add_argument("--fams", default="CE-RP,CE-KH,NS-Sines,NS-Gauss")
a = ap.parse_args()
fams = tuple(a.fams.split(","))
cfg = PRESETS["motion_ivp_s"].replace(name="motion_ivp_s_c", t_in=10, ti_fixed=1, rh_fams=("NS-Sines", "NS-Gauss"))
cases = synthetic_cases(cfg, fams=fams)
for c in cases:
    gn = compute_global_norm(c["samples"])[:2]
    c["global_norm"] = {c["samples"][0]["family"]: tuple(np.asarray(v, np.float32) for v in gn)}
_, ok = run_parity("motion_ivp", cfg, a.ckpt, cases)
sys.exit(0 if ok else 1)
