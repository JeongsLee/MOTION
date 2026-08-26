"""158M NS-PwC/64 finetune that MATCHES the 158M pretraining channel representation: uv = velocity task,
dummies SUPERVISED (pad_zero: density->rho_const=1, pressure/energy/geom/forcing->0) instead of masked. The
plain uv-only finetune re-MASKS the dummies the 158M learned to output -> discards its dummy-supervised
valid-channel structure (and stalled ~7.7%). Keeping pad_zero preserves that structure. Use the 2-term
per-channel loss (loss_perch) so the supervised dummies do NOT dilute the velocity objective: uv gets a
per-channel relative-L1 (clean), dummies get a small absolute-L1 to their constant. Final uv vs the old
114M-base 6.58% / Poseidon-B 3.67% measured by _eval_scot_metric (uv=slots 0,1)."""
import os
from .poseidon_finetune_158m import Cfg as _Base

_N = int(os.environ.get("FT_NSHOT", "64"))
_STEP = os.environ.get("FT_INIT_STEP", "145000")


class Cfg(_Base):
    uv_only = True                # velocity is the task channel
    pad_zero = True               # supervise the dummies (match 158M pretraining), NOT mask them
    rho_const = 1.0               # density -> 1 (as in 158M pretrain)
    loss_perch = True             # 2-term: uv per-channel rel-L1 + dummy abs-L1 -> uv undiluted
    save_dir = f"results/ft158m_NS-PwC_N{_N}_uvpad_perch_s{_STEP}"
