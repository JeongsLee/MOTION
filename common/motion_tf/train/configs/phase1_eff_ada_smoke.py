"""SMOKE of the Phase-1 integrated ADA config — verify 6-family stream (incl uncond), eval, ckpt, tee, metric
run end-to-end in a couple minutes. Tiny: 40 steps, eval@20, ckpt@{20,40}, grad_accum 2."""
from .phase1_eff_ada import Cfg as _Base


class Cfg(_Base):
    grad_accum = 2
    steps = 40
    eval_every = 20
    ckpt_milestones = [20, 40]
    save_dir = "results/phase1_int_ada_SMOKE"
