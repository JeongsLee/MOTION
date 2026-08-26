"""Phase-1 ADA — pw_batched (fast parallel-W) + CFDBENCH_GEOM. Tests the hypothesis that CFDBench's static
boundary mask, when packed into the SHARED scalar slot 2, pollutes that channel for pdearena_ns/incom_ns
(which use slot 2 for smoke/particles) — CFDBench converges fast (strong gradients) so it can dominate the
shared slot-2 representation. Moving the boundary OUT of slot 2 into the model's geometry mask (Π freeze-solid
+ encoder geom input) leaves slot 2 a CLEAN passive-transport scalar across all families. Loss/metric
unchanged (cfdbench c_mask slot2 was already 0). Same fast pw_batched rollout. Clean A/B vs pw_batched (no
cfdgeom) to isolate the slot-2-cleanup effect on pdearena_ns.
Launch (2 GPU): bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwbatched_cfdgeom
"""
import os
from .phase1_eff_ada import Cfg as _Base


class Cfg(_Base):
    parallel_w = True
    pw_batched = True        # fast batched panel rollout (~0.98 s/step)
    step_embed = True
    w_feedback = False
    swap_memory = False
    cfdbench_geom = True     # CFDBench boundary (native ch2) -> geom_mask; cfdbench field slots -> [0,1]
    save_dir = "results/" + os.environ.get("CFDG_SAVE", "phase1_int_ada_152m_6fam_pwb_cfdgeom")
