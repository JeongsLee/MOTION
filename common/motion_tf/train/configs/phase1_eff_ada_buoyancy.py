"""Phase-1 ADA + BUOYANCY operator, trained FROM SCRATCH under the EXACT phase1_eff_ada figure protocol
(6 families, eff-batch 32, 1M views, same lr/warmup/eval/metric/milestones) — the ONLY difference from the
4.43% baseline run is the buoyancy branch ON. Clean A/B: does the scalar->velocity (Boussinesq) source
operator improve the joint curve and the buoyancy-driven family (pdearena_ns, esp. Vy)?
Launch (2 GPU): bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_buoyancy
"""
import os
from .phase1_eff_ada import Cfg as _Base


class Cfg(_Base):
    unified_buoyancy = True
    unified_buoyancy_hidden = int(os.environ.get("BY_HIDDEN", "32"))
    save_dir = "results/" + os.environ.get("BY_SAVE", "phase1_int_ada_152m_6fam_buoyancy")
    # NO init_ckpt → cold from-scratch. datasets / lr / warmup / steps / eval_every / ckpt_milestones all
    # inherited from phase1_eff_ada (identical to the figure run).
