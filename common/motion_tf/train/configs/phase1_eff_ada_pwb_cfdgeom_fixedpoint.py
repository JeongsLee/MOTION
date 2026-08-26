"""Phase-1 ADA — cfdgeom + FIXED-POINT bank re-evaluation (the parallel recursion). The operator bank is
re-evaluated on the W-RECONSTRUCTED evolving state z(t_i)=z0+Σ_{j<i}W_j·dt over K parallel iterations, so the
advection −(u·∇)z sees the EVOLVED velocity (the recursion's PDEArena gain) WITHOUT the sequential Euler loop
— the self-consistency W=bank(z(W)) is solved by Picard iteration, all panels in parallel each sweep. Targets
the ROOT cause of the no-recursion PDEArena gap (frozen IC advection), unlike panel_attn (recombines frozen-
bank outputs) or pure_upsample (high-freq). Cost ≈ (K+1)× pw_batched. Stacks on cfdgeom; A/B vs cfdgeom.
Launch (2 GPU): bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwb_cfdgeom_fixedpoint
"""
import os
from .phase1_eff_ada_pwbatched_cfdgeom import Cfg as _Base


class Cfg(_Base):
    fixedpoint_iters = 2       # K Picard sweeps (3 bank evals); bank re-evaluated on the W-evolved state
    save_dir = "results/" + os.environ.get("FP_SAVE", "phase1_int_ada_152m_6fam_pwb_cfdgeom_fixedpoint")
