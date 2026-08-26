"""Phase-1 ADA — RECURSIVE baseline + 2nd-order (AB2) z-integration. The recursive baseline advances the state
by forward Euler (z_{i+1}=z_i+W_i·dt, 1st order). This replaces it with the explicit 2nd-order Adams-Bashforth
(z_{i+1}=z_i+dt·(1.5·W_i − 0.5·W_{i-1})) — the sequential-rollout analog of trapezoidal (true trapezoidal needs
the unavailable W_{i+1}; AB2 uses the carried previous W, 2nd-order, NO extra bank eval, i=0 → Euler). Tests
whether a more accurate integration of the recursive state improves accuracy over plain Euler. Clean A/B vs the
recursive baseline (job-q6029b64qnt9, 4.43%) — ONLY the integration order differs.
Launch (2 GPU): bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_recursive_ab2
"""
import os
from .phase1_eff_ada import Cfg as _Base


class Cfg(_Base):
    recursive_2nd = True       # AB2 (2nd-order) z-update instead of forward Euler
    save_dir = "results/" + os.environ.get("AB2_SAVE", "phase1_int_ada_152m_6fam_recursive_ab2")
