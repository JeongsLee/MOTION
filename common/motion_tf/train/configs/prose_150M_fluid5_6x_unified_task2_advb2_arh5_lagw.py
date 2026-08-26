"""arh5_lagw — v2 of the Lagrangian transport bank: omega-carrying particles + FFT-Poisson velocity.
v1 (arh5_lag) moves only the passive-scalar slot; but the pdea error is velocity-dominated (15.96% vs
10.27% all-ch) and velocity is NOT conserved along paths (pressure gradient). In 2D incompressible flow
VORTICITY IS materially conserved (Dw/Dt = nu lap w + forcing, smooth along paths) -> particles carry w
along the SAME basis trajectories, splat w(tau), invert exactly on the periodic grid by FFT-Poisson
(lap psi = -w, u = (dpsi/dy, -dpsi/dx); one rfft2d per frame, differentiable) -> the VELOCITY itself
becomes Lagrangian. pdea/uncond are exactly periodic+incompressible = the two weak families. Motion-residual
form ==0 at tau=0 (hard-IC); per-family gates (const 0.02). Base = arh5_lag (scalar v1 kept, shared
trajectories). From-scratch matched 160k, N_p32 arm protocol.
  bash runners/_run_prose_stream_eu.sh motion_tf.train.configs.prose_150M_fluid5_6x_unified_task2_advb2_arh5_lagw
"""
from .prose_150M_fluid5_6x_unified_task2_advb2_arh5_lag import Cfg as _Base


class Cfg(_Base):
    lag_omega = True         # v2: omega particles + FFT-Poisson -> Lagrangian velocity residual
    save_dir = "results/prose_150M_fluid5_6x_unified_task2_advb2_arh5_lagw"
