"""arh5_lag — forward-Lagrangian particle transport bank v1 (user idea 07-22). The Eulerian per-pixel
basis has the wrong inductive bias for advection (fixed grid point sees oscillatory signal as filaments
pass; measured: phase-coherence 0.58 vs BCAT 0.79 on pdearena_ns). In Lagrangian coordinates transport
vanishes: the carried scalar is ~constant along paths (Dc/Dt=0) and the particle trajectory x(t),y(t) is
SMOOTH — exactly what the ADA continuous-time basis represents best. v1: particles on a stride-2 grid carry
the passive-scalar slot (slot 2: pdea/uncond smoke, incom particles) along basis trajectories
Delta(tau) = alpha*u0*tau + sum A_m Phi_m(tau) (Phi(0)=0 hard-IC), rendered by a differentiable normalized
Gaussian splat (forward scatter — vs adv_bank's backward gather), blended by a per-family descriptor gate
(const-0.02 init: live grads, ~identity start). +~10k params. Base = arh5 (2-seg AR, lf4), N_p32/n_modes16
(same arm protocol as ld2/tcu). From-scratch matched 160k. Baseline = arh5 (3.50 / pdea 10.27 / unc 6.97).
  bash runners/_run_prose_stream_eu.sh motion_tf.train.configs.prose_150M_fluid5_6x_unified_task2_advb2_arh5_lag
"""
from .prose_150M_fluid5_6x_unified_task2_advb2_arh5 import Cfg as _Base


class Cfg(_Base):
    lag_channel = True       # forward-Lagrangian particle transport bank (v1, scalar slot)
    lag_stride = 2           # 64^2 = 4096 particles
    lag_modes = 6            # hard-IC temporal basis modes for particle trajectories
    lag_sigma = 1.2          # Gaussian splat width (px)
    N_p = 32
    n_modes = 16
    save_dir = "results/prose_150M_fluid5_6x_unified_task2_advb2_arh5_lag"
    ckpt_every = 10000
    ckpt_keep = 1
    eval_every = 2000
