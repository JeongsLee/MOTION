"""arh5_byo — byl standard + OMEGA-DECODE velocity (physics-typed velocity for the incompressible trio).
Channel-decomposed pdea gap vs BCAT is ~90% VELOCITY (byl@90k: vel 17.3 vs BCAT 11.8; smoke already past
arh5-final): the remaining lever. Velocity increment = P[dW] with dW a learned scalar decoded from the same
latent trajectory (one extra no-bias Dense head + one irfft/frame): solenoidal by construction (error is 95%
solenoidal, measured), hard-IC exact, mean-flow (k=0) carried by the Eulerian delta. Applies to
incom (GT rel-div 0.0016, cleanest) / pdea / uncond via byl's mask; cfd auto-excluded (walls), com excluded
(compressible: divergence IS acoustics). Quantified target: velocity ~-3pp -> all-channel ~-1.5pp ->
pdea ~7.5-8.0 = BCAT parity zone on top of byl (~9.1).
  bash runners/_run_prose_stream_eu.sh motion_tf.train.configs.prose_150M_fluid5_6x_unified_task2_advb2_arh5_byo
"""
from .prose_150M_fluid5_6x_unified_task2_advb2_arh5_byl import Cfg as _Base


class Cfg(_Base):
    omega_decode = True      # solenoidal velocity increment via learned vorticity-potential
    save_dir = "results/prose_150M_fluid5_6x_unified_task2_advb2_arh5_byo"
