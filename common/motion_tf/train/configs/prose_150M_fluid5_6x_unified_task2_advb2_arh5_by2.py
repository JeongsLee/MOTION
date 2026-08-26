"""arh5_by2 — buoyancy bank v2: the three fixes over the dormant unified_buoyancy branch (arm 'by').
(1) per-trajectory magnitude beta = 1 + zero-init Dense(GAP(h_geom)) -- the conditioned pdearena set varies
the buoyancy coefficient per trajectory (ForcingExpert-style readout; missing in the branch);
(2) BOTH axis gradients of the scalar (vertical-axis convention insurance -- bitten twice by axis swaps);
(3) TIME-CONSISTENT source: evaluated on the adv_bank-transported scalar c(tau), not the stale anchored IC
(a static source inherits the transport-phase error it is meant to fix).
Residual = beta * tau * head([c_t, dr, dc]) -> velocity slots; tau-ramp => IC exact; +~1.2k params.
3-way: arh5 (no source) / by (branch, pointwise+static) / by2 (bank, inferred-magnitude+transported).
  bash runners/_run_prose_stream_eu.sh motion_tf.train.configs.prose_150M_fluid5_6x_unified_task2_advb2_arh5_by2
"""
from .prose_150M_fluid5_6x_unified_task2_advb2_arh5 import Cfg as _Base


class Cfg(_Base):
    buoyancy_bank = True     # model-level time-consistent Boussinesq source bank
    N_p = 32
    n_modes = 16
    save_dir = "results/prose_150M_fluid5_6x_unified_task2_advb2_arh5_by2"
    ckpt_every = 10000
    ckpt_keep = 1
    eval_every = 2000
