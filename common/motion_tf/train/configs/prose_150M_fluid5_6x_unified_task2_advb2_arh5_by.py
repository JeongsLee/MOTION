"""arh5_by — the DORMANT buoyancy branch, switched ON (config flag only; zero new code).
Diagnosis chain (07-23): pdea error is velocity-dominated (15.96%), 95% solenoidal (continuity ruled out,
Leray hurts), and a zero-training VIC baseline shows pure omega-transport FAILS (90.8% vel err) because
NS-cond vorticity has a first-order BUOYANCY SOURCE (Dw/Dt != 0). The UnifiedExpert already contains a
purpose-built Boussinesq branch (unified.py: scalar+dy -> velocity tendency, latent-space, zero-init,
"targets buoyancy-driven families (conditioned NS)") that has never been enabled. This arm = arh5 + that
flag. N_p32/n_modes16 arm protocol. From-scratch matched 160k. Baseline = arh5 (3.50/10.27/6.97).
  bash runners/_run_prose_stream_eu.sh motion_tf.train.configs.prose_150M_fluid5_6x_unified_task2_advb2_arh5_by
"""
from .prose_150M_fluid5_6x_unified_task2_advb2_arh5 import Cfg as _Base


class Cfg(_Base):
    unified_buoyancy = True   # Boussinesq scalar->velocity source branch (existing, zero-init)
    N_p = 32
    n_modes = 16
    save_dir = "results/prose_150M_fluid5_6x_unified_task2_advb2_arh5_by"
    ckpt_every = 10000
    ckpt_keep = 1
    eval_every = 2000
