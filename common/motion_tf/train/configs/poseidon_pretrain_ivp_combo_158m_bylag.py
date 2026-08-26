"""B-II typed-transport-decode arm: warm-start the converged 800k base and switch the slot-2 readout
from Eulerian-residual to the NATIVE Lagrangian splat (B-I's byl lever), with the IVP-safe slot-aliveness
gate (lag_gate_mode="slots" -- the B-I temporal-variance gate is identically 0 at T_in=1).
Target: NS-Sines/NS-Gauss carry a transported tracer in slot 2 and are 65% of the base's pretrain error;
NS-PwC (the in-family downstream task) has the same layout. ACE/Wave/Poisson lack velocity slots and are
structurally excluded by the gate. Judged on DOWNSTREAM TRANSFER, not on the pretraining metric.
"""
from .poseidon_pretrain_ivp_combo_158m_scratch import Cfg as _Base


class Cfg(_Base):
    lag_channel = True
    lag_native = True                # slot-2 output = splat projection (structural, not a residual)
    lag_stride = 1                   # 128^2 particles: bilinear splat = identity at tau=0 (hard-IC exact)
    lag_modes = 6
    lag_gate_mode = "slots"          # IVP gate (T_in=1 safe); "desc" would be identically 0 here
    init_from = "/eu/results/poseidon_pretrain_ivp_combo_158m_scratch/ckpt_800000_named.npz"   # NAMED fmt: tolerates the new lag_* vars
    lr = _Base.lr * 0.5              # half peak, fresh cosine from the converged base
    warmup = 2000
    steps = 150000
    eval_every = 5000
    ckpt_every = 10000
    save_dir = "results/poseidon_pretrain_ivp_combo_158m_bylag"
