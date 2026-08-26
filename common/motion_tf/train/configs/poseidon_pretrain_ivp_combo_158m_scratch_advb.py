"""162M combo RE-PRETRAIN with the loose transport bank (adv_bank) — the Benchmark-I from-scratch A/B
showed the time-modulated semi-Lagrangian vocabulary dominating every family's curve from the start
(pdearena −12~14pp at matched views); this rerun unifies the architecture across both benchmarks so the
paper's proposed model = transformer encoder + tendency heads + transport branch + ADA everywhere.
Arch = combo_158m_scratch UNCHANGED + adv_bank (av_* fresh, gate zero-init → strictly additive).
Protocol identical to the original scratch run (2xH100, batch8 -> eff16, 800k ceiling, all-IC ic_reuse);
compare the inline eval curve against the original run's ckpt evals at matched steps and early-stop when
the signal is decisive.
Launch:  bash _run_ivp_eu.sh motion_tf.train.configs.poseidon_pretrain_ivp_combo_158m_scratch_advb
"""
from .poseidon_pretrain_ivp_combo_158m_scratch import Cfg as _Base


class Cfg(_Base):
    adv_bank = True
    adv_bank_M = 4
    adv_bank_K = 4
    save_dir = "/eu/results/poseidon_pretrain_ivp_combo_158m_scratch_advb"
    ckpt_every = 10000
    # v3 stabilizers (clamped rel-L1 pretrain diverged with live glorot V): small-amplitude velocity init,
    # zero time-mixture at start (bootstraps via the V0 branch), tighter displacement clip.
    adv_bank_vstd = 0.02
    adv_bank_tw0 = True
    adv_bank_clip = 2.0
