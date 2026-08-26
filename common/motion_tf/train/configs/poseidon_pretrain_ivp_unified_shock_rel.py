"""PHASE-2 IVP unified — COLD 320k with (1) shock-capturing expert ON and (2) task2-style rel-L2 loss.

Motivated by the phase2-vs-Poseidon channel comparison: our ONLY losses to Poseidon-B are SHOCK velocity
(CE-RP/CE-CRP Riemann families). phase2 was trained with unified_shock OFF; the Task-1 shock test couldn't
cover this (no Riemann families there). So enable the shock-capturing −∇·F̂ feature where it matters.

ALSO: train_ivp was hardcoding MSE, ignoring the config's loss_mse=False/loss_relclamp=4.0. Patched to honor
it → this run uses the task2-style CLAMPED rel-L2 loss (Phase-1 found rel-L2/instance-norm > global-MSE; the
physloss exp confirmed global-MSE hurt Phase-1). NOTE: two changes vs the current phase2 (shock + loss) →
attribution is joint, but both are the intended improvements.

Cold single cosine over 320k (lr 1e-4 / warmup 1000 inherited), no warm-start. Compare to the current phase2
(eu2b, no-shock + MSE, ckpt_276000) on the scOT metric — esp. CE-RP/CE-CRP velocity. 2×H100, _run_ivp_eu.sh."""
from .poseidon_pretrain_ivp_unified import Cfg as _Base


class Cfg(_Base):
    unified_shock = True       # shock-capturing flux feature (targets CE-RP/CE-CRP shock velocity)
    steps = 320000             # match the current phase2 scale
    seed = 3
    save_dir = "results/poseidon_pretrain_ivp_unified_shock_rel"
    # loss = clamped rel-L2 (loss_mse=False, loss_relclamp=4.0) inherited from poseidon_pretrain_ivp,
    # now ACTIVE via the patched train_ivp (was silently MSE before). cold → no init_from.
