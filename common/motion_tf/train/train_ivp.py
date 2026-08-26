"""PHASE-2 IVP training entry — SEPARATE from train_prose_mgpu.py (Phase-1 is untouched; both run).
Pretrains PhysicsOperatorMixture from a single IC snapshot (T_in=1) on Poseidon's pretraining set, with
all2all IC-sampling and masked supervision over the future frames, evaluating with Poseidon's metric
(MEDIAN relative L1 at the FINAL time, physical space). Run:  python -u -m motion_tf.train.train_ivp <config>
"""
from __future__ import annotations
import glob, importlib, json, os, sys
import numpy as np
import tensorflow as tf

from ..model import PhysicsOperatorMixture
from ..data import poseidon
from ..utils import ckpt as ckptlib


# Solution-channel slots per family (mirror of _eval_scot_metric.GROUPS, flattened).
# The inline eval flattens ALL c_mask-supervised channels into one rel-L1; for families
# with a large fed-but-not-physically-scored channel (Wave-Layer's c = slot6, a near-const
# coefficient field copied from input) that channel's huge norm DOMINATES the denominator and
# DILUTES the real solution error to ~0 (Wave looked like 0.6% when slot2 u was actually ~40%).
# Restrict the inline metric to the genuine solution slots so it matches the offline
# slot-specific eval. Families absent here fall back to the full c_mask (no dilutant → identical).
SOLUTION_SLOTS = {
    "CE-RP":      [0, 1, 3, 4], "CE-KH":   [0, 1, 3, 4],
    "CE-CRP":     [0, 1, 3, 4], "CE-Gauss":[0, 1, 3, 4], "CE-RM": [0, 1, 3, 4],
    "NS-Sines":   [0, 1, 2],    "NS-Gauss":[0, 1, 2],    "NS-PwC":[0, 1, 2],
    "ACE":        [2],          "Wave-Layer": [2],   # slot6=c is fed-not-scored → excluded
    "Poisson-Gauss": [2],                            # steady elliptic: solution slot only (slot7=source fed-not-scored)
}


def masked_mse_frames(pred, tgt, cm, frame_mask):
    """pred,tgt:(B,Nt,H,W,C)  cm:(B,C)  frame_mask:(Nt,) — MSE over active channels × valid frames."""
    m = cm[:, None, None, None, :] * frame_mask[None, :, None, None, None]
    se = ((pred - tgt) ** 2) * m
    return tf.reduce_sum(se) / (tf.reduce_sum(m * tf.ones_like(se)) + 1e-8)


def masked_relclamp_frames(pred, tgt, cm, frame_mask, cap=4.0, eps=1e-6):
    """task2-style CLAMPED relative-L2 (Phase-1 prose loss) for the IVP trainer: per-sample per-frame
    num/den, clamped to cap, masked to active channels × valid frames. (Phase-1 found rel-L2/instance-norm
    > global-MSE; phase2 had used MSE — this ports the task2 loss style.)"""
    cmask = cm[:, None, None, None, :]                                   # (B,1,1,1,C)
    num = tf.reduce_sum(((pred - tgt) ** 2) * cmask, axis=[2, 3, 4])      # (B,Nt)
    den = tf.reduce_sum((tgt ** 2) * cmask, axis=[2, 3, 4]) + eps
    ratio = tf.minimum(num / den, float(cap))                            # clamp small-norm blowup
    fm = frame_mask[None, :]                                             # (1,Nt) valid-frame mask
    return tf.reduce_sum(ratio * fm) / (tf.reduce_sum(tf.ones_like(ratio) * fm) + eps)


def masked_rell1_frames(pred, tgt, cm, frame_mask, cap=4.0, eps=1e-6):
    """RELATIVE-L1 (Poseidon's loss family + our eval metric): per-sample per-frame sum|pred-tgt|/sum|tgt|,
    clamped, masked. Aligns the training loss with the comparison metric (median rel-L1 @final)."""
    cmask = cm[:, None, None, None, :]
    num = tf.reduce_sum(tf.abs(pred - tgt) * cmask, axis=[2, 3, 4])       # (B,Nt)
    den = tf.reduce_sum(tf.abs(tgt) * cmask, axis=[2, 3, 4]) + eps
    ratio = tf.minimum(num / den, float(cap))
    fm = frame_mask[None, :]
    return tf.reduce_sum(ratio * fm) / (tf.reduce_sum(tf.ones_like(ratio) * fm) + eps)


def masked_rell1_perch(pred, tgt, cm, frame_mask, cap=4.0, eps=1e-6, dummy_w=0.1):
    """TWO-TERM loss decoupling the velocity TASK from the const-supervised DUMMY channels, so neither the
    channels-joined denominator (const=1 inflates it -> dilutes uv) nor the channels-joined numerator
    (garbage on a 0-target inflates it -> crowds out uv) can distort the uv objective:
      TASK (velocity, slots 0,1): per-channel RELATIVE-L1 (Poseidon per-channel-slice norm) -> uv PRISTINE,
        identical whether the dummy const is 0 or 1.
      DUMMY (other supervised slots, pad const 0 or 1): ABSOLUTE-L1 to target (relative is undefined at 0),
        small weight -> keeps the channel VALID/alive without touching the uv loss.
    (uv_only finetune: TASK is always velocity = slots 0,1.)"""
    C = tf.shape(pred)[-1]
    task = tf.concat([tf.ones([2]), tf.zeros([C - 2])], 0)[None, :]      # (1,C) velocity = slots 0,1
    tmask = cm * task                                                    # task & active
    dmask = cm * (1.0 - task)                                            # dummy (pad const) supervised slots
    fm = frame_mask[None, :, None]                                       # (1,Nt,1)
    num = tf.reduce_sum(tf.abs(pred - tgt), axis=[2, 3])                 # (B,Nt,C) per-channel L1
    den = tf.reduce_sum(tf.abs(tgt), axis=[2, 3]) + eps                  # (B,Nt,C)
    ratio = tf.minimum(num / den, float(cap))                           # (B,Nt,C)
    wt = tmask[:, None, :] * fm
    task_loss = tf.reduce_sum(ratio * wt) / (tf.reduce_sum(wt) + eps)
    abser = tf.reduce_mean(tf.abs(pred - tgt), axis=[2, 3])             # (B,Nt,C) mean|.| (abs, defined at 0)
    wd = dmask[:, None, :] * fm
    dummy_loss = tf.reduce_sum(abser * wd) / (tf.reduce_sum(wd) + eps)
    return task_loss + float(dummy_w) * dummy_loss


def main(cfg_path):
    cfg = importlib.import_module(cfg_path).Cfg
    if bool(getattr(cfg, "bf16", False)):
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
        print("mixed_bfloat16 policy enabled", flush=True)
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.ReductionToOneDevice())
    nrep = strategy.num_replicas_in_sync
    GLOBAL = cfg.batch * nrep
    Nt = cfg.Nt
    # nt_data: TEMPORAL PADDING — the model synthesizes Nt output frames on a basis span extended past the
    # data horizon (cfg.T_final scaled accordingly), so the data-final frame sits INSIDE the ADA coverage
    # instead of at the basis endpoint (Fourier wrap / Legendre max-oscillation live there). Data has only
    # NTD frames; supervision and eval index data with NTD, the model with Nt (>NTD → trailing pad frames
    # are unsupervised).
    NTD = int(getattr(cfg, "nt_data", 0)) or Nt
    jit = bool(getattr(cfg, "jit_compile", True))
    max_ic = int(round(float(getattr(cfg, "max_ic_frac", 0.5)) * (NTD - 1))) if getattr(cfg, "all2all", True) else 0
    print(f"=== IVP {cfg.datasets} | replicas={nrep} global_batch={GLOBAL} Nt={Nt} max_ic={max_ic} ===", flush=True)

    ds, test = poseidon.build_poseidon_stream(cfg, GLOBAL)
    fam_names = test["fam_names"]
    dist_ds = strategy.experimental_distribute_dataset(ds)

    with strategy.scope():
        model = PhysicsOperatorMixture(cfg)
        _wu = int(getattr(cfg, "warmup", 0))
        def _mklr(peak):
            if getattr(cfg, "lr_decay", False):
                return (tf.keras.optimizers.schedules.CosineDecay(
                            peak * 0.02, cfg.steps, alpha=0.05, warmup_target=peak, warmup_steps=_wu)
                        if _wu > 0 else tf.keras.optimizers.schedules.CosineDecay(peak, cfg.steps, alpha=0.05))
            return peak
        _cn = float(getattr(cfg, "clipnorm", 1.0))
        _lr_dec = float(getattr(cfg, "lr_decoder", 0.0))   # >0 => Poseidon-style discriminative LR
        DISC = _lr_dec > 0
        opt = tf.keras.optimizers.Adam(_mklr(cfg.lr), clipnorm=_cn)              # backbone (low LR)
        opt_dec = tf.keras.optimizers.Adam(_mklr(_lr_dec), clipnorm=_cn) if DISC else None  # decoder/head (high LR)
    npar = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    print(f"model params: {npar:,}", flush=True)
    # DECODER-ONLY transfer: restrict the optimized variables to the latent decoder (decode_h/decode_out);
    # everything else (physics-op bank, transformer encoder, W head, upsample) stays FROZEN at the warm-
    # started pretrained values. = the frozen-backbone / decoder-only probe ("how good is the rep?").
    _dkeys = tuple(getattr(cfg, "decoder_keys", ("decode_h", "decode_out", "w_up", "shock_scale")))
    N_DEC = 0
    if bool(getattr(cfg, "decoder_only", False)):
        TVARS = [v for v in model.trainable_variables if any(k in v.name for k in _dkeys)]
        print(f"DECODER-ONLY: optimizing {len(TVARS)}/{len(model.trainable_variables)} vars "
              f"({sum(int(np.prod(v.shape)) for v in TVARS):,} params) -> {[v.name for v in TVARS]}", flush=True)
    elif DISC:
        # Poseidon-style discriminative LR: decoder/head (high LR) + backbone (low LR), nothing frozen.
        DEC_VARS = [v for v in model.trainable_variables if any(k in v.name for k in _dkeys)]
        _ds = set(id(v) for v in DEC_VARS)
        BB_VARS = [v for v in model.trainable_variables if id(v) not in _ds]
        TVARS = DEC_VARS + BB_VARS; N_DEC = len(DEC_VARS)
        print(f"DISCRIMINATIVE LR: decoder/head {N_DEC} vars @ lr={_lr_dec:.1e} + backbone {len(BB_VARS)} vars @ lr={cfg.lr:.1e}", flush=True)
    else:
        TVARS = model.trainable_variables
    lam = float(getattr(cfg, "lambda_gate", 0.0))
    _dummy_desc = tf.zeros((GLOBAL, int(getattr(cfg, "desc_dim", cfg.n_channels))), tf.float32)
    _dummy_cf = tf.zeros((GLOBAL, 4), tf.float32)

    # v0-slot injection (2-frame IC without an arch change): the wave equation's state is (u, du/dt) but
    # the data exposes only u — inject the OBSERVED time derivative v0 = u_i − u_{i−1} (slot v0_from_slot)
    # into an unused dummy slot (v0_to_slot) of the SAME 8-slot input, so the channel count (and the
    # positional ckpt load) is unchanged while the missing half of the Cauchy data becomes visible.
    # Pairs are anchored at i≥1 (i=0 has no history); eval anchors at frame 1 and scores frames 2..Nt−1.
    V0D = int(getattr(cfg, "v0_to_slot", -1)); V0S = int(getattr(cfg, "v0_from_slot", 2))
    TIN2 = V0D >= 0
    if TIN2:
        print(f"  V0-SLOT: input slot{V0D} <- d/dt of slot{V0S} (2-frame IC), pairs anchored at i>=1", flush=True)

    @tf.function(jit_compile=jit)
    def compute_grads(traj, cm, i):
        # all2all: IC = frame i (single snapshot); supervise model output frames k vs data frame i+k.
        if TIN2 and not bool(getattr(cfg, "tin2_anchor0", False)):
            i = tf.maximum(i, 1)                                      # need one history frame
        u0 = traj[:, i]                                               # (B,H,W,C)
        if TIN2:
            # tin2_anchor0: include i=0 pairs with v0=0 — the TRUE zero-velocity IC (closes the anchor-0
            # coverage hole the i>=1 restriction left: eval at anchor 0 was out-of-training-distribution).
            v0 = traj[:, i, :, :, V0S] - traj[:, tf.maximum(i - 1, 0), :, :, V0S]  # observed du/dt (0 at i=0)
            u0 = tf.concat([u0[..., :V0D], v0[..., None], u0[..., V0D + 1:]], axis=-1)
        kidx = tf.minimum(i + tf.range(Nt), NTD - 1)                  # (Nt,) data-frame for each output frame
        tgt = tf.gather(traj, kidx, axis=1)                           # (B,Nt,H,W,C)
        frame_mask = tf.cast(i + tf.range(Nt) <= (NTD - 1), tf.float32)  # (Nt,) pad frames unsupervised
        if float(getattr(cfg, "lead_weight_pow", 0.0)) > 0.0:
            # PANEL-GRADIENT EQUALIZATION: the ADA integral routes a lead-k pair's gradient into panels
            # W_1..W_k only, so panel W_m is supervised with prob ∝ P(lead ≥ m) — late panels starve and
            # drift via weight coupling. Reweight frame k's loss by k^p to rebalance the per-panel signal.
            _lw = tf.pow(tf.cast(tf.range(Nt), tf.float32), float(cfg.lead_weight_pow))
            frame_mask = frame_mask * _lw / (tf.reduce_mean(_lw) + 1e-8)
        if bool(getattr(cfg, "pair_loss", False)):
            # Poseidon-style PAIR diet: supervise ONE random lead per step (vs the all-frame joint loss).
            # Hypothesis probe: the joint loss may force a cross-lead compromise (hedged/smeared frames);
            # sparse per-lead gradients let each lead be fit on its own terms across steps. The forward
            # still predicts all frames — only the gradient focus changes (40 pair-gradients/step, matched
            # to scOT's finetune diet).
            jmax = tf.maximum(Nt - 1 - i, 1)
            u = tf.random.uniform([])
            j = 1 + tf.minimum(tf.cast(u * tf.cast(jmax, tf.float32), tf.int32), jmax - 1)
            frame_mask = tf.one_hot(j, Nt, dtype=tf.float32)

        with tf.GradientTape() as tape:
            pred, alpha = model.call_with_gate(u0, _dummy_desc[: tf.shape(u0)[0]],
                                               _dummy_cf[: tf.shape(u0)[0]], training=True)
            pred = tf.cast(pred, tf.float32)
            if float(getattr(cfg, "loss_rell1", 0.0)) > 0 and bool(getattr(cfg, "loss_perch", False)):
                loss = masked_rell1_perch(pred, tgt, cm, frame_mask, cap=float(cfg.loss_rell1),
                                          dummy_w=float(getattr(cfg, "dummy_w", 0.1)))
            elif float(getattr(cfg, "loss_rell1", 0.0)) > 0:
                loss = masked_rell1_frames(pred, tgt, cm, frame_mask, cap=float(cfg.loss_rell1))
            elif float(getattr(cfg, "loss_relclamp", 0.0)) > 0 and not bool(getattr(cfg, "loss_mse", True)):
                loss = masked_relclamp_frames(pred, tgt, cm, frame_mask, cap=float(cfg.loss_relclamp))
            else:
                loss = masked_mse_frames(pred, tgt, cm, frame_mask)
            reg = lam * tf.cast(tf.reduce_mean(tf.reduce_sum(alpha, -1)), tf.float32)
            total = loss + reg
        grads = tape.gradient(total, TVARS)
        return grads, loss

    @tf.function
    def step_fn(traj, cm, i):
        grads, loss = compute_grads(traj, cm, i)
        # NaN-SKIP: a rare high-velocity batch can overflow the explicit advection rollout → NaN grads
        # (signature: healthy loss → instant NaN, not a gradient climb). Zero ALL grads when any is
        # non-finite so apply_gradients is a no-op → weights HELD at last-good, the bad batch is skipped
        # (vs diverging). Weights never receive NaN, so the next good batch recovers.
        finite = tf.reduce_all([tf.reduce_all(tf.math.is_finite(g)) for g in grads if g is not None])
        grads = [None if g is None else tf.where(finite, g, tf.zeros_like(g)) for g in grads]
        if DISC:                                          # decoder/head @ opt_dec, backbone @ opt
            opt_dec.apply_gradients(zip(grads[:N_DEC], TVARS[:N_DEC]))
            opt.apply_gradients(zip(grads[N_DEC:], TVARS[N_DEC:]))
        else:
            opt.apply_gradients(zip(grads, TVARS))
        return loss

    @tf.function
    def dist_step(traj, cm, i):
        pl = strategy.run(step_fn, args=(traj, cm, i))
        return strategy.reduce(tf.distribute.ReduceOp.MEAN, pl, axis=None)

    # GRADIENT ACCUMULATION: effective batch = batch * grad_accum (micro-batch fits memory; batch40 OOMs
    # because the model predicts all Nt frames at once). cfg.steps counts OPTIMIZER UPDATES.
    ACC = int(getattr(cfg, "grad_accum", 1))
    if ACC > 1:
        with strategy.scope():
            accum = [tf.Variable(tf.zeros_like(v), trainable=False) for v in TVARS]
        # NOTE: NO jit_compile on this outer fn — it wraps strategy.run, and XLA-compiling a strategy.run
        # at the outer level triggers the MirroredStrategy cross-device variable bug (geo0/kernel/replica_1
        # accessed from GPU:0). Mirror dist_step's pattern: jit lives only on the inner compute_grads.
        @tf.function
        def accum_step(traj, cm, i):
            def _f(traj, cm, i):
                grads, loss = compute_grads(traj, cm, i)
                for a, g in zip(accum, grads):
                    if g is not None:
                        a.assign_add(g)
                return loss
            pl = strategy.run(_f, args=(traj, cm, i))
            return strategy.reduce(tf.distribute.ReduceOp.MEAN, pl, axis=None)
        @tf.function
        def apply_accum():
            def _a():
                finite = tf.reduce_all([tf.reduce_all(tf.math.is_finite(a)) for a in accum])
                gs = [tf.where(finite, a / float(ACC), tf.zeros_like(a)) for a in accum]
                if DISC:
                    opt_dec.apply_gradients(zip(gs[:N_DEC], TVARS[:N_DEC]))
                    opt.apply_gradients(zip(gs[N_DEC:], TVARS[N_DEC:]))
                else:
                    opt.apply_gradients(zip(gs, TVARS))
                for a in accum:
                    a.assign(tf.zeros_like(a))
            strategy.run(_a)
        print(f"GRAD-ACCUM: {ACC} micro-batches/update -> effective batch {GLOBAL * ACC}", flush=True)

    # ---- eval: Poseidon metric = MEDIAN rel-L1 at FINAL time, physical space ----
    # cmt is now the SCORE mask (loader yields feed/score split): fed-not-scored channels (e.g. Wave c) are
    # already dropped, and uv_only restricts to velocity — so the loss AND eval both score exactly the
    # predicted channels. No separate solution-slot remask needed.
    xt, cmt = test["traj"], test["cm"]; mt, st, famt = test["mean"], test["std"], test["fam"]

    @tf.function(jit_compile=jit)
    def predict(u0, dsc, cf):
        return tf.cast(model.call_with_gate(u0, dsc, cf, training=False)[0], tf.float32)

    def _evaluate_at(_ANCHOR=None):
        EB = min(max(GLOBAL, 8), int(getattr(cfg, "eval_batch", 8)))   # cap eval batch (batch40 eval OOMs)
        relL1_final = np.zeros((xt.shape[0],), np.float32)
        relL1_traj = np.zeros((xt.shape[0],), np.float32)              # FULL-trajectory: mean rel-L1 over frames 1..Nt-1
        relL1_uv = np.zeros((xt.shape[0],), np.float32)                # uv-only (slots 0,1) @final — readable trend
        relL1_uv_traj = np.zeros((xt.shape[0],), np.float32)           # uv-only full-trajectory
        for s0 in range(0, xt.shape[0], EB):                           # even when train cm supervises dummies
            sl = slice(s0, s0 + EB)
            _A = (1 if TIN2 else int(getattr(cfg, "eval_anchor", 0))) if _ANCHOR is None else _ANCHOR
            if TIN2:
                _Av = max(_A, 0)                                      # v0-slot at anchor A: v0 = u_A − u_{A−1};
                u0 = np.array(xt[sl, _Av])                            # A=0 → v0 ≡ 0 (the true zero-velocity IC)
                u0[..., V0D] = (xt[sl, _Av, :, :, V0S] - xt[sl, _Av - 1, :, :, V0S]) if _Av >= 1 else 0.0
            elif _A > 0:
                u0 = xt[sl, _A]                                       # anchor-A eval, input as loader provides
            else:
                u0 = xt[sl, 0]                                        # IC = frame 0
            dsc = tf.zeros((u0.shape[0], int(getattr(cfg, "desc_dim", cfg.n_channels))), tf.float32)
            cf = tf.zeros((u0.shape[0], 4), tf.float32)
            pred = predict(u0, dsc, cf).numpy()                      # (b,Nt,H,W,C)
            _ARS = int(getattr(cfg, "eval_ar_step", 0))              # >0: AUTOREGRESSIVE eval with hop lead _ARS
            if _ARS > 0:
                # compose short jumps: re-feed the model's own frame-_ARS output as the next anchor
                # (full 8-slot refeed — valid because dummy slots are const-supervised/alive). Probes
                # whether the single-shot long-lead deficit is integration-accumulation (then AR helps)
                # or hedge-floor (then AR compounds). Targets = _A+_ARS, _A+2·_ARS, ... ≤ Nt-1.
                _tgts = list(range(_A + _ARS, NTD, _ARS))
                _cur = np.array(u0)
                _pl = []
                _cmk = cmt[sl][:, None, None, :]                  # (b,1,1,C) score mask: refeed ONLY task slots
                _u00 = np.array(u0)                                   # non-task slots reset to anchor values —
                for _t in _tgts:                                      # finetune (uv_only/slot-loss) unlearns dummy
                    _nx = predict(_cur, dsc, cf).numpy()[:, _ARS]     # constancy, so full-slot refeed explodes
                    _cur = _nx * _cmk + _u00 * (1.0 - _cmk)
                    _pl.append(_cur)
                predAR = np.stack(_pl, axis=1)                        # (b,M,H,W,C) at data frames _tgts
            sc = st[sl][:, None, None, :]; mc = mt[sl][:, None, None, :]
            # --- @final (Poseidon paper protocol) --- (anchor A: pred frame k ↔ data frame A+k → final = Nt−1−A)
            pf = (predAR[:, -1] if _ARS > 0 else pred[:, NTD - 1 - _A]) * sc + mc  # physical final pred
            rf = xt[sl, _tgts[-1] if _ARS > 0 else NTD - 1] * sc + mc # physical final-time ref


            m = cmt[sl][:, None, None, :]                            # score mask (= predicted slots; loader now yields feed/score split)
            num = np.sum(np.abs(pf - rf) * m, axis=(1, 2, 3))
            den = np.sum(np.abs(rf) * m, axis=(1, 2, 3)) + 1e-8
            relL1_final[sl] = num / den
            nu = np.sum(np.abs(pf[..., :2] - rf[..., :2]), axis=(1, 2, 3))   # velocity (slots 0,1) only
            du = np.sum(np.abs(rf[..., :2]), axis=(1, 2, 3)) + 1e-8
            relL1_uv[sl] = nu / du
            # --- FULL trajectory: mean rel-L1 over the predicted frames 1..Nt-1 (the "solve the PDE" metric;
            #     both models predict all frames — ours single-shot, Poseidon per-leadtime — so this is the
            #     fuller, fairer comparison than scoring only the hardest final frame) ---
            sc5 = st[sl][:, None, None, None, :]; mc5 = mt[sl][:, None, None, None, :]
            m5 = cmt[sl][:, None, None, None, :]                     # score mask (= predicted slots)
            # anchor A: pred k ↔ data A+k → score data frames A+1..Nt−1 (A=0 = legacy full range).
            # eval_lead_cap L>0: score ONLY leads 1..L (lead-matched anchor sweep — without it, later
            # anchors look better purely from the shrinking remaining horizon).
            if _ARS > 0:                                             # AR mode: score at the hop grid
                pT = predAR * sc5 + mc5
                rT = xt[sl][:, _tgts] * sc5 + mc5
            else:
                _L = int(getattr(cfg, "eval_lead_cap", 0)) or (NTD - 1 - _A)
                _L = min(_L, NTD - 1 - _A)
                pT = pred[:, 1:1 + _L] * sc5 + mc5                   # (b,L,H,W,C) physical pred futures
                rT = xt[sl, 1 + _A:1 + _A + _L] * sc5 + mc5


            numt = np.sum(np.abs(pT - rT) * m5, axis=(2, 3, 4))      # (b,Nt-1)
            dent = np.sum(np.abs(rT) * m5, axis=(2, 3, 4)) + 1e-8
            relL1_traj[sl] = np.mean(numt / dent, axis=1)            # mean over frames
            nut = np.sum(np.abs(pT[..., :2] - rT[..., :2]), axis=(2, 3, 4))
            dut = np.sum(np.abs(rT[..., :2]), axis=(2, 3, 4)) + 1e-8
            relL1_uv_traj[sl] = np.mean(nut / dut, axis=1)
        med = lambda arr: {nm: float(np.median(arr[famt == fi]))
                           for fi, nm in enumerate(fam_names) if (famt == fi).any()}
        return (float(np.median(relL1_final)), med(relL1_final), med(relL1_uv),
                float(np.median(relL1_traj)), med(relL1_traj), med(relL1_uv_traj))

    def evaluate():
        # eval_anchor_sweep: print full-traj/@final at several anchor frames (anchor-robustness probe;
        # frames scored = A+1..Nt-1, horizon shrinks with A — compare against the SAME sweep on scOT).
        sweep = tuple(getattr(cfg, "eval_anchor_sweep", ()) or ())
        if not sweep:
            return _evaluate_at(None)
        res = None
        for _a in sweep:
            res = _evaluate_at(int(_a))
            print(f"  [ASWEEP anchor={_a}] MEDIAN rel-L1 FULL-TRAJ overall={res[3]*100:.3f}%  @final={res[0]*100:.3f}%", flush=True)
        return res

    def log_eval(tag):
        ov, rep, rep_uv, ovT, repT, rep_uvT = evaluate()
        s = "  ".join(f"{k}={v*100:.3f}%" for k, v in rep.items())
        suv = "  ".join(f"{k}={v*100:.3f}%" for k, v in rep_uv.items())
        sT = "  ".join(f"{k}={v*100:.3f}%" for k, v in repT.items())
        suvT = "  ".join(f"{k}={v*100:.3f}%" for k, v in rep_uvT.items())
        print(f"  [EVAL {tag}] MEDIAN rel-L1 @final (physical, SOL-SLOT) overall={ov*100:.3f}%  | {s}  || UV-ONLY: {suv}", flush=True)
        print(f"  [EVAL {tag}] MEDIAN rel-L1 FULL-TRAJ (physical, SOL-SLOT) overall={ovT*100:.3f}%  | {sT}  || UV-ONLY: {suvT}", flush=True)

    save_dir = getattr(cfg, "save_dir", None); ckpt_every = int(getattr(cfg, "ckpt_every", 0))
    start = 1
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cks = sorted(glob.glob(os.path.join(save_dir, "ckpt_*.npz")),
                     key=lambda p: int(p.split("_")[-1].split(".")[0]))
        if cks:
            ckptlib.load(model, cks[-1]); start = int(cks[-1].split("_")[-1].split(".")[0]) + 1
            print(f"  resumed @ {start}", flush=True)
        elif getattr(cfg, "init_from", None):
            # WARM-RESTART: load weights from another run but keep a FRESH step counter (start=1)
            # and a FRESH LR cosine over cfg.steps. Used to continue pretraining with MORE budget at
            # a re-warmed LR (resuming in-place would replay the decayed-to-floor LR → no learning).
            # On a later crash, save_dir already has ckpts → the resume branch above takes over.
            ckptlib.load(model, cfg.init_from)
            print(f"  warm-start init from {cfg.init_from} (fresh step counter + LR cosine)", flush=True)

    it = iter(dist_ds)
    print("=== IVP training (first step XLA compile, ~minutes) ===", flush=True)
    rng = np.random.default_rng(int(getattr(cfg, "seed", 0)))
    diverged = False
    nan_strk = 0
    _detic = bool(getattr(cfg, "det_ic", False))                      # deterministic all2all IC cycling (vs random)
    def _pick_ic(s, mi):
        # det_ic: cycle ic = 0,1,..,max_ic,0,.. across micro-batches (resume-safe, derived from step) so every
        # IC is used EXACTLY equally (all2all uniform coverage, clean for the paper). else: random per micro-batch.
        if _detic:
            return tf.constant(((s - 1) * ACC + mi) % (max_ic + 1), tf.int32)
        return tf.constant(int(rng.integers(0, max_ic + 1)), tf.int32)
    # per-step bookkeeping (print / NaN-skip / mem / ckpt / decoupled eval) — shared by both loops
    def _post(s, loss, i):
        nonlocal nan_strk, diverged
        if s % 50 == 0 or s == 1:
            fl = float(loss)
            print(f"  step {s:5d} loss={fl:.4e} ic={int(i)}", flush=True)
            if not np.isfinite(fl):                                   # NaN-skip held the weights in step_fn
                nan_strk += 1
                print(f"  NaN @ step {s} — grads skipped (weights held), strike {nan_strk}", flush=True)
                if nan_strk >= 10:                                    # ~500 steps all-NaN = real divergence
                    print(f"  DIVERGED: {nan_strk} consecutive NaN prints — stopping (last good = ckpt_{(s//ckpt_every)*ckpt_every})", flush=True)
                    diverged = True; return
            else:
                nan_strk = 0
        if s == 1:
            try:
                _mi = tf.config.experimental.get_memory_info("GPU:0")
                print(f"  [mem] GPU:0 peak={_mi['peak']/1e9:.1f}GB", flush=True)
            except Exception:
                pass
        if save_dir and ckpt_every and s % ckpt_every == 0:
            ckptlib.save(model, os.path.join(save_dir, f"ckpt_{s}.npz"))
            json.dump({"step": s}, open(os.path.join(save_dir, "state.json"), "w"))
            print(f"  [ckpt] {s}", flush=True)
        # eval cadence DECOUPLED from ckpt: eval_every (default = ckpt_every) lets us watch the curve
        # frequently without writing a checkpoint (and filling the volume) at every eval point.
        _ee = int(getattr(cfg, "eval_every", ckpt_every or 0))
        if _ee and (s % _ee == 0 or s <= 2):   # also eval at step 1,2 — warm-restart canary (did loading/LR shake the weights?)
            log_eval(f"step {s}")

    # IC-REUSE (the read-amortizing all2all): the reader yields a FULL trajectory but each optimizer step
    # uses ONE ic, so naive all-IC coverage re-reads each trajectory (max_ic+1)x — and the single serial
    # HDF5 reader (~18 traj/s, can't be threaded: HDF5 not thread-safe) then caps throughput. Instead, reuse
    # one read batch for ALL ICs 0..max_ic (shuffled) as consecutive updates -> reader load drops ~20x ->
    # compute-bound -> multi-GPU scales. cfg.steps still counts optimizer updates. (ACC=1 path only.)
    _reuse = bool(getattr(cfg, "ic_reuse", False)) and ACC == 1
    if _reuse:
        _icrng = np.random.default_rng(int(getattr(cfg, "seed", 0)) + 7919)
        s = start
        print(f"  IC-REUSE: each read batch -> all {max_ic + 1} ICs as consecutive updates", flush=True)
        while s <= cfg.steps and not diverged:
            traj, cm, _m, _s, _f = next(it)
            ics = _icrng.permutation(max_ic + 1) if max_ic > 0 else np.array([0])
            for ic in ics:
                if s > cfg.steps or diverged:
                    break
                loss = dist_step(traj, cm, tf.constant(int(ic), tf.int32))
                _post(s, loss, int(ic))
                s += 1
    else:
        for s in range(start, cfg.steps + 1):
            if ACC > 1:                                               # accumulate ACC micro-batches -> 1 update
                _ls = []
                for _mi in range(ACC):
                    traj, cm, _m, _s, _f = next(it)
                    i = _pick_ic(s, _mi)
                    _ls.append(accum_step(traj, cm, i))
                apply_accum()
                loss = tf.reduce_mean(_ls)
            else:
                traj, cm, _m, _s, _f = next(it)
                i = _pick_ic(s, 0)                                    # all2all IC frame for this step
                loss = dist_step(traj, cm, i)
            _post(s, loss, int(i))
            if diverged:
                break
    if save_dir and not diverged:
        ckptlib.save(model, os.path.join(save_dir, f"ckpt_{cfg.steps}.npz"))
    if not diverged:
        log_eval("final")
    print("DONE" if not diverged else "STOPPED (diverged)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "motion_tf.train.configs.poseidon_pretrain_ivp")
