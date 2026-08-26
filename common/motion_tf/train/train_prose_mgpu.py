"""Multi-GPU (data-parallel) PROSE-FD training — tf.distribute.MirroredStrategy.

    python -m motion_tf.train.train_prose_mgpu motion_tf.train.configs.prose_swe_mgpu

Data-parallel: per-replica batch = cfg.batch (keep 1 so 1 sample's rollout activation fits one
H100 without checkpoint → jit works). Effective batch = cfg.batch × #GPUs. jit=True (compile is
one-time ~minutes for the unrolled rollout), grad_checkpoint=False. Runs on 1 GPU too (1 replica)
for code validation. Intermediate ckpt saved to save_dir (synced to volume by the runner).
"""
from __future__ import annotations
import glob, importlib, json, os, sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf

for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
# Cap CPU thread pools — on many-vCPU H100 nodes, MirroredStrategy's per-replica pools otherwise
# exhaust the container nproc limit (pthread_create errno 11 → SIGABRT).
try:
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(4)
except RuntimeError:
    pass

from ..model import PhysicsOperatorMixture
from ..data import prose
from ..utils import ckpt as ckptlib


def per_sample_masked_rel_l2(pred, ref, c_mask, t_mask=None, eps=1e-6, cap=0.0):
    m = c_mask[:, None, None, None, :]                               # (B,1,1,1,6) per-sample channel mask
    num = tf.reduce_sum(((pred - ref) ** 2) * m, axis=[2, 3, 4])      # (B,T)
    den = tf.reduce_sum((ref ** 2) * m, axis=[2, 3, 4]) + eps
    ratio = num / den                                                # (B,T) per-frame squared rel-L2
    # cap>0: CLAMP the per-frame ratio so a frame whose normalized target has tiny norm (field ≈ its own
    # mean → den→0) can't blow the loss/grad up (the 180-init-spike pathology). Keeps rel-L2's per-frame
    # scale-invariance + EVAL-metric alignment, without the small-norm divergence. cap=4.0 ≈ 200% rel-L2.
    if cap and cap > 0:
        ratio = tf.minimum(ratio, float(cap))
    if t_mask is not None:                                           # TEMPORAL mask (B,T): exclude padded
        tm = tf.cast(t_mask, ratio.dtype)                            # output frames (uncond t>valid_t) from loss
        return tf.reduce_sum(ratio * tm, axis=1) / (tf.reduce_sum(tm, axis=1) + eps)   # masked mean over valid frames
    return tf.reduce_mean(ratio, axis=1)                             # (B,) per-sample (all frames valid)


def per_sample_relL2_norm(pred, ref, c_mask, t_mask=None, eps=1e-7):
    """EVAL metric = PROSE/BCAT-exact: per-frame rel-L2 NORM (||err||/||ref||, NOT squared), ARITHMETIC mean
    over valid frames → per-sample. evaluate() then takes mean over samples (per family) + class-average over
    families. Matches code/eval/eval_prose_exact.py exactly so the inline-logged curve == the post-hoc curve.
    (Distinct from per_sample_masked_rel_l2 above, which is the SQUARED/capped TRAINING loss — unchanged.)"""
    m = c_mask[:, None, None, None, :]
    num = tf.sqrt(tf.reduce_sum(((pred - ref) ** 2) * m, axis=[2, 3, 4]))   # (B,T) ||err||
    den = tf.sqrt(tf.reduce_sum((ref ** 2) * m, axis=[2, 3, 4])) + eps      # (B,T) ||ref||  (mean-included)
    r = num / den                                                          # (B,T) per-frame rel-L2 norm
    if t_mask is not None:
        tm = tf.cast(t_mask, r.dtype)
        return tf.reduce_sum(r * tm, axis=1) / (tf.reduce_sum(tm, axis=1) + eps)   # mean over VALID frames
    return tf.reduce_mean(r, axis=1)                                       # (B,) per-sample


def per_sample_masked_mse(pred, ref, c_mask, t_mask=None):
    """PROSE's training LOSS: plain MSE over active (masked) elements — NO per-frame denominator (so it
    does NOT blow up on frames whose normalized target has small norm, unlike rel-L2-as-loss). In
    normalized space (instance_norm) the target is ~unit-variance → MSE is O(1). rel-L2 stays the EVAL
    metric only. (Mismatch found 2026-06-11: we were using rel-L2 as the loss → 180-scale init spike.)
    t_mask (B,T): also zero out padded output frames (uncond t>valid_t) so they don't dilute the loss."""
    m = c_mask[:, None, None, None, :]
    se = ((pred - ref) ** 2) * m                                     # (B,T,H,W,C)
    if t_mask is not None:
        tm = tf.cast(t_mask, se.dtype)[:, :, None, None, None]        # (B,T,1,1,1) temporal validity
        se = se * tm
        cnt = tf.reduce_sum(tf.ones_like(se) * m * tm, axis=[1, 2, 3, 4]) + 1e-8
    else:
        cnt = tf.reduce_sum(tf.ones_like(se) * m, axis=[1, 2, 3, 4]) + 1e-8
    num = tf.reduce_sum(se, axis=[1, 2, 3, 4])                        # (B,)
    return num / cnt                                                 # (B,) per-sample MSE


def main(cfg_path):
    cfg = importlib.import_module(cfg_path).Cfg
    jit = bool(getattr(cfg, "jit_compile", True))
    # bf16 mixed precision (set BEFORE model build): conv/expert compute in bf16 (½ memory, ~2×
    # H100 throughput), master weights + loss stay fp32. No loss scaling needed (bf16 keeps fp32's
    # exponent range). The rollout carry + ADA basis + field/loss are fp32 (cast at boundaries).
    if bool(getattr(cfg, "bf16", False)):
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
        print("mixed_bfloat16 policy enabled", flush=True)
    # ReductionToOneDevice avoids NCCL entirely (copy grads to one GPU + add) — the XLA-fused
    # nccl all_reduce thunk fails with "unhandled cuda error" in this container. For 2 GPUs the
    # cost is negligible.
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.ReductionToOneDevice())
    nrep = strategy.num_replicas_in_sync
    GLOBAL = cfg.batch * nrep
    label = ",".join(cfg.datasets) if hasattr(cfg, "datasets") else cfg.dataset
    print(f"=== PROSE mGPU {label} | replicas={nrep} global_batch={GLOBAL} jit={jit} ===", flush=True)

    Ti, Nt = cfg.T_in, cfg.Nt
    cache_dir = getattr(cfg, "cache_dir", None) or os.environ.get("PREBUILT_DIR")
    _stream = bool(getattr(cfg, "stream", False))
    _test = None
    if _stream:                                                      # STREAMING: no RAM load / no copy
        from ..data import stream
        ds, _test = stream.make_train_stream(cfg, GLOBAL, cache_dir)
        fam_names = _test["fam_names"]
        print(f"=== STREAMING train | families={fam_names} ===", flush=True)
    elif hasattr(cfg, "datasets"):                                   # multi-family, load-all-to-RAM
        d = prose.build_multi(cfg.datasets, cfg.n_per, cfg.t_num, seed=cfg.seed, cache_dir=cache_dir)
        cm_all, desc_all = d["c_mask"], d["desc"]
        fam_all, fam_names = d["fam"], d["fam_names"]
    else:                                                            # single-family
        d = prose.build(cfg.dataset, cfg.n_total, cfg.t_num, cfg.t_step, seed=cfg.seed)
        N0 = d["u"].shape[0]
        cm_all = np.tile(d["c_mask"], (N0, 1)).astype(np.float32)
        desc_all = np.tile(d["desc"], (N0, 1)).astype(np.float32)
        fam_all = np.zeros(N0, np.int32); fam_names = [cfg.dataset]
    if not _stream:
        tr = d["tr"]
        u_tr = d["u"][tr]
        uin = u_tr[:, :Ti].astype(np.float32)
        tgt = u_tr[:, Ti - 1: Ti - 1 + Nt].astype(np.float32)
        cm_tr = cm_all[tr].astype(np.float32)
        desc = desc_all[tr].astype(np.float32)
        coef = np.zeros((uin.shape[0], 4), np.float32)
        tm_tr = np.ones((uin.shape[0], Nt), np.float32)              # non-stream: no uncond → all frames valid
        gm_tr = np.ones((uin.shape[0], uin.shape[2], uin.shape[3], 1), np.float32)  # non-stream → no geometry (Π=I)
        N = uin.shape[0]
        print(f"train {uin.shape} -> tgt {tgt.shape}  | per-sample c_mask {cm_tr.shape}", flush=True)
        ds = (tf.data.Dataset.from_tensor_slices((uin, tgt, desc, coef, cm_tr, tm_tr, gm_tr))
              .shuffle(N).repeat().batch(GLOBAL, drop_remainder=True))
    dist_ds = strategy.experimental_distribute_dataset(ds)

    with strategy.scope():
        model = PhysicsOperatorMixture(cfg)
        # Cosine LR decay → lowers the late-training noise floor (constant LR plateaus while loss is
        # still dropping). alpha=0.05 → final LR = 5% of peak. NOTE: ckpt saves model vars only, so a
        # resume restarts the schedule from step 0 — fine for a single full run.
        # warmup (PROSE uses ~10%): ramp 0→lr over warmup steps, THEN cosine-decay to 5%. No-warmup
        # cosine starts at peak lr during the fragile early rollout → the early loss spikes we saw.
        _wu = int(getattr(cfg, "warmup", 0))
        _al = float(getattr(cfg, "lr_alpha", 0.05))   # continuation runs: raise to keep LR in a narrow band
        if getattr(cfg, "lr_decay", False):
            lr = (tf.keras.optimizers.schedules.CosineDecay(
                      cfg.lr * 0.02, cfg.steps, alpha=_al, warmup_target=cfg.lr, warmup_steps=_wu)
                  if _wu > 0 else
                  tf.keras.optimizers.schedules.CosineDecay(cfg.lr, cfg.steps, alpha=_al))
        else:
            lr = cfg.lr
        # clipnorm: ran stable 4000 steps then a rare extreme-magnitude batch spiked the gradient →
        # weights went NaN and stayed NaN. Global-norm clipping caps that spike (standard fix).
        # optimizer knobs (default = plain Adam, our standard). Set cfg.optimizer="adamw" + weight_decay
        # + beta_2 to MATCH the PROSE/BCAT baselines (AdamW, wd 1e-4, beta2 0.95) for a fair-optimizer run.
        _b2 = float(getattr(cfg, "beta_2", 0.999))
        _cn = float(getattr(cfg, "clipnorm", 1.0))
        if str(getattr(cfg, "optimizer", "adam")).lower() == "adamw":
            _wd = float(getattr(cfg, "weight_decay", 1e-4))
            opt = tf.keras.optimizers.AdamW(lr, weight_decay=_wd, beta_2=_b2, clipnorm=_cn)
            print(f"  optimizer: AdamW (wd={_wd}, beta_2={_b2}, clipnorm={_cn})", flush=True)
        else:
            opt = tf.keras.optimizers.Adam(lr, beta_2=_b2, clipnorm=_cn)
        # gradient accumulation: per-GPU batch stays 1 (memory!), but accumulate K micro-batches
        # before applying → effective batch = batch × #GPU × K (reach Poseidon/PROSE-scale eff
        # batch without more memory). K=1 → no accumulation (plain path).
        K = int(getattr(cfg, "grad_accum", 1))
        accum = ([tf.Variable(tf.zeros_like(v), trainable=False) for v in model.trainable_variables]
                 if K > 1 else None)
        # PCGrad (multi-task gradient surgery): treat each of the K micro-batches (= 1 random-family
        # sample) as a separate "task" gradient. Before applying, for every ordered pair (k,j) project
        # g_k onto the normal plane of g_j whenever they CONFLICT (g_k·g_j<0) → removes the destructive
        # inter-family component, then sum. Also logs the mean pairwise cosine = a direct MEASURE of how
        # much the families fight. Needs K separate grad buffers (gbufs). Single-replica only.
        _pcgrad = bool(getattr(cfg, "pcgrad", False)) and K > 1 and nrep == 1
        gbufs = ([[tf.Variable(tf.zeros_like(v), trainable=False) for v in model.trainable_variables]
                  for _ in range(K)] if _pcgrad else None)
        if bool(getattr(cfg, "pcgrad", False)) and not _pcgrad:
            print(f"  [pcgrad] DISABLED (needs K>1 and nrep==1; K={K} nrep={nrep})", flush=True)
    npar = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    print(f"model params: {npar:,}  | grad_accum K={K} -> effective_batch={GLOBAL * K}", flush=True)
    lam = float(cfg.lambda_gate)
    _mse = bool(getattr(cfg, "loss_mse", False))                       # PROSE-style MSE loss (vs rel-L2)
    _relcap = float(getattr(cfg, "loss_relclamp", 0.0))                # >0 → clamped rel-L2 loss (cap ratio)
    # AR-t5: ar_seg=2 → the ADA horizon is cfg.Nt (=6, t=0.5); the model predicts _SL=Nt-1 futures per call
    # and we CHAIN 2 segments (free-running, stop-grad at the boundary) to cover the full 2*_SL=10-future
    # benchmark horizon (data_nt=11). Shorter per-segment horizon tames chaotic-NS exponential error;
    # training FOR AR removes the single-shot→AR distribution shift that made AR-eval diverge. 0 → single-shot.
    _AR = int(getattr(cfg, "ar_seg", 0)); _SL = int(cfg.Nt) - 1

    # jit-compile ONLY the expensive forward+backward (the N_p-unrolled rollout). apply_gradients
    # stays OUTSIDE jit so the cross-replica all-reduce is a normal TF-runtime op, never compiled
    # into XLA's fragile nccl thunk. apply is cheap (Adam on ~5M params) so ~all speedup is kept.
    # jit on the per-replica fn (not the strategy.run wrapper) → each replica compiles its own vars.
    @tf.function(jit_compile=jit)
    def compute_grads(uin, tgt, dsc, cf, cm, tm, gm):
        with tf.GradientTape() as tape:
            _lf = ((lambda p, t, tmm: per_sample_masked_mse(p, t, cm, t_mask=tmm)) if _mse
                   else (lambda p, t, tmm: per_sample_masked_rel_l2(p, t, cm, t_mask=tmm, cap=_relcap)))
            if _AR >= 2:                                              # K-segment autoregressive (short ADA horizon)
                per = 0.0; gate_reg = 0.0; win = uin
                _fair = bool(getattr(cfg, "ar_fair_loss", False))       # fair: ONE masked-mean over ALL segs' valid
                _ff = []; _tg = []; _tmm = []                            # frames (short-horizon uncond not down-weighted)
                for _k in range(_AR):
                    pk, ak = model.call_with_gate(win, dsc, cf, geom_mask=gm, training=True)
                    pk = tf.cast(pk, tf.float32); fut = pk[:, 1:1 + _SL]     # this segment's _SL futures
                    _sl_tg = tgt[:, 1 + _k * _SL:1 + (_k + 1) * _SL]; _sl_tm = tm[:, 1 + _k * _SL:1 + (_k + 1) * _SL]
                    if _fair:
                        _ff.append(fut); _tg.append(_sl_tg); _tmm.append(_sl_tm)   # defer: combine after loop
                    else:
                        per = per + _lf(fut, _sl_tg, _sl_tm)             # legacy: SUM of per-segment means (uncond ~1/K)
                    gate_reg = gate_reg + tf.cast(tf.reduce_sum(tf.reduce_sum(ak, -1)), tf.float32)
                    if _k < _AR - 1:                                   # slide window, append own futures (stop-grad)
                        win = tf.concat([win[:, _SL:], tf.cast(tf.stop_gradient(fut), uin.dtype)], axis=1)
                if _fair:                                               # AR-FAIR: match eval normalization
                    per = _lf(tf.concat(_ff, axis=1), tf.concat(_tg, axis=1), tf.concat(_tmm, axis=1))
            else:
                pred, alpha = model.call_with_gate(uin, dsc, cf, geom_mask=gm, training=True, target=tgt)  # teacher-forcing (gated)
                pred = tf.cast(pred, tf.float32)                      # field fp32 for the loss
                per = _lf(pred, tgt, tm)
                gate_reg = tf.cast(tf.reduce_sum(tf.reduce_sum(alpha, -1)), tf.float32)  # alpha may be bf16
            loss = tf.reduce_sum(per) / GLOBAL + lam * gate_reg / GLOBAL
        g = tape.gradient(loss, model.trainable_variables)
        return loss, g

    def step_fn(uin, tgt, dsc, cf, cm, tm, gm):
        loss, g = compute_grads(uin, tgt, dsc, cf, cm, tm, gm)
        # NaN-SKIP: zero ALL grads when any is non-finite → no-op update (weights held), skip the bad
        # batch instead of going NaN (rare advection/forward overflow). Next good batch recovers.
        finite = tf.reduce_all([tf.reduce_all(tf.math.is_finite(gi)) for gi in g if gi is not None])
        g = [None if gi is None else tf.where(finite, gi, tf.zeros_like(gi)) for gi in g]
        opt.apply_gradients(zip(g, model.trainable_variables))        # all-reduce here, outside jit
        return loss

    @tf.function
    def dist_step(batch):
        per = strategy.run(step_fn, args=batch)
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per, axis=None)

    # --- gradient-accumulation variants (used when K>1) ---
    def micro_fn(uin, tgt, dsc, cf, cm, tm, gm):                      # accumulate, do NOT apply
        loss, g = compute_grads(uin, tgt, dsc, cf, cm, tm, gm)
        for a, gi in zip(accum, g):
            if gi is not None:
                a.assign_add(tf.cast(gi, a.dtype))                    # grads may be bf16; accum fp32
        return loss

    def apply_fn():                                                  # apply mean accum, then zero
        gmean = [a / K for a in accum]
        finite = tf.reduce_all([tf.reduce_all(tf.math.is_finite(g)) for g in gmean])   # NaN-skip
        gmean = [tf.where(finite, g, tf.zeros_like(g)) for g in gmean]
        opt.apply_gradients(zip(gmean, model.trainable_variables))
        for a in accum:
            a.assign(tf.zeros_like(a))

    @tf.function
    def dist_micro(batch):
        per = strategy.run(micro_fn, args=batch)
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per, axis=None)

    @tf.function
    def dist_apply():
        strategy.run(apply_fn)

    # --- PCGrad variants (single-replica) ---
    def _store_k_fn(buf, uin, tgt, dsc, cf, cm, tm, gm):             # compute grad → buf[*] (per micro k)
        loss, g = compute_grads(uin, tgt, dsc, cf, cm, tm, gm)
        for b, gi in zip(buf, g):
            b.assign(tf.cast(gi, b.dtype) if gi is not None else tf.zeros_like(b))
        return loss

    def _make_dist_store(k):                                        # one tf.function per micro index k
        @tf.function
        def f(batch):
            per = strategy.run(lambda *a: _store_k_fn(gbufs[k], *a), args=batch)
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per, axis=None)
        return f
    dist_store = [_make_dist_store(k) for k in range(K)] if _pcgrad else None

    @tf.function
    def dist_pcgrad_apply():
        def f():
            gv = [[tf.identity(v) for v in gbufs[k]] for k in range(K)]   # read buffers → tensors
            def gdot(A, B):
                return tf.add_n([tf.reduce_sum(a * b) for a, b in zip(A, B)])
            sq = [gdot(gv[k], gv[k]) + 1e-12 for k in range(K)]
            nrm = [tf.sqrt(x) for x in sq]
            # conflict metric: mean pairwise cosine of the ORIGINAL task gradients
            cs, npair = tf.constant(0.0), 0.0
            for i in range(K):
                for j in range(i + 1, K):
                    cs = cs + gdot(gv[i], gv[j]) / (nrm[i] * nrm[j]); npair += 1.0
            cos = cs / npair
            # PCGrad: project proj[k] off each conflicting ORIGINAL g_j (sequential, fixed order)
            proj = [list(gv[k]) for k in range(K)]
            for k in range(K):
                for j in range(K):
                    if j == k:
                        continue
                    d = tf.add_n([tf.reduce_sum(p * q) for p, q in zip(proj[k], gv[j])])
                    scale = tf.minimum(d, 0.0) / sq[j]               # <0 only when conflicting
                    proj[k] = [p - scale * q for p, q in zip(proj[k], gv[j])]
            total = [tf.add_n([proj[k][i] for k in range(K)]) / float(K)
                     for i in range(len(gv[0]))]
            opt.apply_gradients(zip(total, model.trainable_variables))
            return cos
        return strategy.run(f)

    # ---- held-out TEST evaluation (per-family rel-L2). Inference reads the mirrored vars on the
    # primary device; non-jit @tf.function so the partial last batch doesn't force a recompile. ----
    if _stream:                                                      # test set materialized by stream
        xin, xtg = _test["xin"], _test["xtg"]
        cm_te, desc_te, cf_te, fam_te = _test["cm"], _test["desc"], _test["coef"], _test["fam"]
        tm_te = _test["tm"]                                           # per-sample temporal validity (Nte,Nt)
        mean_te, std_te = _test["mean"], _test["std"]                 # per-sample input-window stats
        geom_te = _test["geom"]                                       # per-sample geom_mask (cfdbench boundary / ones)
    else:
        te = d["te"]
        u_te = d["u"][te]
        xin = u_te[:, :Ti].astype(np.float32)
        xtg = u_te[:, Ti - 1: Ti - 1 + Nt].astype(np.float32)
        cm_te = cm_all[te].astype(np.float32); desc_te = desc_all[te].astype(np.float32)
        cf_te = np.zeros((xin.shape[0], 4), np.float32); fam_te = np.asarray(fam_all)[te]
        tm_te = np.ones((xin.shape[0], Nt), np.float32)               # non-stream: all frames valid
        mean_te = np.zeros((xin.shape[0], xin.shape[-1]), np.float32) # non-stream data already physical
        std_te = np.ones((xin.shape[0], xin.shape[-1]), np.float32)
        geom_te = np.ones((xin.shape[0], xin.shape[2], xin.shape[3], 1), np.float32)  # non-stream → no geometry
    EVAL_BATCH = max(GLOBAL, 8)

    @tf.function
    def eval_fn(u0, ref, dsc, cf, cm, tm, mean, std, geom):
        sc = std[:, None, None, None, :] + 1e-6                        # DENORMALIZE → physical space (PROSE metric)
        mc = mean[:, None, None, None, :]
        if _AR >= 2:                                                   # K-segment AR: chain _AR*_SL futures
            preds = []; win = u0
            for _k in range(_AR):
                pk = tf.cast(model.call_with_gate(win, dsc, cf, geom_mask=geom)[0], tf.float32)
                preds.append(pk[:, 1:1 + _SL])
                if _k < _AR - 1:
                    win = tf.concat([win[:, _SL:], tf.cast(pk[:, 1:1 + _SL], u0.dtype)], axis=1)
            predf = tf.concat(preds, axis=1) * sc + mc                 # (B, _AR*_SL) futures, physical
            reff = ref[:, 1:1 + _AR * _SL] * sc + mc
            return per_sample_relL2_norm(predf, reff, cm, t_mask=tm[:, 1:1 + _AR * _SL])
        pred, _ = model.call_with_gate(u0, dsc, cf, geom_mask=geom)
        pred = pred * sc + mc; ref = ref * sc + mc                    # (mean-included den); LOSS stays normalized
        # DROP the hard-IC frame (index 0 = t=0 = the last input frame): pred[:,0]=u0=ref[:,0] EXACTLY (hard-IC),
        # so its rel-L2 is identically 0 — a freebie that PROSE/BCAT never get (they score futures-only, frames
        # T_in..end). Scoring pred[:,1:] = the FORECAST horizon = exactly felix-lyx's frames T_in..end, making the
        # inline [EVAL] == eval_prose_exact == PROSE/BCAT-comparable. (tm[:,1:] keeps the per-family valid-frame
        # count correct, incl uncond's 4.) Was: scored all Nt frames incl the IC → ~×11/10 (uncond ×5/4) optimistic.
        pred = pred[:, 1:]; ref = ref[:, 1:]; tm = tm[:, 1:]
        return per_sample_relL2_norm(pred, ref, cm, t_mask=tm)         # (B,) PROSE-exact per-sample rel-L2 norm

    def evaluate():
        eds = tf.data.Dataset.from_tensor_slices(
            (xin, xtg, desc_te, cf_te, cm_te, tm_te, mean_te, std_te, geom_te)).batch(EVAL_BATCH)
        pers = [eval_fn(*b).numpy() for b in eds]
        per = np.concatenate(pers) if pers else np.zeros((0,))         # (Nte,) per-sample rel-L2 norm (PROSE-exact)
        rep = {nm: float(per[fam_te == fi].mean())                     # per-family = ARITHMETIC mean over samples
               for fi, nm in enumerate(fam_names) if (fam_te == fi).any()}
        overall = float(np.mean(list(rep.values()))) if rep else 0.0   # OVERALL = class-average (AVE_BY_CLASS)
        return overall, rep                                            # PROSE/BCAT convention (matches eval_prose_exact)

    def log_eval(tag):
        ov, rep = evaluate()
        fam_str = "  ".join(f"{k}={v*100:.3f}%" for k, v in rep.items())
        print(f"  [EVAL {tag}] test PHYS rel-L2 overall={ov*100:.3f}%  | {fam_str}", flush=True)

    save_dir = getattr(cfg, "save_dir", None); ckpt_every = int(getattr(cfg, "ckpt_every", 0))
    # DATA-EFFICACY run: eval FREQUENTLY (eval_every steps → dense training-trajectory curve, logged) but save
    # ckpts only at LOG-SCALE milestones (ckpt_milestones = explicit step list → bounded disk + post-hoc eval).
    # The frequent inline eval is the primary curve (survives via the runner's /eu log tee); the log-scale
    # ckpts are the post-hoc backup. eval_every defaults to ckpt_every (legacy) if unset.
    eval_every = int(getattr(cfg, "eval_every", 0)) or ckpt_every
    ckpt_milestones = set(int(x) for x in getattr(cfg, "ckpt_milestones", []))
    start = 1
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cks = sorted(glob.glob(os.path.join(save_dir, "ckpt_*.npz")),
                     key=lambda p: int(p.split("_")[-1].split(".")[0]))
        if cks:
            ckptlib.load(model, cks[-1]); start = int(cks[-1].split("_")[-1].split(".")[0]) + 1
            opt.iterations.assign(start - 1)          # align LR schedule: cosine(step) continues, not restart
            print(f"  resumed @ {start} (LR schedule aligned: opt.iterations={start-1})", flush=True)
        elif getattr(cfg, "init_ckpt", None):                         # WARM-START from another run's weights
            init = cfg.init_ckpt                                      # (e.g. Task-1 ckpt) — load as init, the
            ckptlib.load(model, init)                                 # step counter + LR schedule start FRESH
            print(f"  warm-start from {init} (fresh step counter / LR schedule)", flush=True)

    it = iter(dist_ds)
    print("=== training (first step triggers XLA compile, ~minutes) ===", flush=True)
    _cos = float("nan")
    _nan_strk = 0
    for s in range(start, cfg.steps + 1):
        if _pcgrad:                                                   # K micro-grads → PCGrad → apply
            tl = 0.0
            for k in range(K):
                tl += float(dist_store[k](next(it)))
            _cos = float(dist_pcgrad_apply())
            loss = tl / K
        elif K > 1:                                                   # accumulate K micro-batches
            tl = 0.0
            for _ in range(K):
                tl += float(dist_micro(next(it)))
            dist_apply()
            loss = tl / K
        else:
            loss = dist_step(next(it))
        if s % 50 == 0 or s == 1:
            fl = float(loss)
            _ct = f" gcos={_cos:+.3f}" if _pcgrad else ""              # mean pairwise family-grad cosine
            print(f"  step {s:5d} loss={fl:.4e}{_ct}", flush=True)
            if not np.isfinite(fl):                                   # NaN-skip already held the weights
                _nan_strk += 1
                print(f"  NaN @ step {s} — grads skipped (weights held), strike {_nan_strk}", flush=True)
                if _nan_strk >= 10:                                   # ~500 steps all-NaN = real divergence
                    print(f"  DIVERGED: {_nan_strk} consecutive NaN prints — stopping", flush=True)
                    break
            else:
                _nan_strk = 0
        if s == 1:                                                    # actual per-GPU peak memory
            try:
                mi = tf.config.experimental.get_memory_info("GPU:0")
                print(f"  [mem] GPU:0 peak={mi['peak']/1e9:.1f}GB cur={mi['current']/1e9:.1f}GB "
                      f"(batch={cfg.batch}/replica, fits 80GB)", flush=True)
            except Exception:
                pass
        if save_dir and ((ckpt_milestones and s in ckpt_milestones) or
                         (not ckpt_milestones and ckpt_every and s % ckpt_every == 0)):
            ckptlib.save(model, os.path.join(save_dir, f"ckpt_{s}.npz"))
            json.dump({"step": s}, open(os.path.join(save_dir, "state.json"), "w"))
            print(f"  [ckpt] {s}", flush=True)
            _keep = int(getattr(cfg, "ckpt_keep", 0))
            if _keep:                                                 # bounded-disk convergence runs: retain
                _cks = sorted(glob.glob(os.path.join(save_dir, "ckpt_*.npz")),  # only the newest N ckpts
                              key=lambda p: int(p.split("_")[-1].split(".")[0]))
                for _p in _cks[:-_keep]:
                    os.remove(_p)
                    print(f"  [ckpt] pruned {os.path.basename(_p)}", flush=True)
        if eval_every and s % eval_every == 0:                       # FREQUENT inline eval (decoupled from ckpt)
            log_eval(f"step {s}")
    if save_dir:
        ckptlib.save(model, os.path.join(save_dir, f"ckpt_{cfg.steps}.npz"))
    log_eval("final")
    print("DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "motion_tf.train.configs.prose_swe_mgpu")
