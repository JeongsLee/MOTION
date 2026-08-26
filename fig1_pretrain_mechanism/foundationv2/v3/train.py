"""v3 pretraining driver — full-step AR over the 26-family unified corpus (DESIGN_V3).

  python -m v3.train --families pretrain --steps 60000 --save /corpus/results/v3ar_r1

Paradigm (bylfa asset): every training view is a free-running AR rollout — the model
predicts frame k+1 from a window containing its OWN predictions (stop-gradient at each
boundary), and ONE masked mean over all segments' frames is the loss (ar_fair). Steady
families run the same loop as a K_relax-step relaxation supervised at every step
(pseudo-transient continuation), so a single operator serves transient AND steady.

Reuses the v2 ops discipline: never-drop family interleave, std-only normalization,
named-npz warm-start toolkit, MirroredStrategy(ReductionToOneDevice), XLA optional,
periodic per-family eval with best-ckpt tracking.
"""
from __future__ import annotations

import argparse
import functools
import os
import threading
import time

import numpy as np
import tensorflow as tf

from core.pretrain import _WarmupConst, load_ckpt, save_ckpt
from data import loader
from data.registry import (NUM_SLOTS, SEMANTIC_SLOTS, motion_families,
                           pretrain_families)

from .data_ar import bucketed, make_example_ar, stack_batch
from .model import V3Model


# ---------------------------------------------------------------------------------------------
# LATENT RESOLUTION is raised uniformly (--dims2/--dims3), not per family. diag_resolution.py fits
# the box values themselves to the true increment, so its residual is the error the readout CANNOT
# go below however long we train. At r10's 64^2/32^3 the measured floors were binding for
# pdearena_ns 8.24, CE-CRP 6.99, CE-RM 4.08, CE-RP 3.40, geofno_airfoil 3.72, shapenet_car 3.72
# and drivaernet 13.19 (i.e. half of drivaernet's 30.6% was representation, not learning);
# doubling the box puts every one of them under ~1% (128^2 -> 0.00, 64^3 -> 2.01/0.47). The core
# holds no dims-shaped weights, so the same weight set runs at any resolution and r10's
# checkpoint warm-starts directly. Cost: 4x the box cells in 2D and 8x in 3D, which is what the
# sparse-MoE dispatch below is there to pay for.

# FLOOR-AWARE SAMPLING. Plain round-robin spends 1/26 of the budget on every family, but six of
# them have nothing left to learn (shallow_water 0.2 vs floor 0.21, CE-KH 1.2, ACE 1.3,
# diff_react 1.9, incom_ns 3.1, com_ns 3.7) -- 23% of the compute. Weight is the measured
# headroom (error - floor), clipped to [1,3] repeats per cycle, so that budget moves to
# drivaernet 30.6 / Wave-Layer 26.8 / airfrans 23.6 / pdearena_ns 20.0 instead.
HEADROOM = {                          # r10 eval minus the measured box floor, in %
    "shallow_water": 0.0, "CE-KH": 1.1, "ACE": 1.2, "diff_react": 1.6, "incom_ns": 3.0,
    "com_ns": 3.7, "Poisson-Gauss": 4.2, "geofno_pipe": 4.1, "cfdbench": 6.4,
    "CE-Gauss": 5.6, "geofno_elasticity": 7.4, "pdebench3d_cns_Rand_M0.1_Eta1e-08_Zeta1e-08": 10.1,
    "CE-RM": 7.1, "CE-RP": 11.6, "NS-Gauss": 14.6, "pdearena_uncond": 13.3,
    "geofno_airfoil": 14.1, "pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08": 15.6,
    "pdebench3d_cns_Turb_M1.0_Eta1e-08_Zeta1e-08": 16.9, "CE-CRP": 12.6, "NS-Sines": 18.5,
    "shapenet_car": 10.4, "pdearena_ns": 11.8, "airfrans": 23.6, "Wave-Layer": 26.3,
    "drivaernet_pressure": 17.4,
}


# h5py IS NOT THREAD-SAFE, and the corpus adapters read Poseidon .nc through it for ten of the
# families. Once the batch builder moved onto a worker thread, that worker and the main thread's
# evaluate() could open HDF5 concurrently and deadlock inside the library — which is exactly what
# happened to r11: it survived the step-2000 eval (the worker happened to be blocked on a full
# queue) and hung at the step-4000 one. Every corpus read now takes this lock, so the prefetch
# overlap only ever applies between training steps, never against an eval.
_IO = threading.Lock()


def _grid(x, grid_dims, S):
    """(B,N,S) node values -> (B,*grid_dims,S). K-agnostic (was hard-wired to 2 axes)."""
    return tf.reshape(x, tf.concat([[tf.shape(x)[0]], list(grid_dims), [S]], 0))


# PER-FAMILY BOX SIDE (paper 2 design, fixed 2026-08-05 from the measured decode floors in
# paper2_universal/manuscript/supplement.tex). A floor is admissible at at most a third of the
# accuracy targeted for that family; the target is BCAT's published per-family value, not paper 1's.
#   coarse  R=32 / 16  -- floor <= 0.86%: transport- and relaxation-dominated families
#   fine    R=64 / 32  -- shock-carrying families, floor 1-2.6%
#   native  R=128      -- the two turbulence families, floor 0 by construction; they are the ones
#                         we are 2.7x off on and their floor falls only slowly with R
# Mean 2D cells (10*1024 + 4*4096 + 2*16384)/16 = 3712 = 0.91x the flat R=64 setting it replaces.
BOX_TABLE_2D = {
    "shallow_water": 32, "com_ns": 32, "incom_ns": 32, "ACE": 32, "Wave-Layer": 32,
    "CE-KH": 32, "CE-Gauss": 32, "NS-Gauss": 32, "Poisson-Gauss": 32, "diff_react": 32,
    "CE-RP": 64, "CE-CRP": 64, "CE-RM": 64, "NS-Sines": 64,
    "pdearena_ns": 128, "pdearena_uncond": 128,
    # Fig2 downstream: NS-PwC = incompressible transport with sharp vorticity interfaces
    # (NS-Sines class -> 64); GCE-RT = CE-type with RT plumes (CE-RP class -> 64).
    "NS-PwC": 64, "GCE-RT": 64, "PoolBoil-Sub": 64, "PoolBoil-SubX": 64,
}
BOX_TABLE_3D = {
    "pdebench3d_cns_Rand_M0.1_Eta1e-08_Zeta1e-08": 16,
    "pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08": 32,
    "pdebench3d_cns_Turb_M1.0_Eta1e-08_Zeta1e-08": 32,
}


def _box_dims_fn(a):
    """(K, family) -> latent box dims; the SDF volume is resampled to match.

    --box_table 1 uses the measured per-family allocation above and falls back to --dims2/--dims3
    for any family not listed (mesh/geometry families, and anything added later)."""
    def bd(K, fam=None):
        if a.box_table and fam is not None:
            r = (BOX_TABLE_2D if K == 2 else BOX_TABLE_3D).get(fam)
            if r:
                return (int(r),) * K
        return ((a.dims2,) * 2 if K == 2 else (a.dims3,) * 3)
    return bd


def _repeats(families, a):
    """Per-cycle repeat count from the measured headroom (1..max_rep, 1 if unknown)."""
    if not a.floor_weight:
        return [1] * len(families)
    hs = [HEADROOM.get(f, 10.0) for f in families]
    lo, hi = min(hs), max(hs)
    out = []
    for h in hs:
        t = 0.0 if hi <= lo else (h - lo) / (hi - lo)
        out.append(max(1, int(round(1 + t * (a.max_rep - 1)))))
    return out


def _interleave(families, split, a, seed, _bd):
    """Round-robin, each family cycling forever (v2 never-drop fix), `rep[i]` draws per cycle.

    IC-REUSE FOR THE 3D FAMILIES (MOTION-NCS, 2026-08-06). The pdebench3d builder is
    reader-bound: one gzip'd HDF5 trajectory costs ~3.65 s to load and an example uses only
    (enc_n + n_colloc) of its 2M cells — the measured 8-19 s/step of the 3D buckets is the
    reader, not the GPU (same diagnosis as the v2 combo pretrain, where one-read->many-examples
    took 18 -> 73 samples/s). Each 3D family therefore re-uses its last-read trajectory for
    `--reuse3d` consecutive CYCLES, drawing a fresh node/collocation subsample each time:
    the read cost is amortized 1/r while the family's one-example-per-cycle BALANCE and the
    epoch order are unchanged. Diversity cost is negligible at this corpus size (each 3D
    trajectory is revisited ~190 times over 160k steps anyway)."""
    from data.registry import FAMILIES as _F
    def mk(i, ep):
        rng = np.random.default_rng(seed + i + 9973 * ep)
        reuse = a.reuse3d if (families[i] in _F and _F[families[i]].K == 3) else 1
        def gen():
            n = 0
            it = iter(loader.units_for(families[i], split))
            s, left = None, 0
            while True:
                with _IO:                                   # corpus read + example build
                    if left <= 0:
                        try:
                            s = next(it)
                        except StopIteration:
                            return
                        left = max(1, reuse)
                    left -= 1
                    ex = make_example_ar(s, a.t_in, a.k_fut, a.k_relax, a.n_colloc, a.enc_n,
                                         rng, box_dims=_bd(int(s['K']), families[i]),
                                         t_in_min=a.t_in_min, ic_rand=a.ic_rand,
                                         phys_grid=(a.phys_grid if a.phys_w > 0 else 0))
                yield ex
                n += 1
                if a.max_per_fam and n >= a.max_per_fam:
                    return
        return gen()
    gens = [mk(i, 0) for i in range(len(families))]
    ep = [0] * len(families)
    rep = _repeats(families, a)
    print(f"[v3] per-cycle repeats: "
          f"{ {f: r for f, r in zip(families, rep) if r > 1} } (others 1)", flush=True)
    while True:
        for i in range(len(gens)):
            for _ in range(rep[i]):
                try:
                    yield next(gens[i])
                except StopIteration:
                    ep[i] += 1
                    gens[i] = mk(i, ep[i])
                    try:
                        yield next(gens[i])
                    except StopIteration:
                        break


def _prefetched(it, depth=8):
    """DISABLED BY DEFAULT — --prefetch is worth 10-18% and cost two runs.

    The idea was sound (the builder is 3.65 s/sample on pdebench3d_cns_Rand_M0.1 vs 0.04 s on the
    2D families, so those buckets idle the GPU), but moving corpus reads onto a worker thread hung
    both r11 and the bf16 gate: the worker stops inside the read, the main loop waits on q.get()
    forever, and the job stays "running". The adapters go through h5py for ten Poseidon families
    and h5py is not thread-safe. _IO serialises the reads, which removes the eval race but not the
    worker stall itself, so the flag stays off until the builder is made process-parallel instead.
    """
    import queue
    import threading
    q = queue.Queue(maxsize=depth)
    SENT = object()

    def work():
        try:
            for x in it:
                q.put(x)
        except Exception as e:                                    # surface it on the main thread
            q.put(e)
        q.put(SENT)

    threading.Thread(target=work, daemon=True).start()
    while True:
        x = q.get()
        if x is SENT:
            return
        if isinstance(x, Exception):
            raise x
        yield x


def _band_rel(pred_g, y_g, nb=6):
    """STAGE-4 (spectral grading, dense2d only): per-channel relative L2 in log-spaced radial
    wavenumber bands, averaged over admissible bands. Equalizes the phase/high-k content that an
    energy-weighted L2 barely sees. Bands whose GT energy is below 1e-5 of the total are skipped
    (the floor-law guard: never grade below the resolvable spectrum).
    pred_g, y_g: (B,H,W,S) grid fields -> (B,S)."""
    H, W = int(y_g.shape[1]), int(y_g.shape[2])
    P = tf.signal.fft2d(tf.cast(tf.transpose(pred_g, [0, 3, 1, 2]), tf.complex64))
    Y = tf.signal.fft2d(tf.cast(tf.transpose(y_g, [0, 3, 1, 2]), tf.complex64))
    E = tf.square(tf.abs(P - Y))                                    # (B,S,H,W)
    G = tf.square(tf.abs(Y))
    import numpy as _np
    ky = _np.minimum(_np.arange(H), H - _np.arange(H))
    kx = _np.minimum(_np.arange(W), W - _np.arange(W))
    kr = _np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    edges = _np.unique(_np.round(_np.exp(_np.linspace(0, _np.log(max(H, W) / 2), nb + 1))))
    tot = tf.reduce_sum(G, axis=[2, 3]) + 1e-12                     # (B,S)
    rels, ws = [], []
    for i in range(len(edges) - 1):
        m = tf.constant(((kr >= edges[i]) & (kr < edges[i + 1])).astype("float32"))
        num_b = tf.reduce_sum(E * m, axis=[2, 3])
        den_b = tf.reduce_sum(G * m, axis=[2, 3])
        w = tf.cast(den_b > 1e-5 * tot, tf.float32)                 # admissible-band gate
        rels.append(w * tf.sqrt(num_b / (den_b + 1e-12)))
        ws.append(w)
    R = tf.add_n(rels) / tf.maximum(tf.add_n(ws), 1.0)              # (B,S)
    return R


def _rel(pred, y, cm, per_channel, scale=None, fluct=False, den_override=None):
    """Masked relative L2. per_channel=False reproduces the PROSE/BCAT convention (one norm over
    points AND channels) — keep it for EVAL so our numbers stay comparable to the literature.

    per_channel=True normalizes EACH active slot separately and averages. Needed for TRAINING:
    the loader keeps the DC component (std-only norm), so a slot with a large mean — com_ns
    density, ~mean^2+1 — dominates the joint denominator and down-weights the near-zero-mean
    velocity slots by ~10x. That is why com_ns reads 1.5% jointly while its velocity channel sits
    at 66%, and why com_ns was the ONE family frozen through the bylfa convergence extension."""
    if scale is not None:
        # PHYSICAL-SPACE scoring. The loader divides each slot by its own std, and a
        # channel-JOINT rel-L2 is NOT invariant to that rescaling: it reweights the channels.
        # cfdbench is the extreme case — physically vx (inlet flow) dwarfs vy, so in physical
        # units the easy channel dominates the denominator, while in normalized units the hard
        # cross-flow channel gets equal footing. That difference alone is worth several factors,
        # and it is why our numbers cannot be laid next to PROSE-exact/BCAT ones until we undo it.
        sc = scale[:, None, :]
        pred, y = pred * sc, y * sc
    # FLUCTUATION-relative denominator (VRMSE-aligned, train-only): the loader keeps the DC
    # component, so ||y|| is DC-dominated for high-mean slots (rho ~10 sigma, p ~6 sigma) and
    # their fluctuation errors are down-weighted ~10x. Subtracting the per-sample spatial mean
    # from the DENOMINATOR only (numerator still penalizes DC errors) grades every slot on its
    # fluctuation scale.
    #
    # THIS IS THE VRMSE-ALIGNED OBJECTIVE, exactly. Walrus Appendix F.1.1 defines
    #   VRMSE = <|u-v|^2>^(1/2) / (<|u-ubar|^2>+eps)^(1/2),  eps=1e-7, ubar = SPATIAL mean,
    # i.e. the denominator is the per-sample spatial std of the truth -- which is what the line
    # below builds. So --fluct_loss 1 is the metric written as a loss; nothing further is needed
    # to "align with Walrus".
    #
    # DO NOT cite Walrus's RMS(dU) as a loss convention (an earlier version of this comment did,
    # and --fluct_loss 2 was built on that misreading). Walrus p3, "Asymmetric Input/Output
    # Normalization", normalizes INPUTS by RMS_(TimexSpace)(U_t) over the PROVIDED HISTORY and
    # de-normalizes OUTPUTS by RMS_(TimexSpace)(dU_t). That is a per-sample scaling INSIDE the
    # model, computed from the observed window -- not a loss denominator, not a dataset-level
    # statistic, and not the target's frame-to-frame increment. Replicating it means changing the
    # output stage, not this function.
    y_den = y - tf.reduce_mean(y, axis=1, keepdims=True) if fluct else y
    if not per_channel:
        w = cm[:, None, :]
        num = tf.sqrt(tf.reduce_sum(w * tf.square(pred - y), axis=[1, 2]) + 1e-12)
        den = tf.sqrt(tf.reduce_sum(w * tf.square(y_den), axis=[1, 2]) + 1e-12)
        return num / den
    num = tf.sqrt(tf.reduce_sum(tf.square(pred - y), axis=1) + 1e-12)      # (B,S) per slot
    den = tf.sqrt(tf.reduce_sum(tf.square(y_den), axis=1) + 1e-12)
    if den_override is not None:
        # STAGE-3 (tendency grading): denominator = ||y_k - y_{k-1}||, i.e. the size of the
        # GT change this segment. Floored against near-static frames.
        den = tf.maximum(den_override, 1e-3 * tf.sqrt(tf.reduce_sum(tf.square(y), axis=1) + 1e-12))
    if fluct:
        # guard: a spatially near-constant active slot has ~zero fluctuation norm; floor the
        # denominator at 1e-3 of the DC-inclusive norm so its rel term saturates via clamp
        # instead of exploding.
        den = tf.maximum(den, 1e-3 * tf.sqrt(tf.reduce_sum(tf.square(y), axis=1) + 1e-12))
    r = cm * (num / den)                                                    # inactive slots -> 0
    return tf.reduce_sum(r, -1) / tf.maximum(tf.reduce_sum(cm, -1), 1.0)


def singleshot_loss(model, cn, xw, fm, gn, yn, icq, cq, gq, yq, tsm, cm, fbm, op, cond, fid,
                    sdb, K, roles, k_seg, box_dims=None, clamp=4.0, per_channel=False,
                    scale=None, fluct=False):
    """SINGLE-SHOT mode: one trunk call decodes every future frame (model.step_ss); the loss is
    the same masked mean over valid frames as the AR path, so the two modes score identically.
    scale != None -> returns tf.stack([normalized, physical]) from ONE forward (eval use)."""
    qall = tf.concat([cn, cq], 1)
    gall = tf.concat([gn, gq], 1)
    u_prev = tf.concat([xw[:, -1], icq], 1)                       # anchor at nodes + colloc
    preds = model.step_ss(cn, xw, fm, gn, K, roles, cond, op, qall, gall, u_prev,
                          kf=k_seg, fam_id=fid, sdf_box=sdb, box_dims=box_dims)  # (B,KF,N+Q,S)
    total = tf.constant(0.0)
    total_p = tf.constant(0.0)
    total_f = tf.constant(0.0)
    wsum = tf.constant(0.0)
    for k in range(k_seg):
        y_all = tf.concat([yn[:, k], yq[:, k]], 1)
        if int(fluct) >= 2:
            y_prev_all = (u_prev if k == 0
                          else tf.concat([yn[:, k - 1], yq[:, k - 1]], 1))
            dov = tf.sqrt(tf.reduce_sum(tf.square(y_all - y_prev_all), axis=1) + 1e-12)
            rel = tf.minimum(_rel(preds[:, k], y_all, cm, per_channel, den_override=dov), clamp)
        else:
            rel = tf.minimum(_rel(preds[:, k], y_all, cm, per_channel, fluct=bool(fluct)), clamp)
        total += tf.reduce_sum(tsm[:, k] * rel)
        if scale is not None:
            rel_p = tf.minimum(_rel(preds[:, k], y_all, cm, per_channel, scale), clamp)
            total_p += tf.reduce_sum(tsm[:, k] * rel_p)
            # FLUCT eval track: per-channel, mean-removed denominator (scale-invariant per
            # slot, so normalized == physical). The z-score/VRMSE-aligned convention.
            rel_f = tf.minimum(_rel(preds[:, k], y_all, cm, True, fluct=True), clamp)
            total_f += tf.reduce_sum(tsm[:, k] * rel_f)
        wsum += tf.reduce_sum(tsm[:, k])
    w = tf.maximum(wsum, 1e-6)
    if scale is not None:
        return tf.stack([total / w, total_p / w, total_f / w])
    return total / w


def rollout_loss(model, cn, xw, fm, gn, yn, icq, cq, gq, yq, tsm, cm, fbm, op, cond, fid, sdb,
                 K, steady, dense2d, roles, k_seg, grid_dims, box_dims=None, surf=False,
                 clamp=4.0, per_channel=False, scale=None):
    """Free-running AR rollout; returns the ar_fair loss (one masked mean over all
    segments' valid frames). Python-static args (K, steady, dense2d, roles, k_seg,
    grid_dims) select the trace; tensors carry the batch."""
    B = tf.shape(cn)[0]
    Nn = tf.shape(cn)[1]
    qall = tf.concat([cn, cq], 1)
    gall = tf.concat([gn, gq], 1)
    u_prev = tf.concat([xw[:, -1], icq], 1)                     # (B,N+Q,S); steady -> zeros
    win, fmask = xw, fm
    prev_grid = prev2_grid = None
    if dense2d:
        prev_grid = _grid(xw[:, -1], grid_dims, model.S)
        # ITEM H: the frame BEFORE the anchor, so that du/dt enters the native dictionary and a
        # one-frame re-anchor is a complete state for second-order-in-time physics. xw is right
        # aligned and fm marks validity, so a window of length one leaves this None (dt slot zero).
        if int(xw.shape[1]) >= 2:
            prev2_grid = _grid(xw[:, -2], grid_dims, model.S)
    total = tf.constant(0.0)
    total_p = tf.constant(0.0)
    total_f = tf.constant(0.0)
    wsum = tf.constant(0.0)
    for k in range(k_seg):
        pred = model.step(cn, win, fmask, gn, K, roles, cond, op, qall, gall, u_prev,
                          steady=steady, prev_grid=prev_grid, grid_dims=grid_dims,
                          fam_id=fid, sdf_box=sdb, box_dims=box_dims, fbmask=fbm, surf=surf,
                          prev2_grid=prev2_grid)
        kt = 0 if steady else k
        y_all = tf.concat([yn[:, kt], yq[:, kt]], 1)
        rel = tf.minimum(_rel(pred, y_all, cm, per_channel), clamp)         # (B,) normalized
        rel_p = (tf.minimum(_rel(pred, y_all, cm, per_channel, scale), clamp)
                 if scale is not None else None)                            # physical (PROSE conv)
        rel_f = (tf.minimum(_rel(pred, y_all, cm, True, fluct=True), clamp)
                 if scale is not None else None)                            # FLUCT (z-score conv)
        if steady:
            wk = float(k + 1)                                    # later relaxation steps count more
            total += wk * tf.reduce_sum(rel)
            if rel_p is not None:
                total_p += wk * tf.reduce_sum(rel_p)
                total_f += wk * tf.reduce_sum(rel_f)
            wsum += wk * tf.cast(B, tf.float32)
        else:
            total += tf.reduce_sum(tsm[:, k] * rel)              # frame-validity weighted
            if rel_p is not None:
                total_p += tf.reduce_sum(tsm[:, k] * rel_p)
                total_f += tf.reduce_sum(tsm[:, k] * rel_f)
            wsum += tf.reduce_sum(tsm[:, k])
        u_prev = tf.stop_gradient(pred)                          # free-running feedback
        pred_n = u_prev[:, :Nn]
        win = tf.concat([win[:, 1:], pred_n[:, None]], 1)
        fmask = tf.concat([fmask[:, 1:], tf.ones_like(fmask[:, :1])], 1)
        if dense2d:
            prev2_grid, prev_grid = prev_grid, _grid(pred_n, grid_dims, model.S)
    w = tf.maximum(wsum, 1e-6)
    if scale is not None:
        return tf.stack([total / w, total_p / w, total_f / w])
    return total / w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="pretrain")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--batch3d", type=int, default=1)            # 3D buckets: K x activations
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--d_w", type=int, default=32)
    ap.add_argument("--n_p", type=int, default=8)
    ap.add_argument("--d_cond", type=int, default=128)
    ap.add_argument("--dims2", type=int, default=64)
    ap.add_argument("--box_table", type=int, default=0)     # per-family R (paper 2 design table)
    # ITEM G / H (core/dictdec.py): the box carries coefficients that multiply the NATIVE anchor
    # dictionary. dict_shift adds the finite-width shift slots the shock families need. The slot
    # layout has to be global because the coefficient head is one weight set; families that do not
    # need a slot simply learn a zero coefficient for it.
    ap.add_argument("--dict_decode", type=int, default=0)   # 0 = the pre-2026-08-05 decode,
    ap.add_argument("--dict_shift", type=int, default=0)     # bit-identical (no dict_c variable)
    ap.add_argument("--dims3", type=int, default=32)
    ap.add_argument("--t_in", type=int, default=10)
    ap.add_argument("--t_in_min", type=int, default=0)   # >0: sample history length in [min, t_in]
    ap.add_argument("--ic_rand", type=int, default=0)    # 1: randomize the IC's temporal position (train only)
    ap.add_argument("--k_fut", type=int, default=10)             # full AR (bylfa)
    ap.add_argument("--k_relax", type=int, default=4)            # steady relaxation steps
    ap.add_argument("--enc_n", type=int, default=8192)
    ap.add_argument("--n_colloc", type=int, default=2048)
    ap.add_argument("--transport", type=int, default=1)
    ap.add_argument("--surf_attn", type=int, default=0)          # slice-attention M (0 = off)
    ap.add_argument("--moe_experts", type=int, default=0)        # >0: block MLP -> E experts, top-k
    ap.add_argument("--moe_topk", type=int, default=2)
    ap.add_argument("--moe_capacity", type=float, default=0.0)   # >0: sparse dispatch (see axops)
    ap.add_argument("--moe_aux_w", type=float, default=0.0)      # load-balance loss weight (0.01)
    ap.add_argument("--seg_bwd", type=int, default=0)            # backward on N of k_seg segments
    ap.add_argument("--d_shape", type=int, default=0)            # >0: global shape code from sdf_box
    ap.add_argument("--gate_cond", type=int, default=0)          # symbolic cond -> mechanism gates
    ap.add_argument("--gate_experts", type=int, default=0)       # per-cell mixture of gate profiles
    # MOTION-NCS flags (2026-08-06, motion_ncs/PLAN_NCS.md)
    ap.add_argument("--fam_head", type=int, default=0)       # 1 = UNSHARED per-family output head
    # Sparse-support gate boost (item B device ii): the vortex-stretching feature is non-zero in
    # only 3 of 19 families, so its gate sees gradient in ~3/19 of the steps; scaling that
    # gradient by ~19/3 restores the same effective gate learning rate every other bank enjoys —
    # the fix for the measured boundary-layer-gate failure (dL/dgate = 0 on most steps).
    ap.add_argument("--vs_gate_boost", type=float, default=6.0)
    ap.add_argument("--reuse3d", type=int, default=8)    # 3D reader amortization (see _interleave)
    ap.add_argument("--moe_upcycle", type=int, default=0)        # seed every expert from the dense MLP
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--grad_accum", type=int, default=1)   # sum grads over N consecutive
                                                           # (different-family) steps per update
    ap.add_argument("--clamp", type=float, default=4.0)
    ap.add_argument("--per_channel_loss", type=int, default=1)   # train: balance slots; eval stays joint
    ap.add_argument("--fluct_loss", type=int, default=0)         # train: mean-removed (VRMSE-aligned) denominators
    ap.add_argument("--xla", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_per_fam", type=int, default=0)
    ap.add_argument("--floor_weight", type=int, default=0)   # sample by measured headroom
    ap.add_argument("--max_rep", type=int, default=3)        # max draws/cycle for the worst family
    ap.add_argument("--prefetch", type=int, default=0)       # batch-builder queue depth (0 = off)
    ap.add_argument("--bf16", type=int, default=0)           # mixed_bfloat16 (bylfa's policy)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--eval_n", type=int, default=8)
    ap.add_argument("--ckpt_every", type=int, default=2000)
    ap.add_argument("--phys_w", type=float, default=0.0)  # analytic continuity-residual loss (3D)
    ap.add_argument("--phys_diag", type=int, default=0)   # 1 = diagnose only (no gradient apply)
    ap.add_argument("--phys_c", type=float, default=0.0)  # >0: fixed GT-calibrated unit ratio
    ap.add_argument("--phys_distill", type=int, default=0)  # 1: distill corrected teacher (arm-3)
    ap.add_argument("--phys_alpha", type=float, default=0.15)  # teacher correction strength
    ap.add_argument("--phys_grid", type=int, default=16)  # FD lattice per axis (queries prepended)
    ap.add_argument("--init_ckpt", default="")
    ap.add_argument("--save", default="/corpus/results/v3ar_r1")
    a = ap.parse_args()
    os.makedirs(a.save, exist_ok=True)

    if a.bf16:
        # bylfa ran mixed_bfloat16 on this same cluster (train_prose_mgpu.py:88) and it is worth
        # ~2x on H100. Kept OFF by default here and NOT enabled together with the resolution /
        # MoE-dispatch change: this file does a lot of manual tensor arithmetic (interp_box, the
        # tendency banks' FFT path, the surface reads), where a policy change turns silent
        # float32/bfloat16 operand mismatches into hard errors that would be unattributable if
        # they landed in the same run as everything else. Flip it on its own, against a gate job.
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
        print("[v3] mixed_bfloat16 policy enabled", flush=True)

    fams = (motion_families() if a.families == "motion"
            else pretrain_families() if a.families == "pretrain"
            else a.families.split(","))
    fams = [f.name if hasattr(f, "name") else f for f in fams]
    ok = []
    for f in fams:                                            # skip families with no split manifest
        try:
            from data.splits import load_manifest
            load_manifest(f)
            ok.append(f)
        except Exception as e:
            print(f"[v3] skip {f}: no manifest ({e})", flush=True)
    fams = ok
    print(f"[v3] families({len(fams)}): {fams}", flush=True)

    gpus = tf.config.list_physical_devices("GPU")
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.ReductionToOneDevice()) if len(gpus) > 1 else None
    n_rep = strategy.num_replicas_in_sync if strategy else 1
    print(f"[v3] replicas: {n_rep}", flush=True)

    def scope():                                              # fresh CM per use (no re-enter)
        return strategy.scope() if strategy else _null()
    from data.registry import FAMILIES as _FAM
    with scope():
        model = V3Model(S=NUM_SLOTS, d=a.d, depth=a.depth, d_w=a.d_w, n_p=a.n_p, d_cond=a.d_cond,
                        dims2=(a.dims2,) * 2, dims3=(a.dims3,) * 3, t_in=a.t_in,
                        transport=bool(a.transport), n_fams=len(_FAM),
                        n_slices=a.surf_attn, moe_experts=a.moe_experts,
                        moe_topk=a.moe_topk, moe_capacity=a.moe_capacity,
                        gate_cond=a.gate_cond, gate_experts=a.gate_experts,
                        n_semantic=SEMANTIC_SLOTS, d_shape=a.d_shape,
                        dict_decode=a.dict_decode, dict_shift=a.dict_shift,
                        fam_head=a.fam_head)
        opt = tf.keras.optimizers.Adam(_WarmupConst(a.lr, a.warmup), clipnorm=1.0)

    def batch_for(key):
        return a.batch3d * n_rep if key[0] == 3 else a.batch * n_rep

    _bd = _box_dims_fn(a)

    def batches(split, seed):
        it = _interleave(fams, split, a, seed, _bd)
        if a.prefetch:
            it = _prefetched(it, a.prefetch)
        return bucketed(it, batch_for)

    # ------------------------------------------------------------------ build/warm-start
    def _tensors(b):
        keys = ("coords_node", "x_win", "fmask", "geom_node", "y_node", "ic_q",
                "coords_q", "geom_q", "y_q", "tstep_mask", "cmask", "fbmask", "op_multihot",
                "cond", "fam_id", "sdf_box")
        return tuple(tf.constant(b[k]) for k in keys)

    def _static(b):
        gd = tuple(b.get("dims", ())) or None
        return (int(b["K"]), bool(b["steady"]), bool(b["dense2d"]),
                tuple(int(r) for r in b["roles"]), int(b["k_seg"]), gd,
                tuple(int(v) for v in b["box_dims"]), bool(b["surf"]))

    def _mode(b):
        return b.get("mode", "steady" if b["steady"] else "ar")

    print("[v3] building variables (warm_build both K + transport)...", flush=True)
    with scope():
        model.warm_build()
    V = model.trainable_variables
    NAMES = [v.name for v in V]
    assert len(set(NAMES)) == len(NAMES), "variable names must be unique for named ckpts"
    print(f"[v3] {len(V)} vars, {sum(int(np.prod(v.shape)) for v in V)/1e6:.1f}M params",
          flush=True)
    # item B device (ii): support-normalized gate learning rate for the sparse-support vortex
    # bank — a plain python multiplier list, applied to the summed gradients before every
    # apply/accumulate, so it costs nothing inside the graph.
    VS_BOOST = [a.vs_gate_boost if ("gate_vortex" in v.name or "gcond_vortex" in v.name)
                else 1.0 for v in V]
    n_boost = sum(1 for x in VS_BOOST if x != 1.0)
    print(f"[v3] vortex gate-grad boost x{a.vs_gate_boost} on {n_boost} vars", flush=True)

    def _vsboost(gs):
        return [g if b == 1.0 else g * b for g, b in zip(gs, VS_BOOST)]
    if a.init_ckpt and a.moe_upcycle:
        # SPARSE UPCYCLING (Komatsuzaki et al. 2023): every expert is seeded from the ONE dense
        # MLP the checkpoint holds, plus tiny noise to break the symmetry the router would
        # otherwise never resolve. With a near-zero router the top-k experts average back to the
        # dense block, so the model's function is preserved at load and only capacity grows.
        # Expert tensors are named `..._e{e}_...`; stripping that infix gives the dense source.
        import re as _re
        ck = np.load(a.init_ckpt)
        have = set(ck.files)
        ld = up = nw = 0
        for v in V:
            src = _re.sub(r"_e\d+", "", v.name)
            if src in have and tuple(ck[src].shape) == tuple(v.shape):
                w = ck[src]
                if src != v.name:                       # an expert copy of the dense MLP
                    w = w + np.random.normal(0, 1e-3, w.shape).astype(w.dtype)
                    up += 1
                else:
                    ld += 1
                v.assign(w)
            else:
                nw += 1
        print(f"[upcycle] dense-loaded {ld}, expert-copied {up}, new-at-init {nw} "
              f"(router + anything absent from the dense ckpt)", flush=True)
    elif a.init_ckpt:
        # SURFACE-PATH WARM-START GUARD. Two things go wrong when the manifold path changes shape
        # or gains layers, and both were measured on r5's first eval (Poisson-Gauss 6.8 -> 124.7%
        # while every other family was fine):
        #  1. `_try_np_pad` treats ANY widened trailing axis divisible by d_w as an n_p-panel
        #     growth, so the slice-ASSIGNMENT matrix (d, M) 64->512 got zero-padded: the 448 new
        #     columns share a tied zero logit and swallow most of the softmax mass, destroying the
        #     assignment. These variables must come back fresh, not padded.
        #  2. sf_out is only inert while it is ZERO. Loading a partially-trained sf_out next to a
        #     freshly-random sf_k/sf_v injects noise straight into the decode at step 0.
        pristine = {v.name: v.numpy().copy() for v in V} if a.surf_attn else {}
        load_ckpt(V, a.init_ckpt, d_w=a.d_w, names=NAMES)
        if a.surf_attn:
            # Only intervene where the checkpoint could NOT supply the variable correctly, i.e.
            # where its stored shape differs (the slice count M changed) or it is absent. Resetting
            # unconditionally would throw away a trained surface path every time any other part of
            # the model changes — which is exactly what happened on the first attempt here.
            ck = np.load(a.init_ckpt)
            bad = {v.name for v in V
                   if v.name not in ck.files or tuple(ck[v.name].shape) != tuple(v.shape)}
            n_re = n_z = 0
            for v in V:
                if v.name in bad and any(s_ in v.name for s_ in
                                         ("enc_slw", "enc_slgate", "sf_k", "sf_v", "sf_beta")):
                    v.assign(pristine[v.name]); n_re += 1
            if n_re:                       # the token set changed -> re-arm the zero-init output
                for v in V:                # so the path is inert until it has relearned
                    if "sf_out" in v.name:
                        v.assign(tf.zeros_like(v)); n_z += 1
            print(f"[surface] shape-changed slice vars re-initialised: {n_re}, "
                  f"output projections zeroed: {n_z} "
                  f"({'path inert at load' if n_re else 'trained surface path carried over intact'})",
                  flush=True)

    # ------------------------------------------------------------------ train step
    # PER-SEGMENT gradient accumulation. The AR chain has stop-gradient boundaries, so each
    # segment's loss depends only on its OWN forward pass: summing per-segment gradients equals
    # the full-rollout gradient EXACTLY, at 1/k_seg the activation memory (the full unrolled
    # backward OOMed the 3D bucket on 80GB). XLA compiles one segment (same trace reused k_seg
    # times); apply_gradients stays OUTSIDE the jitted scope so the cross-replica reduction goes
    # through ReductionToOneDevice, not XLA's broken-in-container gpu.all_reduce.
    @tf.function(jit_compile=bool(a.xla), reduce_retracing=True)
    def _seg_fwd_bwd(cn, gn, win, fmask, u_prev, prev_grid, prev2_grid, y_all, y_prev_all,
                     w_k, wsum_inv,
                     cm, fbm, op, cond, fid, sdb, qall, gall, K, steady, dense2d, roles,
                     grid_dims, box_dims, surf):
        with tf.GradientTape() as tape:
            pred = model.step(cn, win, fmask, gn, K, roles, cond, op, qall, gall, u_prev,
                              steady=steady, prev_grid=prev_grid, grid_dims=grid_dims,
                              fam_id=fid, sdf_box=sdb, box_dims=box_dims, fbmask=fbm, surf=surf,
                              prev2_grid=prev2_grid)
            if int(a.fluct_loss) >= 2 and not steady:
                dov = tf.sqrt(tf.reduce_sum(tf.square(y_all - y_prev_all), axis=1) + 1e-12)
                rel = tf.minimum(_rel(pred, y_all, cm, bool(a.per_channel_loss),
                                      den_override=dov), a.clamp)
                if int(a.fluct_loss) == 3 and dense2d:
                    Nn_ = tf.shape(cn)[1]
                    pg = _grid(pred[:, :Nn_], grid_dims, model.S)
                    yg = _grid(y_all[:, :Nn_], grid_dims, model.S)
                    br = tf.minimum(_band_rel(pg, yg), a.clamp)     # (B,S)
                    brm = tf.reduce_sum(cm * br, -1) / tf.maximum(tf.reduce_sum(cm, -1), 1.0)
                    rel = 0.5 * rel + 0.5 * brm
            else:
                rel = tf.minimum(_rel(pred, y_all, cm, bool(a.per_channel_loss),
                                      fluct=bool(a.fluct_loss)), a.clamp)
            L = tf.reduce_sum(w_k * rel) * wsum_inv
            if a.moe_aux_w:                       # router anti-imbalance (Switch); see model.moe_aux
                L = L + a.moe_aux_w * model.moe_aux() * wsum_inv * tf.reduce_sum(w_k)
        g = tape.gradient(L, V)
        g = [gi if gi is not None else tf.zeros_like(v) for gi, v in zip(g, V)]
        return L, g, pred

    @tf.function(jit_compile=bool(a.xla), reduce_retracing=True)
    def _seg_fwd(cn, gn, win, fmask, u_prev, prev_grid, prev2_grid, cm, fbm, op, cond, fid, sdb,
                 qall, gall, K, steady, dense2d, roles, grid_dims, box_dims, surf):
        """Advance one AR segment with NO tape. The rollout has to pass through every segment to
        reach the later ones, but only the segments we back-propagate need their activations kept,
        and the backward is ~2x the forward — so skipping it is where the time is."""
        return model.step(cn, win, fmask, gn, K, roles, cond, op, qall, gall, u_prev,
                          steady=steady, prev_grid=prev_grid, grid_dims=grid_dims,
                          fam_id=fid, sdf_box=sdb, box_dims=box_dims, fbmask=fbm, surf=surf,
                          prev2_grid=prev2_grid)

    # Gradient ACCUMULATION ACROSS FAMILIES. A batch is bucketed by (K, steady, dense2d, N), so a
    # single step only ever sees ONE family and the update direction swings between 26 of them.
    # The 6-family model this line descends from mixed families inside every batch; here we
    # recover the same thing by summing the gradients of `grad_accum` consecutive steps — which
    # land on different families by construction (round-robin interleave) — before applying once.
    # Costs no extra memory: the per-segment gradients are already materialised one at a time.
    with scope():
        # Build the optimizer slots HERE, in the strategy scope and outside any tf.function.
        # Left to be created lazily on the first apply_gradients they land on whatever device the
        # tracing context happens to be on, and MirroredStrategy then refuses to read them
        # ("Trying to access resource Adam/m/... on a different device").
        opt.build(V)
        accum = ([tf.Variable(tf.zeros_like(v), trainable=False, name=f"acc{i}")
                  for i, v in enumerate(V)] if a.grad_accum > 1 else [])

    @tf.function(reduce_retracing=True)
    def _grad_step(cn, xw, fm, gn, yn, icq, cq, gq, yq, tsm, cm, fbm, op, cond, fid, sdb,
                   K, steady, dense2d, roles, k_seg, grid_dims, box_dims, surf,
                   bwd0=0, nbwd=0):
        B = tf.shape(cn)[0]
        Nn = tf.shape(cn)[1]
        qall = tf.concat([cn, cq], 1)
        gall = tf.concat([gn, gq], 1)
        u_prev = tf.concat([xw[:, -1], icq], 1)
        win, fmask = xw, fm
        prev_grid = _grid(xw[:, -1], grid_dims, model.S) if dense2d else None
        # ITEM H: two-frame anchor (see the note in the eval rollout above).
        prev2_grid = (_grid(xw[:, -2], grid_dims, model.S)
                      if dense2d and int(xw.shape[1]) >= 2 else None)
        Bf = tf.cast(B, tf.float32)
        # SEGMENT-SUBSAMPLED BACKWARD. Cost is linear in the number of segments we differentiate,
        # and `ar_fair` is a mean over segments — so differentiating a rotating WINDOW of them each
        # step is an unbiased estimate of the full-rollout gradient (normalised by the window's own
        # weight, not the total) at 10F+nB instead of 10F+10B. Truncating the HORIZON instead would
        # be cheaper still and wrong: the eval is a 10-step free-running rollout, so the later
        # segments are exactly what we are scored on and must keep receiving gradient.
        seg = list(range(k_seg)) if nbwd <= 0 else [
            k for k in range(k_seg) if bwd0 <= k < bwd0 + nbwd]
        if steady:
            wsum = Bf * float(sum(k + 1 for k in seg))
        else:
            wsum = tf.reduce_sum(tf.gather(tsm, seg, axis=1))
        wsum_inv = 1.0 / tf.maximum(wsum, 1e-6)
        acc = [tf.zeros_like(v) for v in V]
        total = tf.constant(0.0)
        for k in range(k_seg):
            kt = 0 if steady else k
            y_all = tf.concat([yn[:, kt], yq[:, kt]], 1)
            y_prev_all = (tf.concat([xw[:, -1], icq], 1) if kt == 0
                          else tf.concat([yn[:, kt - 1], yq[:, kt - 1]], 1))
            w_k = (float(k + 1) * tf.ones([B]) if steady else tsm[:, k])
            if k in seg:
                L, g, pred = _seg_fwd_bwd(cn, gn, win, fmask, u_prev, prev_grid, prev2_grid,
                                          y_all, y_prev_all, w_k, wsum_inv, cm, fbm, op, cond, fid, sdb,
                                          qall, gall, K, steady, dense2d, roles, grid_dims,
                                          box_dims, surf)
                acc = [ai + gi for ai, gi in zip(acc, g)]
                total += L
            else:
                pred = _seg_fwd(cn, gn, win, fmask, u_prev, prev_grid, prev2_grid, cm, fbm, op,
                                cond, fid, sdb, qall, gall, K, steady, dense2d, roles, grid_dims,
                                box_dims, surf)
            u_prev = pred                                       # outside any tape: free-running
            pred_n = pred[:, :Nn]
            win = tf.concat([win[:, 1:], pred_n[:, None]], 1)
            fmask = tf.concat([fmask[:, 1:], tf.ones_like(fmask[:, :1])], 1)
            if dense2d:
                prev2_grid, prev_grid = prev_grid, _grid(pred_n, grid_dims, model.S)
        acc = _vsboost(acc)                       # sparse-support gate LR (item B device ii)
        if a.grad_accum > 1:
            for s_, g_ in zip(accum, acc):        # replica-local add; the APPLY happens outside
                s_.assign_add(g_)                 # (apply_gradients under strategy.run + tf.cond
        else:                                     #  triggers a merge_call error)
            opt.apply_gradients(zip(acc, V))
        return total

    # ------------------------------------------------------------------ single-shot mode
    # (MOTION-NCS): the smooth families' whole training step is ONE trunk evaluation, so the
    # forward+backward fits in a single jitted function — no per-segment accumulation needed.
    @tf.function(jit_compile=bool(a.xla), reduce_retracing=True)
    def _ss_fwd_bwd(cn, xw, fm, gn, yn, icq, cq, gq, yq, tsm, cm, fbm, op, cond, fid, sdb,
                    K, roles, k_seg, box_dims):
        with tf.GradientTape() as tape:
            L = singleshot_loss(model, cn, xw, fm, gn, yn, icq, cq, gq, yq, tsm, cm, fbm,
                                op, cond, fid, sdb, K, roles, k_seg, box_dims,
                                clamp=a.clamp, per_channel=bool(a.per_channel_loss),
                                fluct=int(a.fluct_loss))
        g = tape.gradient(L, V)
        g = [gi if gi is not None else tf.zeros_like(v) for gi, v in zip(g, V)]
        return L, g

    @tf.function(reduce_retracing=True)
    def _ss_grad_step(cn, xw, fm, gn, yn, icq, cq, gq, yq, tsm, cm, fbm, op, cond, fid, sdb,
                      K, steady, dense2d, roles, k_seg, grid_dims, box_dims, surf,
                      bwd0=0, nbwd=0):
        L, g = _ss_fwd_bwd(cn, xw, fm, gn, yn, icq, cq, gq, yq, tsm, cm, fbm, op, cond,
                           fid, sdb, K, roles, k_seg, box_dims)
        g = _vsboost(g)
        if a.grad_accum > 1:
            for s_, g_ in zip(accum, g):
                s_.assign_add(g_)
        else:
            opt.apply_gradients(zip(g, V))
        return L

    # EVAL runs in BOTH spaces from one forward: normalized (v2-wall continuity, the 18.09 bar)
    # and PHYSICAL (denormalized joint rel-L2 = the PROSE/BCAT reference convention — the loader's
    # per-slot std is undone via `scale` before the norm, since a channel-joint rel-L2 is NOT
    # invariant to per-channel rescaling). Literature comparison uses the physical number.
    @tf.function(reduce_retracing=True)
    def _fwd_loss(cn, xw, fm, gn, yn, icq, cq, gq, yq, tsm, cm, fbm, op, cond, fid, sdb, sc,
                  K, steady, dense2d, roles, k_seg, grid_dims, box_dims, surf):
        return rollout_loss(model, cn, xw, fm, gn, yn, icq, cq, gq, yq, tsm, cm, fbm, op,
                            cond, fid, sdb, K, steady, dense2d, roles, k_seg, grid_dims,
                            box_dims, surf, clamp=1e9, scale=sc)

    @tf.function(reduce_retracing=True)
    def _fwd_loss_ss(cn, xw, fm, gn, yn, icq, cq, gq, yq, tsm, cm, fbm, op, cond, fid, sdb, sc,
                     K, steady, dense2d, roles, k_seg, grid_dims, box_dims, surf):
        return singleshot_loss(model, cn, xw, fm, gn, yn, icq, cq, gq, yq, tsm, cm, fbm,
                               op, cond, fid, sdb, K, roles, k_seg, box_dims, clamp=1e9,
                               scale=sc)

    def _split(t, ctx):
        per = tf.shape(t)[0] // n_rep
        i = ctx.replica_id_in_sync_group
        return t[i * per:(i + 1) * per]

    @tf.function(reduce_retracing=True)
    def _dist_step(dts, K, steady, dense2d, roles, k_seg, grid_dims, box_dims, surf,
                   bwd0, nbwd):
        pr = strategy.run(_grad_step,
                          args=(*dts, K, steady, dense2d, roles, k_seg, grid_dims, box_dims,
                                surf, bwd0, nbwd))
        return strategy.reduce(tf.distribute.ReduceOp.MEAN, pr, axis=None)

    @tf.function(reduce_retracing=True)
    def _dist_step_ss(dts, K, steady, dense2d, roles, k_seg, grid_dims, box_dims, surf,
                      bwd0, nbwd):
        pr = strategy.run(_ss_grad_step,
                          args=(*dts, K, steady, dense2d, roles, k_seg, grid_dims, box_dims,
                                surf, bwd0, nbwd))
        return strategy.reduce(tf.distribute.ReduceOp.MEAN, pr, axis=None)

    def _apply_accum(n):        # eager: once per grad_accum steps, cost is negligible
        opt.apply_gradients((s_ / n, v) for s_, v in zip(accum, V))
        for s_ in accum:
            s_.assign(tf.zeros_like(s_))

    def run_batch(b, upd=0):
        ts = _tensors(b)
        st = _static(b)
        ss = _mode(b) == "ss"
        k_seg = st[4]
        n = (min(a.seg_bwd, k_seg) if a.seg_bwd > 0 else 0) if not ss else 0
        # rotate the window so every segment is differentiated equally often; the number of
        # distinct (bwd0, nbwd) pairs is what bounds the extra tf.function traces
        w = (upd % max(1, -(-k_seg // n)) * n) if n else 0
        if n_rep <= 1 or ts[0].shape[0] < n_rep:                  # too small to split
            return (_ss_grad_step if ss else _grad_step)(*ts, *st, w, n)
        dist = tuple(strategy.experimental_distribute_values_from_function(
            functools.partial(_split, t)) for t in ts)
        return (_dist_step_ss if ss else _dist_step)(dist, *st, w, n)

    # ------------------------------------------------------------------ physics loss (Fig2 arm-2)
    # Analytic continuity residual on the 3D transient buckets: d(rho)/dt from the model's OWN
    # tendency (JVP through the decoder, no time differences) must balance a divergence flux
    # computed by central FD on the prepended coarse lattice. The unknown unit ratio between the
    # two terms (frame time vs domain length) is eliminated per batch by the closed-form
    # least-squares scalar c* = -<drdt,div>/<div,div> (self-calibrating law; stop-gradient), so
    # the loss penalizes exactly the component of drdt inconsistent with ANY mass-conserving
    # flux scaling. Applied as its own optimizer step (alternating with the data step): the data
    # path's segment-backward machinery stays untouched. Single-replica pilot only.
    if a.phys_w > 0 and a.phys_distill:
        # ARM-3: AMORTIZED REFINEMENT (self-distillation). The verified test-time pull-back
        # corrector (alpha, trust cap, ridge, family unit-ratio) builds a CORRECTED teacher
        # prediction from the same trunk output (all stop-gradient), and the plain student
        # decode is pulled toward it. Physics never generates its own gradient direction --
        # it only reshapes the target; the data loss (separate step) keeps veto power. The
        # deployed model is the PLAIN forward: zero inference overhead, no calibration.
        # Correction runs in POINT space on the phys lattice: W(pt) = interp(W_box) per panel,
        # rank-1 LSQ toward -c*div through the rho row of the decoder Jacobian, z = panel mean,
        # decode -- no box write-back needed (decode is pointwise).
        _G3 = a.phys_grid ** 3
        from v3.model import interp_box as _ibox
        # SGD, NOT Adam: Adam normalizes gradient scale, so in an alternating-step design a
        # tiny auxiliary loss gets data-equal authority regardless of lambda (measured: distill
        # loss ~0.006 degraded the 1k eval 15.1 -> 16.05). SGD steps are proportional.
        opt_phys = tf.keras.optimizers.SGD(a.lr)

        @tf.function(reduce_retracing=True)
        def _phys_step(cn, xw, fm, gn, yn, icq, cq, gq, yq, tsm, cm, fbm, op, cond, fid, sdb,
                       sc, K, steady, dense2d, roles, k_seg, grid_dims, box_dims, surf):
            g3 = a.phys_grid
            dims = tuple(box_dims)
            s5v = tf.cast(tf.reshape(sc, [tf.shape(sc)[0], 1, -1]), tf.float32)  # (B,1,S)
            lat = cq[:, :_G3]                                   # lattice coords (B,G3,K)
            glat = gq[:, :_G3]
            anchor = tf.cast(icq[:, :_G3], tf.float32)          # prev frame at lattice
            with tf.GradientTape() as tape:
                Wp, e = model.trunk_W(cn, xw, fm, gn, K, roles, cond, op, box_dims=dims)
                Wp = tf.cast(Wp, tf.float32)
                Wl = tf.reshape(_ibox(tf.reshape(Wp, tf.concat([tf.shape(Wp)[:-2], [-1]], 0)),
                                      lat, dims),
                                [tf.shape(lat)[0], _G3, model.d_w, model.n_p])  # (B,G3,dw,np)
                z_s = tf.reduce_mean(Wl, -1)
                pred_s = tf.cast(model.dec(z_s, coords=lat, roles=roles, geom=glat,
                                           cond=e, fam_id=fid), tf.float32)     # student delta

                # ---- teacher: corrected panels, entirely stop-gradient ----
                Wt = tf.stop_gradient(Wl)
                et = tf.stop_gradient(e)
                for m in range(model.n_p):
                    cum = tf.reduce_sum(Wt[..., :m + 1], -1) / float(model.n_p)
                    with tf.GradientTape(watch_accessed_variables=False) as tp:
                        tp.watch(cum)
                        d_m = tf.cast(model.dec(cum, coords=lat, roles=roles, geom=glat,
                                                cond=et, fam_id=fid), tf.float32)
                        d_rho = tf.reduce_sum(d_m[..., 3])
                    Jr = tf.cast(tp.gradient(d_rho, cum), tf.float32)           # (B,G3,dw)
                    U = (anchor + d_m) * s5v                                    # physical
                    Uc = tf.reshape(U, [tf.shape(U)[0], g3, g3, g3, tf.shape(U)[-1]])
                    rho = Uc[..., 3]
                    div = tf.zeros_like(rho)
                    for ax_i, ch in ((1, 0), (2, 1), (3, 2)):
                        f = rho * Uc[..., ch]
                        div += 0.5 * (tf.roll(f, -1, axis=ax_i) - tf.roll(f, 1, axis=ax_i))
                    tgt = tf.reshape(-a.phys_c * div,
                                     [tf.shape(U)[0], _G3]) / s5v[:, :, 3]
                    w_m = Wt[..., m]
                    jw = tf.reduce_sum(Jr * w_m, -1)
                    jj = tf.reduce_sum(Jr * Jr, -1)
                    ridge = 1e-2 * tf.reduce_mean(jj)
                    dw = Jr * ((tgt - jw) / (jj + ridge))[..., None]
                    cap = 2.0 * tf.sqrt(tf.reduce_mean(tf.square(w_m)) + 1e-12)
                    dw = tf.clip_by_value(dw, -cap, cap)
                    Wt = tf.concat([Wt[..., :m], (w_m + a.phys_alpha * dw)[..., None],
                                    Wt[..., m + 1:]], -1)
                z_t = tf.stop_gradient(tf.reduce_mean(Wt, -1))
                pred_t = tf.stop_gradient(tf.cast(
                    model.dec(z_t, coords=lat, roles=roles, geom=glat,
                              cond=et, fam_id=fid), tf.float32))
                nrm = tf.stop_gradient(tf.reduce_mean(tf.square(pred_t))) + 1e-12
                Lp = a.phys_w * tf.reduce_mean(tf.square(pred_s - pred_t)) / nrm
            g = tape.gradient(Lp, V)
            g = [gi if gi is not None else tf.zeros_like(v) for gi, v in zip(g, V)]
            opt_phys.apply_gradients(zip(g, V))
            dmag = tf.sqrt(tf.reduce_mean(tf.square(pred_t - pred_s)))
            return tf.stack([Lp, dmag, tf.sqrt(nrm), 0.0, 0.0, 0.0, 0.0, 0.0])

    elif a.phys_w > 0:
        _G3 = a.phys_grid ** 3

        @tf.function(reduce_retracing=True)
        def _phys_step(cn, xw, fm, gn, yn, icq, cq, gq, yq, tsm, cm, fbm, op, cond, fid, sdb,
                       sc, K, steady, dense2d, roles, k_seg, grid_dims, box_dims, surf):
            qall = tf.concat([cn, cq], 1)
            gall = tf.concat([gn, gq], 1)
            u_prev = tf.concat([xw[:, -1], icq], 1)
            Nn = tf.shape(cn)[1]
            icq_full = icq                                     # (B,Q,S) anchor at colloc queries
            g3 = a.phys_grid
            with tf.GradientTape() as tape:
                pred, dudt = model.step(cn, xw, fm, gn, K, roles, cond, op, qall, gall, u_prev,
                                        fam_id=fid, sdf_box=sdb, box_dims=box_dims, fbmask=fbm,
                                        ret_dudt=True)

                def cube_all(t):                               # concat tensor (nodes+queries)
                    x = tf.cast(t[:, Nn:Nn + _G3], tf.float32)
                    return tf.reshape(x, [tf.shape(x)[0], g3, g3, g3, tf.shape(x)[-1]])

                def cube(t):                                   # colloc-only tensor (lattice first)
                    x = tf.cast(t[:, :_G3], tf.float32)
                    return tf.reshape(x, [tf.shape(x)[0], g3, g3, g3, tf.shape(x)[-1]])
                s5 = tf.cast(tf.reshape(sc, [tf.shape(sc)[0], 1, 1, 1, -1]), tf.float32)
                U = cube_all(pred) * s5                            # physical fields
                R = cube_all(dudt) * s5                            # physical rates
                rho, drdt = U[..., 3], R[..., 3]               # slots: 0 vx 1 vy 2 vz 3 density

                def _div(F):                                   # periodic central differences
                    dv = tf.zeros_like(F[..., 3])
                    for ax_i, ch in ((1, 0), (2, 1), (3, 2)):
                        f = F[..., 3] * F[..., ch]
                        dv += 0.5 * (tf.roll(f, -1, axis=ax_i) - tf.roll(f, 1, axis=ax_i))
                    return dv
                div = _div(U)
                # CALIBRATE THE UNIT RATIO ON GROUND TRUTH, NOT ON THE MODEL. The first run
                # re-fit c per batch against the model's own drdt; when the coarse-lattice law
                # carried little signal this drove c -> 0 and the loss degenerated into
                # shrinking the model's tendency (measured: rms(drdt) 0.40 -> 0.012 in 100
                # steps). c* below depends only on the DATA (target-frame rate vs GT flux
                # divergence), so the model cannot influence its own constraint.
                Ugt = cube(yq[:, 0]) * s5
                dy_ = (cube(yq[:, 0]) - cube(icq_full)) * s5
                div_gt = _div(Ugt)
                num = tf.reduce_sum(dy_[..., 3] * div_gt, axis=[1, 2, 3], keepdims=True)
                den = tf.reduce_sum(div_gt * div_gt, axis=[1, 2, 3], keepdims=True) + 1e-12
                c = (tf.constant(a.phys_c) if a.phys_c > 0
                     else tf.stop_gradient(-num / den))       # GT-calibrated unit ratio
                # STOP-GRADIENT THROUGH THE FLUX (2nd cheat found in the first live run): with
                # div differentiable the cheapest descent direction was to ROUGHEN the predicted
                # fields until div matched -drdt/c (measured: rms(div) 0.25 -> 2.3 while eval
                # degraded 13.5 -> 14.9 vs the data-only arm). The verified test-time corrector
                # treats the state as GIVEN and nudges only the rate toward the law RHS; the
                # training loss now does the same: r = drdt - sg(target).
                r = drdt + c * tf.stop_gradient(div)
                nrm = tf.stop_gradient(tf.reduce_mean(tf.square(dy_[..., 3]))) + 1e-12
                Lp = a.phys_w * tf.reduce_mean(tf.square(r)) / nrm
            if not a.phys_diag:
                g = tape.gradient(Lp, V)
                g = [gi if gi is not None else tf.zeros_like(v) for gi, v in zip(g, V)]
                opt.apply_gradients(zip(g, V))
            # 3-way correlations pin down which term is broken: dy = the DATA-side rate
            # (first target frame minus anchor at the lattice, physical, per frame).
            dyr = dy_[..., 3]
            def _corr(x, y):
                xm = x - tf.reduce_mean(x); ym = y - tf.reduce_mean(y)
                return (tf.reduce_sum(xm * ym)
                        / (tf.norm(xm) * tf.norm(ym) + 1e-12))
            diag = tf.stack([Lp, tf.reduce_mean(c),
                             tf.sqrt(tf.reduce_mean(tf.square(drdt))),
                             tf.sqrt(tf.reduce_mean(tf.square(div))),
                             _corr(drdt, dyr), _corr(dyr, -div), _corr(drdt, -div),
                             tf.sqrt(tf.reduce_mean(tf.square(dyr)))])
            return diag

    # ------------------------------------------------------------------ eval
    def evaluate():
        per_fam = {}                                             # fam -> (normalized, physical, fluct)
        for fam in fams:
            rng = np.random.default_rng(1234)
            vals = []
            n = 0
            try:
                it = iter(loader.units_for(fam, "val"))
                while True:
                    with _IO:                               # never concurrent with the prefetcher
                        try:
                            s = next(it)
                        except StopIteration:
                            break
                        ex = make_example_ar(s, a.t_in, a.k_fut, a.k_relax, a.n_colloc,
                                             a.enc_n, rng, box_dims=_bd(int(s["K"]), fam))
                    b = stack_batch([ex])
                    fwd = _fwd_loss_ss if _mode(b) == "ss" else _fwd_loss
                    vals.append(np.asarray(fwd(*_tensors(b), tf.constant(b["scale"]),
                                               *_static(b))))
                    n += 1
                    if n >= a.eval_n:
                        break
            except Exception as e:                               # missing manifest etc.
                print(f"  [eval] {fam}: skipped ({e})", flush=True)
            if vals:
                per_fam[fam] = np.mean(np.stack(vals), 0)        # (3,)
        if not per_fam:
            return float("nan")
        can = float(np.mean([v[0] for v in per_fam.values()]))
        cap = float(np.mean([v[1] for v in per_fam.values()]))
        caf = float(np.mean([v[2] for v in per_fam.values()]))
        msg = " ".join(f"{f}={v[1]*100:.1f}" for f, v in sorted(per_fam.items()))
        msgn = " ".join(f"{f}={v[0]*100:.1f}" for f, v in sorted(per_fam.items()))
        msgf = " ".join(f"{f}={v[2]*100:.1f}" for f, v in sorted(per_fam.items()))
        print(f"[EVAL-PHYS] class-avg rel-L2 {cap*100:.2f}% | {msg}", flush=True)
        print(f"[EVAL-NORM] class-avg rel-L2 {can*100:.2f}% | {msgn}", flush=True)
        print(f"[EVAL-FLUCT] class-avg rel-L2 {caf*100:.2f}% | {msgf}", flush=True)
        return cap                                               # ckpt_best on the PHYSICAL number

    # ------------------------------------------------------------------ loop
    step = 0
    ema = None
    best_ca = float("inf")
    t0 = time.time()
    while step < a.steps:
        for b in batches("train", a.seed + step):
            L = float(run_batch(b, step // max(a.grad_accum, 1)))
            if (a.phys_w > 0 and n_rep == 1 and int(b["K"]) == 3
                    and _mode(b) != "ss" and not bool(b["steady"])):
                _d = np.asarray(_phys_step(*_tensors(b), tf.constant(b["scale"]), *_static(b)))
                if step % 50 == 0 or a.steps <= 200 or step < 20:
                    if a.phys_distill:
                        print(f"  distill {_d[0]:.5f} |t-s|={_d[1]:.3e} rms(t)={_d[2]:.3e}",
                              flush=True)
                    else:
                        print(f"  phys-residual {_d[0]:.4f} c={_d[1]:.3e} "
                              f"rms(drdt)={_d[2]:.3e} rms(div)={_d[3]:.3e} rms(dy)={_d[7]:.3e} "
                              f"corr(drdt,dy)={_d[4]:.3f} corr(dy,-div)={_d[5]:.3f} "
                              f"corr(drdt,-div)={_d[6]:.3f}", flush=True)
            ema = L if ema is None else 0.98 * ema + 0.02 * L
            step += 1
            if a.grad_accum > 1 and step % a.grad_accum == 0:
                _apply_accum(tf.constant(float(a.grad_accum)))
            if a.steps <= 200:                                 # gate runs: show every step
                print(f"step {step} [{b['families'][0]}] K={b['K']} steady={b['steady']} "
                      f"loss {L:.4f}", flush=True)
            if step % 50 == 0:
                print(f"step {step} loss {L:.4f} ema {ema:.4f} "
                      f"({step/(time.time()-t0):.2f} it/s)", flush=True)
            if step % a.eval_every == 0:
                ca = evaluate()
                if ca == ca and ca < best_ca:
                    best_ca = ca
                    save_ckpt(model, V, os.path.join(a.save, "ckpt_best.npz"), names=NAMES)
                    print(f"  [ckpt] new best class-avg {ca*100:.2f}%", flush=True)
            if step % a.ckpt_every == 0:
                save_ckpt(model, V, os.path.join(a.save, f"ckpt_{step}.npz"), names=NAMES)
                olds = sorted((f for f in os.listdir(a.save)
                               if f.startswith("ckpt_") and f != "ckpt_best.npz"),
                              key=lambda f: int(f.split("_")[1].split(".")[0]))
                for f in olds[:-3]:
                    os.remove(os.path.join(a.save, f))
            if step >= a.steps:
                break
    save_ckpt(model, V, os.path.join(a.save, f"ckpt_{step}.npz"), names=NAMES)
    print("[v3] done", flush=True)


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *e):
        return False


if __name__ == "__main__":
    main()
