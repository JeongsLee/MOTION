"""Fig2a stage-3 pilot: test-time pull-back correction of the detached instance (3D CNS).

For each val trajectory: free-running 10-segment AR rollout; in each segment the panel
tendencies W (the detached instance) are corrected toward the continuity law by a rank-1
least-squares pull-back through the decoder Jacobian (rho row), with per-sample
GT-calibrated unit ratio, ridge and a trust cap. Correction alternatives ALPHAS are run
side by side (alpha=0 is the uncorrected baseline); scoring is physical rel-L2 over the
10 frames at the box-cell grid.

  python -u fig2_refine3d.py <ckpt.npz> [n_samples]
"""
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.pretrain import load_ckpt
from data import loader
from data.registry import FAMILIES, NUM_SLOTS, SEMANTIC_SLOTS
from v3.data_ar import _cell_coords, make_example_ar, stack_batch
from v3.model import V3Model, interp_box

FAM = "pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08"
DIMS = (32, 32, 32)                      # box_table 3D M1.0
ALPHAS = [0.0, 0.15, 0.25, 0.35]
RIDGE_F = 1e-2
TRUST = 2.0                              # cap |dw| at TRUST x rms(w) per panel
N_EVAL = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SKIP = int(os.environ.get("SKIP", "0"))       # process samples [SKIP, SKIP+N_EVAL)

ck = sys.argv[1]
NC = int(np.prod(DIMS))
cells = _cell_coords(DIMS)[None]                                    # (1,NC,3)

model = V3Model(S=NUM_SLOTS, d=384, depth=8, d_w=16, n_p=16, d_cond=128,
                dims2=(64, 64), dims3=DIMS, t_in=10, transport=True,
                n_fams=len(FAMILIES), n_semantic=SEMANTIC_SLOTS,
                gate_cond=1, dict_decode=0, fam_head=1)
model.warm_build()
V = model.trainable_variables
NAMES = [v.name for v in V]
load_ckpt(V, ck, d_w=16, names=NAMES)
print(f"loaded {ck}", flush=True)

rng = np.random.default_rng(1234)
it = iter(loader.units_for(FAM, "val"))
for _ in range(SKIP):
    ex_skip = next(it)
    _ = make_example_ar(ex_skip, 10, 10, 4, 512, 8192, rng, box_dims=DIMS)  # keep rng in sync
res = {a: [] for a in ALPHAS}
n = 0
while n < N_EVAL:
    try:
        s = next(it)
    except StopIteration:
        break
    ex = make_example_ar(s, 10, 10, 4, 512, 8192, rng, box_dims=DIMS)
    b = stack_batch([ex])
    cn = tf.constant(b["coords_node"]); xw0 = tf.constant(b["x_win"])
    fm0 = tf.constant(b["fmask"]); gn = tf.constant(b["geom_node"])
    cond = tf.constant(b["cond"]); op = tf.constant(b["op_multihot"])
    fid = tf.constant(b["fam_id"]); fbm = tf.constant(b["fbmask"])
    roles = tuple(int(r) for r in b["roles"])
    sc = np.asarray(b["scale"], np.float32)[0]                      # (S,)
    K = 3

    # GT at the cell grid, physical: gather from the native field array
    fields = np.asarray(s["fields"], np.float32)                    # (T, 128^3(flat later), S)
    flat = fields.reshape(fields.shape[0], -1, fields.shape[-1])
    nat = int(round(flat.shape[1] ** (1 / 3)))
    st = nat // DIMS[0]
    ax = np.arange(0, nat, st)[:DIMS[0]]
    gi = (ax[:, None, None] * nat * nat + ax[None, :, None] * nat
          + ax[None, None, :]).ravel()
    anchor_t = 9                                                    # window = frames 0..9
    y_cells = flat[anchor_t + 1:anchor_t + 11][:, gi]               # (10,NC,S) physical
    u0_cells = flat[anchor_t][gi][None] / sc[None, None]            # (1,NC,S) normalized
    cellsT = tf.constant(cells.astype(np.float32))
    gq = tf.zeros([1, NC, 8], tf.float32)

    # per-sample GT unit ratio at this grid: dy vs central-diff flux divergence
    s5 = sc
    Ug = (y_cells[0] ).reshape(DIMS + (-1,))
    rho_g, u_g = Ug[..., 3], Ug
    div_g = np.zeros(DIMS, np.float32)
    for a_, ch in ((0, 0), (1, 1), (2, 2)):
        f = rho_g * Ug[..., ch]
        div_g += 0.5 * (np.roll(f, -1, axis=a_) - np.roll(f, 1, axis=a_))
    dy_g = (y_cells[0][:, 3] - flat[anchor_t][gi][:, 3]).reshape(DIMS)
    c_gt = float(-(dy_g * div_g).sum() / ((div_g ** 2).sum() + 1e-12))

    def rollout(alpha):
        win, fmask = xw0, fm0
        u_prev_cells = tf.constant(u0_cells)                        # (1,NC,S) normalized
        u_prev_nodes = win[:, -1]
        preds = []
        for k in range(10):
            Wp, e = model.trunk_W(cn, win, fmask, gn, K, roles, cond, op, box_dims=DIMS)
            Wp = tf.cast(Wp, tf.float32)
            if alpha > 0:
                Wf = tf.reshape(Wp, [1, NC, model.d_w, model.n_p])
                for m in range(model.n_p):
                    cum = tf.reduce_sum(Wf[..., :m + 1], -1) / float(model.n_p)
                    with tf.GradientTape() as tp:
                        tp.watch(cum)
                        d_m = model.dec(cum, coords=cellsT, roles=roles, geom=gq,
                                        cond=e, fam_id=fid)
                        d_rho = tf.reduce_sum(tf.cast(d_m, tf.float32)[..., 3])
                    J = tf.cast(tp.gradient(d_rho, cum), tf.float32)        # (1,NC,d_w)
                    U = (tf.cast(u_prev_cells, tf.float32)
                         + tf.cast(d_m, tf.float32)) * s5[None, None]
                    Uc = tf.reshape(U, (1,) + DIMS + (-1,))
                    rho = Uc[..., 3]
                    div = tf.zeros_like(rho)
                    for a_, ch in ((1, 0), (2, 1), (3, 2)):
                        f = rho * Uc[..., ch]
                        div += 0.5 * (tf.roll(f, -1, axis=a_) - tf.roll(f, 1, axis=a_))
                    tgt = tf.reshape(-c_gt * div, [1, NC]) / s5[3]          # normalized rate
                    w_m = Wf[..., m]
                    jw = tf.reduce_sum(J * w_m, -1)                         # (1,NC)
                    jj = tf.reduce_sum(J * J, -1)
                    ridge = RIDGE_F * tf.reduce_mean(jj)
                    dw = J * ((tgt - jw) / (jj + ridge))[..., None]
                    cap = TRUST * tf.sqrt(tf.reduce_mean(tf.square(w_m)) + 1e-12)
                    dw = tf.clip_by_value(dw, -cap, cap)
                    Wf = tf.concat([Wf[..., :m], (w_m + alpha * dw)[..., None],
                                    Wf[..., m + 1:]], -1)
                Wp = tf.reshape(Wf, tf.shape(Wp))
            z = tf.reduce_mean(Wp, -1)                              # (1,*dims,d_w)
            zc = tf.reshape(z, [1, NC, model.d_w])
            d_cells = tf.cast(model.dec(zc, coords=cellsT, roles=roles, geom=gq,
                                        cond=e, fam_id=fid), tf.float32)
            zn = tf.cast(interp_box(z, cn, DIMS), tf.float32)
            d_nodes = tf.cast(model.dec(zn, coords=cn, roles=roles,
                                        geom=tf.cast(gn, tf.float32),
                                        cond=e, fam_id=fid), tf.float32)
            fb = tf.cast(fbm, tf.float32)[:, None, :]
            keep = tf.concat([tf.ones([model.n_semantic]),
                              tf.zeros([model.S - model.n_semantic])], 0)[None, None]
            u_prev_cells = u_prev_cells * keep + d_cells * fb
            u_prev_nodes = u_prev_nodes * keep + d_nodes * fb
            preds.append(u_prev_cells.numpy()[0])
            win = tf.concat([win[:, 1:], u_prev_nodes[:, None]], 1)
            fmask = tf.concat([fmask[:, 1:], tf.ones_like(fmask[:, :1])], 1)
        return np.stack(preds)                                      # (10,NC,S) normalized

    row = []
    cmv = np.asarray(b["cmask"], np.float32)[0][None, None]         # (1,1,S) scored channels
    for a_ in ALPHAS:
        p = rollout(a_) * sc[None, None]                            # physical
        num = np.sqrt((((p - y_cells) * cmv) ** 2).sum((1, 2)))
        den = np.sqrt(((y_cells * cmv) ** 2).sum((1, 2))) + 1e-12
        r = 100 * float((num / den).mean())
        res[a_].append(r)
        row.append(f"a={a_}: {r:.2f}%")
    print(f"sample {SKIP + n} c_gt={c_gt:.4f} | " + "  ".join(row), flush=True)
    n += 1
    del fields, flat, y_cells
    import gc
    gc.collect()

print("=== MEAN rel-L2 over", n, "samples ===")
for a_ in ALPHAS:
    print(f"alpha={a_}: {np.mean(res[a_]):.3f}%", flush=True)
print("REFINE_DONE", flush=True)
