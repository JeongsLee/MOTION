"""W-panel post-hoc optimization pilot (data-side signals; the law-side pull-back is
exhausted after FT -- refine sweep 08-11). Conditions on a 10-frame free rollout:

  BASE        single encoder pass (the deployed path)
  ENS         W-ensemble: K encoder passes on different node subsamples of the SAME
              window, averaged in W (box lattice is shared); each replica keeps its own
              feedback window, all fed from the ensemble-mean prediction.
  REFIT       amortization-gap correction: re-anchor one frame back (window ..8 with a
              duplicated first frame), LSQ-correct the latent-mean z toward the OBSERVED
              frame 9 (rho row), and apply that dz to every subsequent segment.
  ENS+REFIT   both.

  python -u fig2_wopt_pilot.py <ckpt.npz> [n_samples]   (env: SKIP, K_ENS, REF_BETA)
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

FAM = os.environ.get("WOPT_FAM", "pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08")
KD = int(os.environ.get("WOPT_KD", "3"))
DIMS = tuple(int(x) for x in os.environ.get("WOPT_DIMS", "32,32,32").split(","))
K_ENS = int(os.environ.get("K_ENS", "4"))
ENC_N = int(os.environ.get("ENC_N", "8192"))
REF_BETA = float(os.environ.get("REF_BETA", "1.0"))
RIDGE_F = 1e-2
TRUST = 2.0
N_EVAL = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SKIP = int(os.environ.get("SKIP", "0"))
D4G = os.environ.get("D4_G", "")     # 3D flips: fx/fy/fz combos e.g. "fx", "fxfy"
SAVE_TAG = os.environ.get("SAVE_TAG", "")

def _d4(su):
    if not D4G:
        return su
    f = np.array(su["fields"], np.float32, copy=True)   # (T,X,Y,Z,S)
    if "fx" in D4G: f = f[:, ::-1]; f[..., 0] *= -1
    if "fy" in D4G: f = f[:, :, ::-1]; f[..., 1] *= -1
    if "fz" in D4G: f = f[:, :, :, ::-1]; f[..., 2] *= -1
    return {**{k: su[k] for k in su.keys()}, "fields": np.ascontiguousarray(f)}

ck = sys.argv[1]
NC = int(np.prod(DIMS))
cells = _cell_coords(DIMS)[None]

model = V3Model(S=NUM_SLOTS, d=384, depth=8, d_w=16, n_p=16, d_cond=128,
                dims2=(64, 64), dims3=DIMS, t_in=10, transport=True,
                n_fams=len(FAMILIES), n_semantic=SEMANTIC_SLOTS,
                gate_cond=1, dict_decode=0, fam_head=1)
model.warm_build()
V = model.trainable_variables
load_ckpt(V, ck, d_w=16, names=[v.name for v in V])
print(f"loaded {ck}", flush=True)

rng = np.random.default_rng(1234)
it = iter(loader.units_for(FAM, "val"))
for _ in range(SKIP):
    s_skip = next(it)
    _ = make_example_ar(s_skip, 10, 10, 4, 512, 8192, rng, box_dims=DIMS)

CONDS = tuple(os.environ.get("WOPT_CONDS", "BASE,ENS,REFIT,ENS+REFIT").split(","))
res = {c: [] for c in CONDS}
n = 0
while n < N_EVAL:
    try:
        s = _d4(next(it))
    except StopIteration:
        break
    exs = [make_example_ar(s, 10, 10, 4, 512, ENC_N, rng, box_dims=DIMS)
           for _ in range(K_ENS)]                       # replica 0 doubles as the BASE pass
    bs = [stack_batch([e]) for e in exs]
    reps = []
    for b in bs:
        reps.append(dict(cn=tf.constant(b["coords_node"]), xw=tf.constant(b["x_win"]),
                         fm=tf.constant(b["fmask"]), gn=tf.constant(b["geom_node"])))
    b0 = bs[0]
    cond = tf.constant(b0["cond"]); op = tf.constant(b0["op_multihot"])
    fid = tf.constant(b0["fam_id"]); fbm = tf.constant(b0["fbmask"])
    roles = tuple(int(r) for r in b0["roles"])
    sc = np.asarray(b0["scale"], np.float32)[0]

    fields = np.asarray(s["fields"], np.float32)
    flat = fields.reshape(fields.shape[0], -1, fields.shape[-1])
    nat = int(round(flat.shape[1] ** (1 / KD)))
    st = max(1, nat // DIMS[0])
    ax = np.arange(0, nat, st)[:DIMS[0]]
    if KD == 3:
        gi = (ax[:, None, None] * nat * nat + ax[None, :, None] * nat
              + ax[None, None, :]).ravel()
    else:
        gi = (ax[:, None] * nat + ax[None, :]).ravel()
    anchor_t = 9
    y_cells = flat[anchor_t + 1:anchor_t + 11][:, gi]                 # (10,NC,S) physical
    u0_cells = flat[anchor_t][gi][None] / sc[None, None]
    cellsT = tf.constant(cells.astype(np.float32))
    gq = tf.zeros([1, NC, 8], tf.float32)
    keep = tf.concat([tf.ones([model.n_semantic]),
                      tf.zeros([model.S - model.n_semantic])], 0)[None, None]
    fb = tf.cast(fbm, tf.float32)[:, None, :]

    dz = tf.zeros([1, NC, model.d_w], tf.float32); gap = 0.0
    NEED_REFIT = any("REFIT" in c for c in CONDS)
    # ---- REFIT: dz from the observable segment (anchor shifted one frame back) ----
    if NEED_REFIT:
      s_shift = {k: s[k] for k in s.keys()}
      s_shift["fields"] = np.concatenate([fields[:1], fields[:-1]], 0)  # dup frame0 -> anchor=8
      ex_s = make_example_ar(s_shift, 10, 10, 4, 512, 8192, rng, box_dims=DIMS)
      b_s = stack_batch([ex_s])
      Wp_s, e_s = model.trunk_W(tf.constant(b_s["coords_node"]), tf.constant(b_s["x_win"]),
                              tf.constant(b_s["fmask"]), tf.constant(b_s["geom_node"]),
                              KD, roles, cond, op, box_dims=DIMS)
      z_s = tf.reshape(tf.reduce_mean(tf.cast(Wp_s, tf.float32), -1), [1, NC, model.d_w])
      u8 = flat[anchor_t - 1][gi][None] / sc[None, None]                # normalized anchor(8)
      y9n = flat[anchor_t][gi][None] / sc[None, None]                   # observed frame 9
      with tf.GradientTape() as tp:
        tp.watch(z_s)
        d_s = model.dec(z_s, coords=cellsT, roles=roles, geom=gq, cond=e_s, fam_id=fid)
        d_rho = tf.reduce_sum(tf.cast(d_s, tf.float32)[..., 3])
      J = tf.cast(tp.gradient(d_rho, z_s), tf.float32)                  # (1,NC,d_w)
      pred9 = tf.cast(u8, tf.float32) + tf.cast(d_s, tf.float32)
      resid = tf.cast(y9n, tf.float32)[..., 3] - pred9[..., 3]          # (1,NC) normalized rho gap
      jj = tf.reduce_sum(J * J, -1)
      dz = J * (resid / (jj + RIDGE_F * tf.reduce_mean(jj)))[..., None]
      cap = TRUST * tf.sqrt(tf.reduce_mean(tf.square(z_s)) + 1e-12)
      dz = tf.clip_by_value(dz, -cap, cap)
      gap = float(tf.sqrt(tf.reduce_mean(tf.square(resid))))

    def rollout(use_ens, use_refit):
        R = reps if use_ens else reps[:1]
        wins = [r["xw"] for r in R]
        fms = [r["fm"] for r in R]
        u_prev_cells = tf.constant(u0_cells, tf.float32)
        preds = []
        for k in range(10):
            zs = []
            e = None
            for r, win, fmask in zip(R, wins, fms):
                Wp, e = model.trunk_W(r["cn"], win, fmask, r["gn"], KD, roles, cond, op,
                                      box_dims=DIMS)
                zs.append(tf.reshape(tf.reduce_mean(tf.cast(Wp, tf.float32), -1),
                                     [1, NC, model.d_w]))
            zc = tf.add_n(zs) / float(len(zs))
            if use_refit:
                zc = zc + REF_BETA * dz
            d_cells = tf.cast(model.dec(zc, coords=cellsT, roles=roles, geom=gq,
                                        cond=e, fam_id=fid), tf.float32)
            u_prev_cells = u_prev_cells * keep + d_cells * fb
            preds.append(u_prev_cells.numpy()[0])
            zbox = tf.reshape(zc, (1,) + DIMS + (model.d_w,))
            new_wins, new_fms = [], []
            for r, win, fmask in zip(R, wins, fms):
                zn = tf.cast(interp_box(zbox, r["cn"], DIMS), tf.float32)
                d_nodes = tf.cast(model.dec(zn, coords=r["cn"], roles=roles,
                                            geom=tf.cast(r["gn"], tf.float32),
                                            cond=e, fam_id=fid), tf.float32)
                u_nodes = win[:, -1] * keep + d_nodes * fb
                new_wins.append(tf.concat([win[:, 1:], u_nodes[:, None]], 1))
                new_fms.append(tf.concat([fmask[:, 1:], tf.ones_like(fmask[:, :1])], 1))
            wins, fms = new_wins, new_fms
        return np.stack(preds)

    cmv = np.asarray(b0["cmask"], np.float32)[0][None, None]
    row = []
    CMAP = {"BASE": (0, 0), "ENS": (1, 0), "REFIT": (0, 1), "ENS+REFIT": (1, 1)}
    for cname in CONDS:
        ue, ur = CMAP[cname]
        p = rollout(bool(ue), bool(ur)) * sc[None, None]
        num = np.sqrt((((p - y_cells) * cmv) ** 2).sum((1, 2)))
        den = np.sqrt(((y_cells * cmv) ** 2).sum((1, 2))) + 1e-12
        r = 100 * float((num / den).mean())
        res[cname].append(r)
        row.append(f"{cname}: {r:.2f}%")
        if SAVE_TAG and cname == "ENS":
            np.savez_compressed(f"/code-vol/motion_fv2/figs_m1s2b/prod_{SAVE_TAG}_s{SKIP + n}.npz",
                                pred=p, gt=y_cells)
    print(f"sample {SKIP + n} gap9={gap:.4f} | " + "  ".join(row), flush=True)
    n += 1
    del fields, flat, y_cells
    import gc
    gc.collect()

print("=== MEAN rel-L2 over", n, "samples ===")
for cname in CONDS:
    print(f"{cname}: {np.mean(res[cname]):.3f}%", flush=True)
print("WOPT_DONE", flush=True)
