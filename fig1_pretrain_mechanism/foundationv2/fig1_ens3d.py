"""Fig-1 3D refresh under the OFFICIAL inference convention (W-ensemble K=8, 08-11).
Only the non-dense families carry encoder-subsample noise, so only the 3D columns/panels
change; 2D dense panels are untouched. Emits fig1_ens3d.npz with, per 3D family:
  ens_base rel-L2 (panel-b caption / class-avg refresh), knockout dErr for the fig1c
  SEL heads (3D columns), and for M1.0/Turb the final-frame display volume decoded from
  the ensemble-mean W (panel-b isosurface refresh).

  python -u fig1_ens3d.py <ckpt.npz>     (env: W_ENS=8, N_SAMP=2, FAMS=comma list)
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

KENS = int(os.environ.get("W_ENS", "8"))
N_SAMP = int(os.environ.get("N_SAMP", "2"))
SKIP = int(os.environ.get("SKIP", "0"))
KO_LIST = [m for m in os.environ.get("KO_LIST", ",".join(["advection","reaction_nl","wave","shock","compressible","buoyancy","diffusion","vortex","spectral"])).split(",") if m]
M01 = "pdebench3d_cns_Rand_M0.1_Eta1e-08_Zeta1e-08"
M10 = "pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08"
T3 = "pdebench3d_cns_Turb_M1.0_Eta1e-08_Zeta1e-08"
FAMS = os.environ.get("FAMS", f"{M01},{M10},{T3}").split(",")
BOX = {M01: (16, 16, 16), M10: (32, 32, 32), T3: (32, 32, 32)}
SEL = None  # per-process KO_LIST
VOL_N = 64
OUT = sys.argv[2] if len(sys.argv) > 2 else "/code-vol/motion_fv2/figs_m1s2b"

ck = sys.argv[1]
model = V3Model(S=NUM_SLOTS, d=384, depth=8, d_w=16, n_p=16, d_cond=128,
                dims2=(64, 64), dims3=(32, 32, 32), t_in=10, transport=True,
                n_fams=len(FAMILIES), n_semantic=SEMANTIC_SLOTS,
                gate_cond=1, dict_decode=0, fam_head=1)
model.warm_build()
V = model.trainable_variables
load_ckpt(V, ck, d_w=16, names=[v.name for v in V])
print(f"loaded {ck}", flush=True)
rng = np.random.default_rng(1234)
out = {}

for FAM in FAMS:
    DIMS = BOX[FAM]
    NC = int(np.prod(DIMS))
    cellsT = tf.constant(_cell_coords(DIMS)[None].astype(np.float32))
    gq = tf.zeros([1, NC, 8], tf.float32)
    keep = tf.concat([tf.ones([model.n_semantic]),
                      tf.zeros([model.S - model.n_semantic])], 0)[None, None]
    it = iter(loader.units_for(FAM, "val"))
    for _ in range(SKIP):
        s_sk = next(it)
        _ = [make_example_ar(s_sk, 10, 10, 4, 512, 8192, rng, box_dims=DIMS) for _ in range(KENS)]
    base_r, ko_r = [], {m: [] for m in KO_LIST}
    for si in range(N_SAMP):
        s = next(it)
        exs = [make_example_ar(s, 10, 10, 4, 512, 8192, rng, box_dims=DIMS)
               for _ in range(KENS)]
        bs = [stack_batch([e]) for e in exs]
        reps = [dict(cn=tf.constant(b["coords_node"]), xw=tf.constant(b["x_win"]),
                     fm=tf.constant(b["fmask"]), gn=tf.constant(b["geom_node"]))
                for b in bs]
        b0 = bs[0]
        cond = tf.constant(b0["cond"]); op = tf.constant(b0["op_multihot"])
        fid = tf.constant(b0["fam_id"]); fbm = tf.constant(b0["fbmask"])
        roles = tuple(int(r) for r in b0["roles"])
        sc = np.asarray(b0["scale"], np.float32)[0]
        fields = np.asarray(s["fields"], np.float32)
        flat = fields.reshape(fields.shape[0], -1, fields.shape[-1])
        nat = int(round(flat.shape[1] ** (1 / 3)))
        st = max(1, nat // DIMS[0])
        ax = np.arange(0, nat, st)[:DIMS[0]]
        gi = (ax[:, None, None] * nat * nat + ax[None, :, None] * nat
              + ax[None, None, :]).ravel()
        y_cells = flat[10:20][:, gi]
        u0 = flat[9][gi][None] / sc[None, None]
        fbt = tf.cast(fbm, tf.float32)[:, None, :]

        def rollout(want_vol=False):
            wins = [r["xw"] for r in reps]
            fms = [r["fm"] for r in reps]
            u_prev = tf.constant(u0, tf.float32)
            preds, zc = [], None
            for k in range(10):
                zs, e = [], None
                for r, win, fmask in zip(reps, wins, fms):
                    Wp, e = model.trunk_W(r["cn"], win, fmask, r["gn"], 3, roles,
                                          cond, op, box_dims=DIMS)
                    zs.append(tf.reshape(tf.reduce_mean(tf.cast(Wp, tf.float32), -1),
                                         [1, NC, model.d_w]))
                zc = tf.add_n(zs) / float(len(zs))
                d_c = tf.cast(model.dec(zc, coords=cellsT, roles=roles, geom=gq,
                                        cond=e, fam_id=fid), tf.float32)
                u_prev = u_prev * keep + d_c * fbt
                preds.append(u_prev.numpy()[0])
                zbox = tf.reshape(zc, (1,) + DIMS + (model.d_w,))
                nw, nf = [], []
                for r, win, fmask in zip(reps, wins, fms):
                    zn = tf.cast(interp_box(zbox, r["cn"], DIMS), tf.float32)
                    d_n = tf.cast(model.dec(zn, coords=r["cn"], roles=roles,
                                            geom=tf.cast(r["gn"], tf.float32),
                                            cond=e, fam_id=fid), tf.float32)
                    u_n = win[:, -1] * keep + d_n * fbt
                    nw.append(tf.concat([win[:, 1:], u_n[:, None]], 1))
                    nf.append(tf.concat([fmask[:, 1:], tf.ones_like(fmask[:, :1])], 1))
                wins, fms = nw, nf
            vol = None
            if want_vol:                                       # final-frame display volume
                n = VOL_N
                axq = (np.arange(n) + 0.5) / n
                g3 = np.stack(np.meshgrid(axq, axq, axq, indexing="ij"), -1).reshape(-1, 3)
                q = tf.constant(g3[None].astype(np.float32))
                stv = nat // n
                anchor_v = (fields[9, ::stv, ::stv, ::stv, :]
                            / sc[None, None, None, :]).reshape(1, -1, fields.shape[-1])
                zq = tf.cast(interp_box(tf.reshape(zc, (1,) + DIMS + (model.d_w,)), q, DIMS),
                             tf.float32)
                dq = tf.cast(model.dec(zq, coords=q, roles=roles,
                                       geom=tf.zeros([1, n ** 3, 8]), cond=e, fam_id=fid),
                             tf.float32)
                uq = tf.constant(anchor_v, tf.float32) * keep + dq * fbt
                # NOTE: display volume is the FINAL-step state only when preds are chained on
                # the same lattice; here we re-add the last delta at the anchor -- approximate
                # for display; rel numbers come from the cell rollout above.
                vol = uq.numpy()[0]
            return np.stack(preds), vol

        cmv = np.asarray(b0["cmask"], np.float32)[0][None, None]

        def score(p):
            p = p * sc[None, None]
            num = np.sqrt((((p - y_cells) * cmv) ** 2).sum((1, 2)))
            den = np.sqrt(((y_cells * cmv) ** 2).sum((1, 2))) + 1e-12
            return float((num / den).mean())

        pb, vol = rollout(want_vol=(si == 0 and SKIP == 0 and FAM in (M10, T3) and int(os.environ.get("WANT_VOL", "1"))))
        base_r.append(score(pb))
        if vol is not None:
            ch = int(np.nonzero(cmv[0, 0])[0][-1])
            out[f"vol_{FAM}_pred"] = (vol[:, ch] * sc[ch]).reshape(VOL_N, VOL_N, VOL_N)
            stv = nat // VOL_N
            out[f"vol_{FAM}_gt"] = fields[19, ::stv, ::stv, ::stv, ch]
        for m in KO_LIST:
            g0 = model.banks.gates[m].numpy()
            model.banks.gates[m].assign(tf.zeros_like(model.banks.gates[m]))
            gc0 = model.banks.gcond[m].kernel.numpy()
            model.banks.gcond[m].kernel.assign(tf.zeros_like(model.banks.gcond[m].kernel))
            pk, _ = rollout()
            model.banks.gates[m].assign(g0)
            model.banks.gcond[m].kernel.assign(gc0)
            ko_r[m].append(score(pk))
        print(f"[{FAM}] s{si} base={base_r[-1]*100:.2f}% "
              + " ".join(f"{m}:{(ko_r[m][-1]-base_r[-1])*100:+.2f}" for m in KO_LIST), flush=True)
        import gc
        gc.collect()
    out[f"base_{FAM}_s{SKIP}"] = np.mean(base_r)
    for m in KO_LIST:
        out[f"ko_{FAM}_{m}_s{SKIP}"] = np.mean(ko_r[m])
    print(f"[{FAM}] ENS{KENS} base={np.mean(base_r)*100:.2f}%", flush=True)

tag = os.environ.get("OUT_TAG", "x")
np.savez_compressed(os.path.join(OUT, f"fig1_ens3d_{tag}.npz"), **out)
print("ENS3D_DONE", flush=True)
