"""Fig-1 3D refresh under the OFFICIAL inference convention (08-11(25)):
subsample W-ensemble K=8 (W-space mean, shared lattice) x isotropic flips {id,fx,fy,fz}
(FIELD-space inverse-transform mean -- latents are not flip-covariant). Graph-mode
rollout (the eager harness OOMs at 32^3 x K8).

  python -u fig1_ens3d_v2.py <ckpt> <out_dir>
  env: FAM, SKIP, N_SAMP=1, KO_LIST (comma; empty=base only), WANT_VOL, OUT_TAG, W_ENS=8
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
SKIP = int(os.environ.get("SKIP", "0"))
N_SAMP = int(os.environ.get("N_SAMP", "1"))
KO_LIST = [m for m in os.environ.get("KO_LIST", "").split(",") if m]
FLIPS = os.environ.get("FLIPS", "id,fx,fy,fz").split(",")
M01 = "pdebench3d_cns_Rand_M0.1_Eta1e-08_Zeta1e-08"
M10 = "pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08"
T3 = "pdebench3d_cns_Turb_M1.0_Eta1e-08_Zeta1e-08"
FAM = os.environ.get("FAM", M10)
BOX = {M01: (16,) * 3, M10: (32,) * 3, T3: (32,) * 3}[FAM]
NC = int(np.prod(BOX))
WANT_VOL = int(os.environ.get("WANT_VOL", "0"))
VOL_N = 64
OUT = sys.argv[2] if len(sys.argv) > 2 else "/code-vol/motion_fv2/figs_m1s2b"
TAG = os.environ.get("OUT_TAG", "x")

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
cellsT = tf.constant(_cell_coords(BOX)[None].astype(np.float32))
gqC = tf.zeros([1, NC, 8], tf.float32)
keep = tf.concat([tf.ones([model.n_semantic]),
                  tf.zeros([model.S - model.n_semantic])], 0)[None, None]


def flip_fields(f, g):
    f = np.array(f, np.float32, copy=True)                # (T,X,Y,Z,S)
    if "fx" in g: f = f[:, ::-1]; f[..., 0] *= -1
    if "fy" in g: f = f[:, :, ::-1]; f[..., 1] *= -1
    if "fz" in g: f = f[:, :, :, ::-1]; f[..., 2] *= -1
    return np.ascontiguousarray(f)


def inv_pred(p, g):                                       # (10,NC,S) on BOX lattice
    p = p.reshape((10,) + BOX + (-1,)).copy()
    if "fx" in g: p = p[:, ::-1]; p[..., 0] *= -1
    if "fy" in g: p = p[:, :, ::-1]; p[..., 1] *= -1
    if "fz" in g: p = p[:, :, :, ::-1]; p[..., 2] *= -1
    return p.reshape(10, NC, -1)


@tf.function(reduce_retracing=True)
def ens_roll(cn, xw, fm, gn, cond, op, fid, fbm, u0, roles):
    """cn/xw/fm/gn stacked (K, ...). W-mean over K each step; single cell decode;
    per-replica node feedback in one batched dec call."""
    win, fmask = xw, fm
    u_prev = u0
    fb = tf.cast(fbm, tf.float32)[:, None, :]
    outs = tf.TensorArray(tf.float32, size=10)
    for k in tf.range(10):
        Wp, e = model.trunk_W(cn, win, fmask, gn, 3, list(roles), cond, op, box_dims=BOX)
        z1 = tf.reduce_mean(tf.cast(Wp, tf.float32), -1)                    # (K,*dims,d_w)
        zm = tf.reduce_mean(z1, 0, keepdims=True)                           # ens mean
        zc = tf.reshape(zm, [1, NC, model.d_w])
        d_c = tf.cast(model.dec(zc, coords=cellsT, roles=list(roles), geom=gqC,
                                cond=e[:1], fam_id=fid[:1]), tf.float32)
        u_prev = u_prev * keep + d_c * fb[:1]
        outs = outs.write(k, u_prev[0])
        zbox = tf.reshape(zm, (1,) + BOX + (model.d_w,))
        zboxK = tf.tile(zbox, [tf.shape(cn)[0], 1, 1, 1, 1])
        zn = tf.cast(interp_box(zboxK, cn, BOX), tf.float32)
        d_n = tf.cast(model.dec(zn, coords=cn, roles=list(roles),
                                geom=tf.cast(gn, tf.float32), cond=e, fam_id=fid),
                      tf.float32)
        u_n = win[:, -1] * keep + d_n * fb
        win = tf.concat([win[:, 1:], u_n[:, None]], 1)
        fmask = tf.concat([fmask[:, 1:], tf.ones_like(fmask[:, :1])], 1)
    return outs.stack()


it = iter(loader.units_for(FAM, "val"))
for _ in range(SKIP):
    s_sk = next(it)
    for g in FLIPS:
        _ = [make_example_ar({**{k: s_sk[k] for k in s_sk.keys()},
                              "fields": flip_fields(s_sk["fields"], g)} if g != "id" else s_sk,
                             10, 10, 4, 512, 8192, rng, box_dims=BOX) for _ in range(KENS)]
out = {}
for si in range(N_SAMP):
    s = next(it)
    fields = np.asarray(s["fields"], np.float32)
    flat = fields.reshape(fields.shape[0], -1, fields.shape[-1])
    nat = int(round(flat.shape[1] ** (1 / 3)))
    st = max(1, nat // BOX[0])
    ax = np.arange(0, nat, st)[:BOX[0]]
    gi = (ax[:, None, None] * nat * nat + ax[None, :, None] * nat + ax[None, None, :]).ravel()
    y_cells = flat[10:20][:, gi]
    packs = {}
    for g in FLIPS:
        sg = s if g == "id" else {**{k: s[k] for k in s.keys()},
                                  "fields": flip_fields(fields, g)}
        exs = [make_example_ar(sg, 10, 10, 4, 512, 8192, rng, box_dims=BOX)
               for _ in range(KENS)]
        bs = [stack_batch([e]) for e in exs]
        cat = lambda k: tf.constant(np.concatenate([b[k] for b in bs], 0))
        fg = np.asarray(sg["fields"], np.float32).reshape(flat.shape)
        u0g = fg[9][gi][None] / np.asarray(bs[0]["scale"], np.float32)[0][None, None]
        packs[g] = dict(cn=cat("coords_node"), xw=cat("x_win"), fm=cat("fmask"),
                        gn=cat("geom_node"), cond=cat("cond"), op=cat("op_multihot"),
                        fid=cat("fam_id"), fbm=tf.constant(bs[0]["fbmask"]),
                        u0=tf.constant(u0g, tf.float32),
                        roles=tuple(int(r) for r in bs[0]["roles"]),
                        sc=np.asarray(bs[0]["scale"], np.float32)[0],
                        cm=np.asarray(bs[0]["cmask"], np.float32)[0][None, None])

    def run_all():
        preds = []
        for g in FLIPS:
            p = packs[g]
            pr = ens_roll(p["cn"], p["xw"], p["fm"], p["gn"], p["cond"], p["op"],
                          p["fid"], p["fbm"], p["u0"], p["roles"]).numpy()
            preds.append(inv_pred(pr, g))
        return np.mean(preds, 0)

    p0 = packs["id"]

    def score(pred):
        pp = pred * p0["sc"][None, None]
        num = np.sqrt((((pp - y_cells) * p0["cm"]) ** 2).sum((1, 2)))
        den = np.sqrt(((y_cells * p0["cm"]) ** 2).sum((1, 2))) + 1e-12
        return float((num / den).mean())

    pb = run_all()
    out[f"base_{FAM}_s{SKIP + si}"] = score(pb)
    if int(os.environ.get("SAVE_PRED", "0")):
        out[f"pred_{FAM}_s{SKIP + si}"] = (pb * p0["sc"][None, None]).astype(np.float32)
        out[f"gt_{FAM}_s{SKIP + si}"] = y_cells.astype(np.float32)
    if WANT_VOL:
        ch = int(np.nonzero(p0["cm"][0, 0])[0][-1])
        stv = nat // VOL_N
        vsub = pb[-1].reshape(BOX + (-1,))[..., ch] * p0["sc"][ch]
        out[f"vol_{FAM}_pred_box"] = vsub                      # BOX-res ensemble final state
        out[f"vol_{FAM}_gt"] = fields[19, ::stv, ::stv, ::stv, ch]
        out[f"vol_{FAM}_rel_official"] = out[f"base_{FAM}_s{SKIP + si}"]
    for m in KO_LIST:
        g0 = model.banks.gates[m].numpy()
        model.banks.gates[m].assign(tf.zeros_like(model.banks.gates[m]))
        gc0 = model.banks.gcond[m].kernel.numpy()
        model.banks.gcond[m].kernel.assign(tf.zeros_like(model.banks.gcond[m].kernel))
        out[f"ko_{FAM}_{m}_s{SKIP + si}"] = score(run_all())
        model.banks.gates[m].assign(g0)
        model.banks.gcond[m].kernel.assign(gc0)
    print(f"[{FAM}] s{SKIP + si} base={out[f'base_{FAM}_s{SKIP + si}']*100:.2f}% "
          + " ".join(f"{m}:{(out[f'ko_{FAM}_{m}_s{SKIP + si}'] - out[f'base_{FAM}_s{SKIP + si}'])*100:+.2f}"
                     for m in KO_LIST), flush=True)

np.savez_compressed(os.path.join(OUT, f"fig1_ensv2_{TAG}.npz"), **out)
print("ENSV2_DONE", flush=True)
