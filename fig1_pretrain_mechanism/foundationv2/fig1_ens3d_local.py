"""LOCAL flips8 product-ensemble scorer (08-12): sub-K x isotropic flip group incl.
COMPOSITES on the ft3d ckpt, M1.0 val 8, run on the local GPU (RTX 4070 Ti) while the
cluster H100s are saturated. Units are rebuilt from the exported val_{i}.npz dumps
(fields (21,128,128,128,5) physical [vx,vy,vz,rho,p] -> NUM_SLOTS-wide grid unit).

  ~/tf215venv/bin/python fig1_ens3d_local.py <ckpt.npz> <val_dir> <out_dir>
  env: W_ENS=8, FLIPS=id,fx,fy,fz,fxfy,fxfz,fyfz,fxfyfz, N_SAMP=8, SKIP=0
"""
import os
import sys

import numpy as np
import tensorflow as tf

for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")   # cluster convention; halves
# the banks stencil activations, which is what blows the 12GB card in fp32

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.pretrain import load_ckpt
from data.registry import FAMILIES, NUM_SLOTS, SEMANTIC_SLOTS
from v3.data_ar import _cell_coords, make_example_ar, stack_batch
from v3.model import V3Model, interp_box

KENS = int(os.environ.get("W_ENS", "8"))
FLIPS = os.environ.get("FLIPS", "id,fx,fy,fz,fxfy,fxfz,fyfz,fxfyfz").split(",")
N_SAMP = int(os.environ.get("N_SAMP", "8"))
SKIP = int(os.environ.get("SKIP", "0"))
FAM = "pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08"
BOX = (32, 32, 32)
NC = int(np.prod(BOX))
CK, VAL_DIR = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else "."

model = V3Model(S=NUM_SLOTS, d=384, depth=8, d_w=16, n_p=16, d_cond=128,
                dims2=(64, 64), dims3=(32, 32, 32), t_in=10, transport=True,
                n_fams=len(FAMILIES), n_semantic=SEMANTIC_SLOTS,
                gate_cond=1, dict_decode=0, fam_head=1)
model.warm_build()
V = model.trainable_variables
load_ckpt(V, CK, d_w=16, names=[v.name for v in V])
print(f"loaded {CK}", flush=True)
rng = np.random.default_rng(1234)
cellsT = tf.constant(_cell_coords(BOX)[None].astype(np.float32))
gqC = tf.zeros([1, NC, 8], tf.float32)
keep = tf.concat([tf.ones([model.n_semantic]),
                  tf.zeros([model.S - model.n_semantic])], 0)[None, None]


from data import symbolic as SY

_COND = SY.encode(FAMILIES[FAM])


def unit_from_npz(path):
    vol = np.load(path)["fields"].astype(np.float32)      # (21,128,128,128,5)
    T = vol.shape[0]
    f = np.zeros((T, 128 ** 3, NUM_SLOTS), np.float32)
    f[..., :5] = vol.reshape(T, -1, 5)
    cm = np.zeros(NUM_SLOTS, bool)
    cm[:5] = True
    return {"family": FAM, "mode": "grid", "K": 3, "fields": f, "cmask": cm,
            "dims": np.array([128, 128, 128]), "tmask": np.ones(T, np.float32),
            "roles": np.array(FAMILIES[FAM].roles),
            "op_multihot": _COND["op_multihot"],
            "op_ids": _COND["op_ids"], "param_ids": _COND["param_ids"],
            "param_feats": _COND["param_feats"], "param_mask": _COND["param_mask"]}


def flip_fields(f, g):                                    # (T, N, S) flat -> flip via 4D view
    v = f.reshape(f.shape[0], 128, 128, 128, -1)
    v = np.array(v, copy=True)
    if "fx" in g: v = v[:, ::-1];       v[..., 0] *= -1
    if "fy" in g: v = v[:, :, ::-1];    v[..., 1] *= -1
    if "fz" in g: v = v[:, :, :, ::-1]; v[..., 2] *= -1
    return np.ascontiguousarray(v.reshape(f.shape))


def inv_pred(p, g):                                       # (10, NC, S) on BOX lattice
    p = p.reshape((10,) + BOX + (-1,)).copy()
    if "fx" in g: p = p[:, ::-1];       p[..., 0] *= -1
    if "fy" in g: p = p[:, :, ::-1];    p[..., 1] *= -1
    if "fz" in g: p = p[:, :, :, ::-1]; p[..., 2] *= -1
    return p.reshape(10, NC, -1)


NCHUNK = int(os.environ.get("CHUNKS", "8"))              # trunk in K/NCHUNK-sized chunks
CSZ = KENS // NCHUNK


def ens_roll(cn, xw, fm, gn, cond, op, fid, fbm, u0, roles):
    """EAGER K-sequential path for the 12GB local card: each trunk call sees CSZ
    replicas and its buffers free immediately; zm = mean over ALL K, so the ensemble
    math is identical to the one-batch H100 graph path."""
    win, fmask = xw, fm
    u_prev = u0
    fb = tf.cast(fbm, tf.float32)[:, None, :]
    outs = []
    sl = lambda t, i: t[i * CSZ:(i + 1) * CSZ]
    for k in range(10):
        zs, es = [], []
        for i in range(NCHUNK):
            Wp, e_i = model.trunk_W(sl(cn, i), sl(win, i), sl(fmask, i), sl(gn, i),
                                    3, list(roles), cond[:CSZ], op[:CSZ],
                                    box_dims=BOX)
            zs.append(tf.reduce_sum(tf.reduce_mean(tf.cast(Wp, tf.float32), -1),
                                    0, keepdims=True))
            es.append(e_i)
        zm = tf.add_n(zs) / float(KENS)
        zc = tf.reshape(zm, [1, NC, model.d_w])
        d_c = tf.cast(model.dec(zc, coords=cellsT, roles=list(roles), geom=gqC,
                                cond=es[0][:1], fam_id=fid[:1]), tf.float32)
        u_prev = u_prev * keep + d_c * fb[:1]
        outs.append(u_prev[0].numpy())
        zbox = tf.reshape(zm, (1,) + BOX + (model.d_w,))
        new_win, new_fm = [], []
        for i in range(NCHUNK):
            zboxK = tf.tile(zbox, [CSZ, 1, 1, 1, 1])
            zn = tf.cast(interp_box(zboxK, sl(cn, i), BOX), tf.float32)
            d_n = tf.cast(model.dec(zn, coords=sl(cn, i), roles=list(roles),
                                    geom=tf.cast(sl(gn, i), tf.float32), cond=es[i],
                                    fam_id=fid[:CSZ]), tf.float32)
            u_n = sl(win, i)[:, -1] * keep + d_n * fb[:CSZ]
            new_win.append(tf.concat([sl(win, i)[:, 1:], u_n[:, None]], 1))
            new_fm.append(tf.concat([sl(fmask, i)[:, 1:],
                                     tf.ones_like(sl(fmask, i)[:, :1])], 1))
        win = tf.concat(new_win, 0)
        fmask = tf.concat(new_fm, 0)
    return np.stack(outs)


nat = 128
st = nat // BOX[0]
axg = np.arange(0, nat, st)[:BOX[0]]
gi = (axg[:, None, None] * nat * nat + axg[None, :, None] * nat
      + axg[None, None, :]).ravel()
out = {}
for si in range(SKIP, SKIP + N_SAMP):
    s = unit_from_npz(os.path.join(VAL_DIR, f"val_{si}.npz"))
    flat = s["fields"]
    y_cells = flat[10:20][:, gi]
    preds = []
    p0 = None
    for g in FLIPS:
        sg = s if g == "id" else {**s, "fields": flip_fields(flat, g)}
        exs = [make_example_ar(sg, 10, 10, 4, 512, 8192, rng, box_dims=BOX)
               for _ in range(KENS)]
        bs = [stack_batch([e]) for e in exs]
        cat = lambda k: tf.constant(np.concatenate([b[k] for b in bs], 0))
        u0g = sg["fields"][9][gi][None] / np.asarray(bs[0]["scale"], np.float32)[0][None, None]
        pr = ens_roll(cat("coords_node"), cat("x_win"), cat("fmask"),
                      cat("geom_node"), cat("cond"), cat("op_multihot"),
                      cat("fam_id"), tf.constant(bs[0]["fbmask"]),
                      tf.constant(u0g, tf.float32),
                      tuple(int(r) for r in bs[0]["roles"]))
        preds.append(inv_pred(pr, g))
        if p0 is None:
            p0 = bs[0]
        print(f"  s{si} flip {g} done", flush=True)
    pb = np.mean(preds, 0)
    sc = np.asarray(p0["scale"], np.float32)[0]
    cmv = np.asarray(p0["cmask"], np.float32)[0][None, None]
    pp = pb * sc[None, None]
    # paper-standard: whole-window joint rel; also per-frame mean for continuity
    reljoint = 100 * np.linalg.norm((pp - y_cells) * cmv) / (np.linalg.norm(y_cells * cmv) + 1e-12)
    num = np.sqrt((((pp - y_cells) * cmv) ** 2).sum((1, 2)))
    den = np.sqrt(((y_cells * cmv) ** 2).sum((1, 2))) + 1e-12
    relpf = 100 * float((num / den).mean())
    out[f"joint_s{si}"] = reljoint
    out[f"pf_s{si}"] = relpf
    print(f"[{FAM}] s{si} ens{KENS}x{len(FLIPS)}: joint {reljoint:.2f}% | per-frame {relpf:.2f}%",
          flush=True)
js = [out[f"joint_s{k}"] for k in range(SKIP, SKIP + N_SAMP)]
pf = [out[f"pf_s{k}"] for k in range(SKIP, SKIP + N_SAMP)]
print(f"=== ens{KENS}x{len(FLIPS)} PAPER MEAN joint {np.mean(js):.2f}% (med {np.median(js):.2f})"
      f" | per-frame mean {np.mean(pf):.2f}% (med {np.median(pf):.2f})", flush=True)
np.savez_compressed(os.path.join(OUT, f"ens8_local_{SKIP}_{N_SAMP}.npz"), **out)
print("LOCAL_ENS8_DONE", flush=True)
