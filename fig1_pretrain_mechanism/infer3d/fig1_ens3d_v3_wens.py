"""v3 BATCHED product-ensemble inference (08-12, user directive: the ensemble is
embarrassingly parallel -- batch it). One trunk call carries ALL |G| x K replicas;
the W-mean is per-flip (block mean over the K axis), each flip's rollout feeds back
in its own transformed frame, and the inverse-transform + flip average happens once
at the end. bf16 + optional XLA. Math identical to fig1_ens3d_v2 (verified: same
seeds -> same windows -> same per-replica trunk inputs).

  python -u fig1_ens3d_v3.py <ckpt> <out_dir>
  env: FAM, SKIP, N_SAMP, W_ENS=8, FLIPS=id,fx,fy,fz, OUT_TAG, SAVE_PRED, XLA=1
"""
import os
import sys
import time

import numpy as np
import tensorflow as tf

if int(os.environ.get("BF16", "0")):
    import tensorflow as _tf0
    _tf0.keras.mixed_precision.set_global_policy("mixed_bfloat16")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.pretrain import load_ckpt
from data import loader
from data.registry import FAMILIES, NUM_SLOTS, SEMANTIC_SLOTS
from v3.data_ar import _cell_coords, make_example_ar, stack_batch
from v3.model import V3Model, interp_box

KENS = int(os.environ.get("W_ENS", "8"))
SKIP = int(os.environ.get("SKIP", "0"))
N_SAMP = int(os.environ.get("N_SAMP", "1"))
FLIPS = os.environ.get("FLIPS", "id,fx,fy,fz").split(",")
G = len(FLIPS)
FAM = os.environ.get("FAM", "pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08")
BOX = tuple(int(x) for x in os.environ.get("BOX", "32,32,32").split(","))
K = len(BOX)
NC_W = int(np.prod(BOX))
QGRID = int(os.environ.get("QGRID", "0"))          # 0 = read out on the W box itself
QB = (QGRID,) * len(BOX) if QGRID else BOX
NC = int(np.prod(QB))
OUT = sys.argv[2] if len(sys.argv) > 2 else "/code-vol/motion_fv2/figs_m1s2b"
TAG = os.environ.get("OUT_TAG", "v3")

ck = sys.argv[1]
MD = int(os.environ.get("MD", "384"))          # model dims: m1 = 384/8/16, m2 = 896/12/48
MDEPTH = int(os.environ.get("MDEPTH", "8"))
MDW = int(os.environ.get("MDW", "16"))
model = V3Model(S=NUM_SLOTS, d=MD, depth=MDEPTH, d_w=MDW, n_p=16, d_cond=128,
                dims2=(64, 64), dims3=(32, 32, 32), t_in=10, transport=True,
                n_fams=len(FAMILIES), n_semantic=SEMANTIC_SLOTS,
                gate_cond=1, dict_decode=0, fam_head=1)
model.warm_build()
V = model.trainable_variables
load_ckpt(V, ck, d_w=MDW, names=[v.name for v in V])
print(f"loaded {ck}", flush=True)
rng = np.random.default_rng(1234)
cells1 = tf.constant(_cell_coords(BOX)[None].astype(np.float32))

keep = tf.concat([tf.ones([model.n_semantic]),
                  tf.zeros([model.S - model.n_semantic])], 0)[None, None]


import itertools as _it

_OHG = []                                 # full octahedral group O_h: 6 perms x 8 signs
_DET = []
for _perm in _it.permutations(range(3)):
    _P = np.zeros((3, 3), int)
    for _i, _pp in enumerate(_perm):
        _P[_i, _pp] = 1
    for _sg in _it.product([1, -1], repeat=3):
        _OHG.append((_perm, _sg))
        _DET.append(int(round(np.linalg.det(np.diag(_sg) @ _P))))
assert len(_OHG) == 48
_ROT24 = _OHG                             # r<i> tokens index the FULL 48-element group


def rot_fields(f, ri):
    """Proper rotation ri of a (T, X, Y, Z, S) field: grid axes permuted+flipped, the
    velocity slots (0,1,2) rewired with the same permutation and signs."""
    perm, sg = _ROT24[ri]
    v = np.transpose(f, (0,) + tuple(1 + p for p in perm) + (4,))
    for a in range(3):
        if sg[a] < 0:
            v = np.flip(v, axis=1 + a)
    v = np.array(v, copy=True)
    vel = v[..., :3].copy()
    for a in range(3):
        v[..., a] = sg[a] * vel[..., perm[a]]
    return np.ascontiguousarray(v)


def rot_inv_pred(pr, ri):
    """Inverse rotation of a (10, NC, S) readout-lattice prediction."""
    perm, sg = _ROT24[ri]
    v = pr.reshape((10,) + QB + (-1,))
    # forward was: axes->perm with signs; inverse: undo signs, then inverse-permute
    for a in range(3):
        if sg[a] < 0:
            v = np.flip(v, axis=1 + a)
    inv = [0, 0, 0]
    for a in range(3):
        inv[perm[a]] = a
    v = np.transpose(v, (0,) + tuple(1 + i for i in inv) + (4,))
    v = np.array(v, copy=True)
    vel = v[..., :3].copy()
    for a in range(3):
        v[..., perm[a]] = sg[a] * vel[..., a]
    return np.ascontiguousarray(v.reshape(10, NC, -1))



def _mat(g):
    """(perm, signs) of a spatial-group element, in the index convention of _ROT24."""
    if g == "id":
        return (0, 1, 2), (1, 1, 1)
    if g.startswith("r") and g[1:].isdigit():
        return _ROT24[int(g[1:])]
    perm, sg = (0, 1, 2), [1, 1, 1]
    for a, tok in enumerate(("fx", "fy", "fz")):
        if tok in g:
            sg[a] = -1
    return perm, tuple(sg)


def transform_batch(b, u0, g):
    """Apply a spatial-group element to an ALREADY BUILT example instead of rebuilding
    it from a transformed volume. The encoder then sees exactly T(identity input) on the
    same physical points, so the equivariance defect carries no encoder-sampling noise
    (which index-matched resampling cannot avoid: a transform moves the points)."""
    if g == "id":
        return b, u0
    if g.startswith("s") and g[1:].isdigit():
        n = int(g[1:])                                  # native cells along x
        out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in b.items()}
        out["coords_node"][..., 0] = (b["coords_node"][..., 0] + n / 128.0) % 1.0
        m = n // (128 // BOX[0])                        # -> query cells
        ub = np.roll(u0.reshape((1,) + BOX + (-1,)), m, axis=1)
        return out, np.ascontiguousarray(ub).reshape(u0.shape)
    perm, sg = _mat(g)
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in b.items()}
    c = b["coords_node"].copy()                       # (B, N, 3) in [0,1]
    # A reflection about the domain centre maps cell j to 127-j, which takes the
    # stride-4 scoring lattice {0,4,...} to {3,7,...} -- a different set (no even-stride
    # subset of an even grid is closed under it). Composing the reflection with a
    # one-cell translation, itself an exact symmetry of the periodic domain, restores
    # closure, so the transformed run is scored on exactly the same points.
    _eps_cell = 1.0 / 128.0
    for a in range(3):
        ca = c[..., perm[a]]
        out["coords_node"][..., a] = ca if sg[a] > 0 else (1.0 - ca + _eps_cell) % 1.0
    for key in ("x_win", "y_node"):                   # velocity slots rotate with the frame
        if key in b:
            v = b[key].copy()
            for a in range(3):
                out[key][..., a] = sg[a] * v[..., perm[a]]
    ub = u0.reshape((1,) + BOX + (-1,)).copy()        # anchor lives on the box lattice
    ub = np.transpose(ub, (0,) + tuple(1 + p for p in perm) + (4,))
    for a in range(3):
        if sg[a] < 0:                                  # flip + one-site roll: the same
            ub = np.roll(np.flip(ub, axis=1 + a), 1, axis=1 + a)   # closure fix
    ub = np.array(ub, copy=True)
    vel = ub[..., :3].copy()
    for a in range(3):
        ub[..., a] = sg[a] * vel[..., perm[a]]
    return out, ub.reshape(u0.shape)

def flip_fields(f, g):                                    # (T, *sp, S)
    f = np.array(f, np.float32, copy=True)
    if "fx" in g: f = f[:, ::-1];       f[..., 0] *= -1
    if "fy" in g: f = f[:, :, ::-1];    f[..., 1] *= -1
    if K == 3 and "fz" in g: f = f[:, :, :, ::-1]; f[..., 2] *= -1
    return np.ascontiguousarray(f)


def inv_pred(p, g):                                       # (10, NC, S) readout lattice
    p = p.reshape((10,) + QB + (-1,)).copy()
    if "fx" in g: p = p[:, ::-1];       p[..., 0] *= -1
    if "fy" in g: p = p[:, :, ::-1];    p[..., 1] *= -1
    if K == 3 and "fz" in g: p = p[:, :, :, ::-1]; p[..., 2] *= -1
    return p.reshape(10, NC, -1)



# ---- W-space group averaging (08-17) -------------------------------------------------
# The symmetry axis is normally averaged AFTER decoding, in field space, because the
# latent box is not known to transform covariantly.  W_GROUP_AVG=1 tests that directly:
# every element's W box is mapped back to the identity frame, averaged there, and the
# single averaged box is decoded.  W_VEC_SLOTS=n additionally rewires the first n latent
# channels as a vector under the same element (n=0 leaves channels untouched).
W_GROUP_AVG = int(os.environ.get("W_GROUP_AVG", "0"))
W_VEC_SLOTS = int(os.environ.get("W_VEC_SLOTS", "0"))


def _box_act(box, g, inverse):
    """Act with group element g on a (B,*BOX,d_w) latent box: spatial axes permuted and
    flipped, and (if W_VEC_SLOTS>=3) the first three channels rewired as a vector."""
    perm, sg = _mat(g)
    inv = [0, 0, 0]
    for a in range(3):
        inv[perm[a]] = a
    if inverse:
        for a in range(3):
            if sg[a] < 0:
                box = tf.reverse(box, [1 + a])
        box = tf.transpose(box, [0] + [1 + i for i in inv] + [4])
    else:
        box = tf.transpose(box, [0] + [1 + p for p in perm] + [4])
        for a in range(3):
            if sg[a] < 0:
                box = tf.reverse(box, [1 + a])
    if W_VEC_SLOTS >= 3:
        v = box[..., :3]
        cols = [None, None, None]
        if inverse:
            for a in range(3):
                cols[perm[a]] = sg[a] * v[..., a:a + 1]
        else:
            for a in range(3):
                cols[a] = sg[a] * v[..., perm[a]:perm[a] + 1]
        box = tf.concat(cols + [box[..., 3:]], -1)
    return box


def _w_group_mean(zbox0, flips):
    """Map every element back to the identity frame and average there."""
    back = [_box_act(zbox0[i:i + 1], flips[i], True) for i in range(len(flips))]
    return tf.reduce_mean(tf.concat(back, 0), 0, keepdims=True)   # (1,*BOX,d_w)


_ROLL_CACHE = {}


def _make_roll(Gc):
    FLIPS_C = globals().get("FLIPS_CHUNK", FLIPS[:Gc])
    # One compiled rollout per chunk width: static shapes let XLA fuse the whole
    # 10-step loop (the dynamic tf.shape version could not be jitted).
    @tf.function(jit_compile=bool(int(os.environ.get("XLA", "0"))),
                 reduce_retracing=True)
    def _roll(cn, xw, fm, gn, cond, op, fid, fbm, u0, roles, cellsG):
        """All G*K replicas in ONE batch. Rows are ordered flip-major: replica r belongs
        to flip r // KENS. zm is the per-flip block mean; each flip block feeds back from
        its own zm; u_prev holds G cell-lattice states advanced in their own frames."""
        win, fmask = xw, fm
        u_prev = u0                                            # (G, NC, S)
        fb = tf.cast(fbm, tf.float32)[:, None, :]              # (G*K, 1, S)
        outs = tf.TensorArray(tf.float32, size=10)
        for k in tf.range(10):
            Wp, e = model.trunk_W(cn, win, fmask, gn, K, list(roles), cond, op,
                                  box_dims=BOX)
            z1 = tf.reduce_mean(tf.cast(Wp, tf.float32), -1)   # (G*K, *BOX, d_w)
            zg = tf.reshape(z1, tf.concat([[Gc, KENS], BOX, [model.d_w]], 0))
            zm = tf.reduce_mean(zg, 1)                         # (G, *BOX, d_w) per-flip mean
            zbox0 = tf.reshape(zm, tf.concat([[Gc], BOX, [model.d_w]], 0))
            if W_GROUP_AVG:
                # average the tendency itself, in the identity frame, then push the one
                # averaged box back out to every element's frame for decode and feedback
                wbar = _w_group_mean(zbox0, FLIPS_C)
                zbox0 = tf.concat([_box_act(wbar, g_, False) for g_ in FLIPS_C], 0)
                zm = tf.reshape(zbox0, tf.concat([[Gc], BOX, [model.d_w]], 0))
            zc = (tf.reshape(zm, [Gc, NC, model.d_w]) if not QGRID
                  else tf.cast(interp_box(zbox0, cellsG, BOX), tf.float32))
            eg = tf.reshape(e, (Gc, KENS, -1))[:, 0]            # one cond row per flip
            fidg = fid[::KENS]
            d_c = tf.cast(model.dec(zc, coords=cellsG, roles=list(roles),
                                    geom=tf.zeros([Gc, NC, 8]),
                                    cond=eg, fam_id=fidg), tf.float32)
            u_prev = u_prev * keep + d_c * fb[::KENS]
            u_prev = tf.ensure_shape(u_prev, [None, NC, None])
            outs = outs.write(k, u_prev)
            zmK = tf.reshape(tf.tile(zbox0[:, None], [1, KENS] + [1] * (K + 1)),
                             tf.concat([[Gc * KENS], BOX, [model.d_w]], 0))
            zn = tf.cast(interp_box(zmK, cn, BOX), tf.float32)
            d_n = tf.cast(model.dec(zn, coords=cn, roles=list(roles),
                                    geom=tf.cast(gn, tf.float32), cond=e, fam_id=fid),
                          tf.float32)
            u_n = win[:, -1] * keep + d_n * fb
            win = tf.concat([win[:, 1:], u_n[:, None]], 1)
            fmask = tf.concat([fmask[:, 1:], tf.ones_like(fmask[:, :1])], 1)
        return outs.stack()                                    # (10, G, NC, S)


    _ROLL_CACHE[(Gc, tuple(FLIPS_C))] = _roll
    return _roll


def ens_roll_all(cn, xw, fm, gn, cond, op, fid, fbm, u0, roles, cellsG):
    Gc = int(u0.shape[0])
    key = (Gc, tuple(globals().get("FLIPS_CHUNK", FLIPS[:Gc])))
    fn = _ROLL_CACHE.get(key) or _make_roll(Gc)
    return fn(cn, xw, fm, gn, cond, op, fid, fbm, u0, roles, cellsG)



it = iter(loader.units_for(FAM, "val"))
for _ in range(SKIP):
    # SKIP IS FREE (08-12): the old path rebuilt K*|G| examples per skipped sample --
    # each one gathering from a ~2.8 GB unit, with a full-volume copy per flip -- purely
    # to keep one shared RNG stream aligned. Per-sample seeding gives the same
    # reproducibility at zero cost (it was ~80% of the wall-clock at SKIP=7).
    next(it)
rng = np.random.default_rng(1234 + 977 * SKIP
                           + 100003 * sum(os.environ.get("NOISE_TAG", "").encode()))
out = {}
for si in range(N_SAMP):
    s = next(it)
    fields = np.asarray(s["fields"], np.float32)
    dims = tuple(int(d) for d in s["dims"])
    vol = fields.reshape((fields.shape[0],) + dims + (fields.shape[-1],))
    _bc = int(os.environ.get("BOOST", "0"))               # cells/frame along BOOST_AX
    if _bc:
        # GALILEAN BOOST, exact on the periodic lattice: u'(x,t) = u(x - Vt, t) + V with
        # V = _bc cells/frame. The two signs go together -- the pattern is carried in the
        # direction of the added velocity, so frame t rolls FORWARD (towards increasing
        # index) by _bc*t cells. Pairing a backward roll with +V is not a symmetry: it
        # leaves a continuity residual 2V.grad(rho) and makes a covariant operator score
        # WORST, since it would advect the pattern the other way. Both the observed window
        # and the scored future are boosted, so this asks whether the operator solves the
        # SAME physics in a moving frame.
        _ax = int(os.environ.get("BOOST_AX", "0"))
        _V = _bc * (1.0 / dims[_ax]) / (1.0 / 20.0)       # dx/dt in physical units
        vol = np.stack([np.roll(vol[t], _bc * t, axis=_ax) for t in range(vol.shape[0])])
        vol = np.array(vol, copy=True)
        vol[..., _ax] += _V
        fields = vol.reshape(fields.shape)
        s = {**{k2: s[k2] for k2 in s.keys()}, "fields": fields}
        flat_boost = fields.reshape(fields.shape[0], -1, fields.shape[-1])
        print(f"[boost] {_bc} cell/frame on axis {_ax} -> V={_V:.4f}", flush=True)
    flat = fields.reshape(fields.shape[0], -1, fields.shape[-1])
    nat = dims[0]
    st = max(1, nat // QB[0])                 # scoring/readout lattice, not the W box
    ax = np.arange(0, nat, st)[:QB[0]]
    if K == 3:
        gi = (ax[:, None, None] * dims[1] * dims[2] + ax[None, :, None] * dims[2]
              + ax[None, None, :]).ravel()
    else:
        gi = (ax[:, None] * dims[1] + ax[None, :]).ravel()
    y_cells = flat[10:20][:, gi]
    if QGRID:
        _q1 = (ax.astype(np.float32) + 0.5) / float(nat)      # data sample sites
        _qcoords = np.stack(np.meshgrid(_q1, _q1, _q1, indexing="ij"),
                            -1).reshape(-1, 3).astype(np.float32)
    else:
        _qcoords = None
    _t0 = time.time()
    bs_all, u0s = [], []
    XFI = int(os.environ.get("INPUT_TRANSFORM", "0"))
    if XFI:
        # The group acts on the SAMPLED POINT SET, not on the volume: build the
        # identity replicas once, then transform their coordinates and vector
        # components. Every transform then sees the same physics at the same
        # physical points, so the defect measures equivariance alone (index-matched
        # resampling cannot achieve this -- a transform moves the points).
        _exs = [make_example_ar(s, 10, 10, 4, 512, 8192, rng, box_dims=BOX)
                for _ in range(KENS)]
        _bs = [stack_batch([e]) for e in _exs]
        _u0 = flat[9][gi] / np.asarray(_bs[0]["scale"], np.float32)[0]
        for g in FLIPS:
            _tu = None
            for b in _bs:
                tb, _tu = transform_batch(b, _u0, g)
                bs_all.append(tb)
            u0s.append(_tu)
    else:
        _seed_g = int(rng.integers(2 ** 31))
        for g in FLIPS:
            if int(os.environ.get("SEED_PER_G", "0")):
                rng = np.random.default_rng(_seed_g)   # same node draws under every transform
            if g.startswith("r") and g[1:].isdigit():
                tv = rot_fields(vol, int(g[1:]))
            elif g.startswith("s") and g[1:].isdigit():
                # TRANSLATION: exact on the periodic lattice AND on the physics; on the
                # coarse coefficient lattice it moves the sampling phase, so averaging over
                # shifts cancels truncation artifacts rather than model noise.
                _n = int(g[1:])
                tv = np.roll(vol, (_n, _n, _n), axis=(1, 2, 3))
            elif g == "id":
                tv = None
            else:
                tv = flip_fields(vol, g)
            sg = s if tv is None else {**{k2: s[k2] for k2 in s.keys()},
                                       "fields": tv.reshape(flat.shape)}
            exs = [make_example_ar(sg, 10, 10, 4, 512, 8192, rng, box_dims=BOX)
                   for _ in range(KENS)]
            bs_all += [stack_batch([e]) for e in exs]
            fg = np.asarray(sg["fields"], np.float32).reshape(flat.shape)
            _anchor = (flat[9][gi] if (g.startswith("s") and g[1:].isdigit())
                       else fg[9][gi])
            u0s.append(_anchor / np.asarray(bs_all[-1]["scale"], np.float32)[0])
    p0 = bs_all[0]
    CH_G = max(1, int(os.environ.get("CHUNK_G", "4")))
    if W_GROUP_AVG and CH_G < G:
        raise SystemExit("W_GROUP_AVG needs CHUNK_G >= G (the mean spans the whole group)")     # flips per GPU chunk (exact:
    # feedback couples only within a flip, so chunking whole flip groups changes nothing)
    pr_parts = []
    for c0 in range(0, G, CH_G):
        c1 = min(G, c0 + CH_G)
        idx = slice(c0 * KENS, c1 * KENS)
        bs_c = bs_all[idx]
        globals()["G_CUR"] = c1 - c0
        globals()["FLIPS_CHUNK"] = FLIPS[c0:c1]
        _cc = (_cell_coords(BOX)[None] if not QGRID else
           _qcoords[None]).astype(np.float32)
        _cg = []
        for gg in FLIPS[c0:c1]:
            if (gg.startswith("s") and gg[1:].isdigit()
                    and not int(os.environ.get("INPUT_TRANSFORM", "0"))):
                # volume-shift path only: decode back in place. Under INPUT_TRANSFORM the
                # shift already lives in the input coordinates, and shifting the query
                # points as well would apply the translation twice.
                _cg.append((_cc - int(gg[1:]) / float(nat)) % 1.0)
            else:
                _cg.append(_cc)
        cellsG = tf.constant(np.concatenate(_cg, 0))
        cat = lambda k2: tf.constant(np.concatenate([b[k2] for b in bs_c], 0))
        pr_c = ens_roll_all(cat("coords_node"), cat("x_win"), cat("fmask"),
                            cat("geom_node"), cat("cond"), cat("op_multihot"),
                            cat("fam_id"),
                            tf.constant(np.concatenate([b["fbmask"] for b in bs_c], 0)),
                            tf.constant(np.stack(u0s[c0:c1]), tf.float32),
                            tuple(int(r) for r in p0["roles"]), cellsG).numpy()
        pr_parts.append(pr_c)
    pr = np.concatenate(pr_parts, axis=1)                   # (10, G, NC, S)
    _dt = time.time() - _t0
    print(f"[timing] sample {SKIP+si}: {_dt:.1f}s total | {_dt/max(1,G):.2f}s per "
          f"replica-group | G={G} K={KENS} XLA={os.environ.get('XLA','0')}", flush=True)
    def _inv(block, g):
        if g.startswith("r") and g[1:].isdigit():
            return rot_inv_pred(block, int(g[1:]))
        if g.startswith("s") and g[1:].isdigit():
            if not int(os.environ.get("INPUT_TRANSFORM", "0")):
                return block        # decoded back in place by the shifted query coords
            m = int(g[1:]) // (128 // BOX[0])
            bb = block.reshape((10,) + QB + (-1,))
            return np.ascontiguousarray(np.roll(bb, -m, axis=1).reshape(10, NC, -1))
        return inv_pred(block, g)
    inv_all = [_inv(pr[:, gi_], FLIPS[gi_]) for gi_ in range(G)]
    pb = np.mean(inv_all, 0)
    SUBSETS = {}
    if all(g.startswith("r") for g in FLIPS) and G == 48:
        idxs = [int(g[1:]) for g in FLIPS]
        id_perm = tuple(range(3))
        SUBSETS["flips8"] = [k for k, gi_ in enumerate(idxs) if _OHG[gi_][0] == id_perm]
        SUBSETS["rot24"] = [k for k, gi_ in enumerate(idxs) if _DET[gi_] == 1]
    sc = np.asarray(p0["scale"], np.float32)[0]
    cm = np.asarray(p0["cmask"], np.float32)[0][None, None]
    pp = pb * sc[None, None]
    num = np.sqrt((((pp - y_cells) * cm) ** 2).sum((1, 2)))
    den = np.sqrt(((y_cells * cm) ** 2).sum((1, 2))) + 1e-12
    rel = float((num / den).mean())
    out[f"base_{FAM}_s{SKIP + si}"] = rel
    for nm_, ks_ in SUBSETS.items():
        pps = np.mean([inv_all[k] for k in ks_], 0) * sc[None, None]
        n_ = np.sqrt((((pps - y_cells) * cm) ** 2).sum((1, 2)))
        r_ = float((n_ / den).mean())
        out[f"{nm_}_{FAM}_s{SKIP + si}"] = r_
        print(f"    [subset {nm_}|K{KENS}] rel={r_*100:.2f}%", flush=True)
    if int(os.environ.get("SAVE_PERG", "0")):
        for _gi, _g in enumerate(FLIPS):
            out[f"perg_{_g}_s{SKIP + si}"] = (inv_all[_gi] * sc[None, None]).astype(np.float32)
    if int(os.environ.get("SAVE_PRED", "0")):
        out[f"pred_{FAM}_s{SKIP + si}"] = pp.astype(np.float32)
        out[f"gt_{FAM}_s{SKIP + si}"] = y_cells.astype(np.float32)
    print(f"[{FAM}] s{SKIP + si} base={rel*100:.2f}% (batched G{G}xK{KENS})", flush=True)
np.savez_compressed(os.path.join(OUT, f"fig1_ensv3_{TAG}.npz"), **out)
print("ENSV3_DONE", flush=True)
