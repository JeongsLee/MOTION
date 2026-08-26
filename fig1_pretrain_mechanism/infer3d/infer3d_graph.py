"""IN-GRAPH inference-only ensemble for the 3D CNS families (08-14, user directive:
"추론전용 ... 새로운 파일로").  Nothing here is imported by training: it is a separate
entry point that reuses the trained weights and the model's own trunk/decoder.

WHY THIS FILE EXISTS.  fig1_ens3d_v3 builds one example per replica on the HOST — for
R=24 that is 24 numpy gathers out of a 587 MB volume plus 24 host->device transfers.
Measured on m2s2 (24 replicas, 32^3 box, H100):

    IT=0  volume rotated per element, 24 host builds   190.4 s   (7.93 s/replica)
    IT=2  no volume rotation,        24 host builds     57.5 s   (2.40 s/replica, XLA)
    IT=1  no volume rotation,         1 host build      37.4 s   (1.56 s/replica, XLA)
                                                        ^^^^^^ the 0.84 s/replica gap
                                                        is exactly the host build.

IT=1 pays 0 for the builds but shares ONE point set across all replicas, which costs
accuracy (s0 rel-L2 8.72 % vs 7.03 % when each replica draws its own points).  This file
removes the host build WITHOUT removing the diversity: the point draw, the gather, the
normalization and the group action all happen inside the graph, so every replica still
sees its own points and the trunk sees one fused batch.

WHAT IS AND IS NOT REPLICATED.  The rollout, decoder and readout are the model's own
(imported, not reimplemented).  Only the example *builder* is re-expressed in TF, and
only the part inference needs:

  * y_node / y_q / ic_q targets are NOT built.  They are supervision, and inference
    scores against `y_cells` gathered separately -- in the host builder they cost a
    second gather the same size as x_win, for nothing.
  * geom is zero for the CNS families (no sdf_vol, geom_kind != interface), so the
    geometry pathway is fed zeros, exactly as the host builder does for this family.
  * `_pick` with no wall weights is `rng.integers(0, N, size=n)` -- uniform WITH
    replacement -- so tf.random.uniform reproduces it.  The host builder additionally
    de-duplicates (np.unique) and pads back to enc_n; at N=2.1e6, n=8192 the collision
    rate is ~0.2 %, so the draws are distributionally the same but NOT bitwise equal to
    a given numpy seed.  Verification is therefore distributional (see VERIFY below).

VERIFY.  Run with ENS=id (single replica, no transform) and compare against
fig1_ens3d_v3's per-sample `base` for the same family: the two must agree to within the
sampling spread of a single draw.  Then run the 24-element rotation set and compare the
ensemble number against the measured diagonal-24 (s0 rel-L2 7.03 %, 8-sample mean 10.05
/ median 7.38).  A systematic offset means a convention was mis-copied -- most likely
the cell-coordinate origin or the reflection's one-cell closure shift.

  python -u infer3d_graph.py <ckpt.npz> <out_dir>
  env: FAM, SKIP, N_SAMP, ENS (id | rot24), BOX, ENC_N, ARCH_D/ARCH_DEPTH/ARCH_DW,
       XLA, BF16, CHUNK_R (replicas per GPU chunk; >=36 OOMs at d_w=48), SAVE_PRED
"""
import itertools as _it
import os
import sys
import time

import numpy as np
import tensorflow as tf

if int(os.environ.get("BF16", "1")):
    tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.pretrain import load_ckpt                                    # noqa: E402
from data import loader                                                # noqa: E402
from data.registry import FAMILIES, NUM_SLOTS, SEMANTIC_SLOTS          # noqa: E402
from v3.data_ar import _cell_coords, make_example_ar, stack_batch      # noqa: E402
from v3.model import V3Model, interp_box                               # noqa: E402

FAM = os.environ.get("FAM", "pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08")
BOX = tuple(int(x) for x in os.environ.get("BOX", "32,32,32").split(","))
K = len(BOX)
NC = int(np.prod(BOX))          # latent box cell count
ENC_N = int(os.environ.get("ENC_N", "8192"))
T_IN = 10
# QGRID decouples the READOUT lattice from the latent box: the decoder is coordinate-based,
# so the solution instance can be evaluated on any grid without re-running the encoder or
# the bank.  Panel b of Fig. 1 displays 3D isosurfaces on a 64^3 lattice while the latent
# box for these families stays at 32^3, which is what the checkpoint was trained with.
QGRID = int(os.environ.get("QGRID", "0"))
QB = (QGRID,) * 3 if QGRID else None
PANEL_B = int(os.environ.get("PANEL_B", "0"))     # write b3d_<fam>_{gt,pred} volumes
PANEL_LEAD = int(os.environ.get("PANEL_LEAD", "5"))   # 1-based predicted step to display
NCQ = int(np.prod(QB)) if QGRID else NC   # readout cell count
SKIP = int(os.environ.get("SKIP", "0"))
N_SAMP = int(os.environ.get("N_SAMP", "1"))
CHUNK_R = int(os.environ.get("CHUNK_R", "24"))
OUT = sys.argv[2] if len(sys.argv) > 2 else "/corpus/results/_infer3d"
TAG = os.environ.get("OUT_TAG", "graph")

# ---------------------------------------------------------------- octahedral group
_OHG, _DET = [], []
for _perm in _it.permutations(range(3)):
    _P = np.zeros((3, 3), int)
    for _i, _pp in enumerate(_perm):
        _P[_i, _pp] = 1
    for _sg in _it.product([1, -1], repeat=3):
        _OHG.append((_perm, _sg))
        _DET.append(int(round(np.linalg.det(np.diag(_sg) @ _P))))
ROT24 = [i for i in range(48) if _DET[i] == 1]                   # proper rotations only
ENS = os.environ.get("ENS", "rot24")
ELEMS = [0] if ENS == "id" else ROT24
R = len(ELEMS)
PERM = np.array([_OHG[i][0] for i in ELEMS], np.int32)           # (R,3)
SIGN = np.array([_OHG[i][1] for i in ELEMS], np.float32)         # (R,3)

# ---------------------------------------------------------------- model
_D = int(os.environ.get("ARCH_D", "384"))
_DEP = int(os.environ.get("ARCH_DEPTH", "8"))
_DW = int(os.environ.get("ARCH_DW", "16"))
print(f"[arch] d={_D} depth={_DEP} d_w={_DW} | ENS={ENS} R={R} box={BOX}", flush=True)
model = V3Model(S=NUM_SLOTS, d=_D, depth=_DEP, d_w=_DW, n_p=16, d_cond=128,
                dims2=(64, 64), dims3=BOX, t_in=T_IN, transport=True,
                n_fams=len(FAMILIES), n_semantic=SEMANTIC_SLOTS,
                gate_cond=1, dict_decode=0, fam_head=1)
model.warm_build()
V = model.trainable_variables
load_ckpt(V, sys.argv[1], d_w=_DW, names=[v.name for v in V])
print(f"loaded {sys.argv[1]}", flush=True)
keep = tf.concat([tf.ones([model.n_semantic]),
                  tf.zeros([model.S - model.n_semantic])], 0)[None, None]


# ------------------------------------------------------- in-graph example assembly
@tf.function(reduce_retracing=True)
def draw_and_transform(flat_win, dims, scale, seed, perm, sign):
    """Draw ENC_N nodes per replica, gather the window, and apply the group action.

    flat_win : (T_IN, N_all, S) float32, the input window on the full lattice
    returns  : cn (R, ENC_N, 3), xw (R, T_IN, ENC_N, S)

    The group acts on the SAMPLED POINTS, not on the volume: rotating the volume and
    then sampling is the same map as sampling and then rotating the coordinates, and
    the second costs a gather of ENC_N points instead of a copy of the whole field.
    """
    n_all = tf.shape(flat_win)[1]
    ni = tf.random.stateless_uniform([R, ENC_N], seed=seed, minval=0,
                                     maxval=n_all, dtype=tf.int32)
    xw = tf.gather(flat_win, ni, axis=1)                     # (T, R, ENC_N, S)
    xw = tf.transpose(xw, [1, 0, 2, 3]) / scale              # (R, T, ENC_N, S)
    # flat index -> cell-centre coordinates, C-order, matching _cell_coords()
    d0, d1, d2 = dims[0], dims[1], dims[2]
    i0 = ni // (d1 * d2)
    i1 = (ni // d2) % d1
    i2 = ni % d2
    c = tf.stack([(tf.cast(i0, tf.float32) + 0.5) / tf.cast(d0, tf.float32),
                  (tf.cast(i1, tf.float32) + 0.5) / tf.cast(d1, tf.float32),
                  (tf.cast(i2, tf.float32) + 0.5) / tf.cast(d2, tf.float32)], -1)
    # axis permutation, then reflection about the domain centre.  The reflection is
    # composed with a one-cell translation so the transformed run lands on the same
    # scoring lattice (an even-stride subset of an even grid is not closed under a
    # bare reflection) -- the same closure fix the host path applies.
    cp = tf.gather(c, perm, axis=-1, batch_dims=1)           # (R, ENC_N, 3)
    eps = 1.0 / 128.0
    sgn = sign[:, None, :]
    cn = tf.where(sgn > 0, cp, tf.math.floormod(1.0 - cp + eps, 1.0))
    # velocity slots rotate with the frame: u'_a = sg_a * u_{perm[a]}
    vel = tf.gather(xw[..., :3], perm, axis=-1, batch_dims=1)   # (R, T, ENC_N, 3)
    vel = vel * sign[:, None, None, :]
    xw = tf.concat([vel, xw[..., 3:]], -1)
    return cn, xw


def rot_inv_pred(pr, ei):
    """Inverse of element ei on a (10, NCQ, S) readout-lattice prediction (host, cheap)."""
    perm, sg = _OHG[ei]
    _L = QB if QGRID else BOX
    v = pr.reshape((10,) + _L + (-1,))
    for a in range(3):
        if sg[a] < 0:
            v = np.roll(np.flip(v, axis=1 + a), 1, axis=1 + a)
    inv = [0, 0, 0]
    for a in range(3):
        inv[perm[a]] = a
    v = np.transpose(v, (0,) + tuple(1 + i for i in inv) + (4,))
    v = np.array(v, copy=True)
    vel = v[..., :3].copy()
    for a in range(3):
        v[..., perm[a]] = sg[a] * vel[..., a]
    return np.ascontiguousarray(v.reshape(10, NCQ, -1))


def rot_u0(u0, ei):
    """Forward element ei on the (NCQ, S) anchor, which lives on the READOUT lattice."""
    perm, sg = _OHG[ei]
    ub = u0.reshape((QB if QGRID else BOX) + (-1,)).copy()
    ub = np.transpose(ub, tuple(perm) + (3,))
    for a in range(3):
        if sg[a] < 0:
            ub = np.roll(np.flip(ub, axis=a), 1, axis=a)
    ub = np.array(ub, copy=True)
    vel = ub[..., :3].copy()
    for a in range(3):
        ub[..., a] = sg[a] * vel[..., perm[a]]
    return ub.reshape(u0.shape)


# ---------------------------------------------------------------- rollout (model's own)
_CACHE = {}


def make_roll(Rc):
    @tf.function(jit_compile=bool(int(os.environ.get("XLA", "1"))), reduce_retracing=True)
    def _roll(cn, xw, fm, gn, cond, op, fid, fbm, u0, roles, cells):
        win, fmask, u_prev = xw, fm, u0
        fb = tf.cast(fbm, tf.float32)[:, None, :]
        outs = tf.TensorArray(tf.float32, size=10)
        for k in tf.range(10):
            Wp, e = model.trunk_W(cn, win, fmask, gn, K, list(roles), cond, op, box_dims=BOX)
            z1 = tf.reduce_mean(tf.cast(Wp, tf.float32), -1)
            zbox = tf.reshape(z1, tf.concat([[Rc], BOX, [model.d_w]], 0))
            zc = (tf.reshape(zbox, [Rc, NC, model.d_w]) if not QGRID
                  else tf.cast(interp_box(zbox, cells, BOX), tf.float32))
            d_c = tf.cast(model.dec(zc, coords=cells, roles=list(roles),
                                    geom=tf.zeros([Rc, NCQ, 8]), cond=e, fam_id=fid), tf.float32)
            u_prev = u_prev * keep + d_c * fb
            u_prev = tf.ensure_shape(u_prev, [None, NCQ, None])
            outs = outs.write(k, u_prev)
            zn = tf.cast(interp_box(zbox, cn, BOX), tf.float32)
            d_n = tf.cast(model.dec(zn, coords=cn, roles=list(roles),
                                    geom=tf.cast(gn, tf.float32), cond=e, fam_id=fid), tf.float32)
            u_n = win[:, -1] * keep + d_n * fb
            win = tf.concat([win[:, 1:], u_n[:, None]], 1)
            fmask = tf.concat([fmask[:, 1:], tf.ones_like(fmask[:, :1])], 1)
        return outs.stack()
    _CACHE[Rc] = _roll
    return _roll


# ---------------------------------------------------------------- main
SPLIT = os.environ.get("SPLIT", "val")
# UIDS selects EXPLICIT trajectories regardless of which split they land in -- needed for the
# reference's canonical tail (90-97), which our own manifest scatters across train and test.
UIDS = os.environ.get("UIDS", "").strip()
if UIDS:
    from data import adapters as _ad                                       # noqa: E402
    from data.loader import _as_dict, _remap_slots                         # noqa: E402
    from data.registry import FAMILIES as _FAMS                            # noqa: E402
    _want = [u.strip() for u in UIDS.split(",") if u.strip()]
    _spec = _FAMS[FAM]
    _pool = dict(_ad.units(_spec))
    _missing = [u for u in _want if u not in _pool]
    assert not _missing, f"uids not found: {_missing}; available e.g. {list(_pool)[:6]}"
    print(f"[uids] {len(_want)} explicit trajectories: {_want}", flush=True)
    it = iter([_remap_slots(_as_dict(_ad.load(_spec, _pool[u])), FAM) for u in _want])
    SPLIT = "uid"
else:
    it = iter(loader.units_for(FAM, SPLIT))
for _ in range(SKIP):
    next(it)
out = {}
cells_np = _cell_coords(QB if QGRID else BOX)[None].astype(np.float32)
for si in range(N_SAMP):
    s = next(it)
    t0 = time.time()
    dims = tuple(int(x) for x in s["dims"])
    fields = np.asarray(s["fields"], np.float32)
    flat = fields.reshape(fields.shape[0], -1, fields.shape[-1])
    # ONE host build, only for the replica-invariant fields (cond, op, fam_id, feedback
    # mask, roles, fmask) and the normalization scale -- none of these vary per replica.
    ex = make_example_ar(s, T_IN, 10, 4, 512, ENC_N, np.random.default_rng(1234 + 977 * (SKIP + si)),
                         box_dims=BOX)
    b0 = stack_batch([ex])
    scale = np.asarray(b0["scale"], np.float32)[0]                     # (1,S)
    _RL = QB if QGRID else BOX
    gi = np.arange(int(np.prod(dims))).reshape(dims)
    gi = gi[::dims[0] // _RL[0], ::dims[1] // _RL[1], ::dims[2] // _RL[2]].ravel()
    u0_id = flat[9][gi] / scale
    win_np = (flat[0:T_IN]).astype(np.float32)                         # (T, N_all, S)

    flat_win = tf.constant(win_np)
    cn, xw = draw_and_transform(flat_win, tf.constant(dims, tf.int32),
                                tf.constant(scale[None], tf.float32),
                                tf.constant([1234 + 977 * (SKIP + si), 7], tf.int32),
                                tf.constant(PERM), tf.constant(SIGN))
    u0 = np.stack([rot_u0(u0_id, e) for e in ELEMS]).astype(np.float32)
    tile = lambda a: np.repeat(np.asarray(a), R, axis=0)               # noqa: E731

    def field(*names):
        """stack_batch key lookup with fallbacks -- a silently missing field would be
        fed as zeros and change the answer, so a wrong guess must raise here."""
        for n in names:
            if n in b0:
                return b0[n]
        raise KeyError(f"none of {names} in batch; keys = {sorted(b0.keys())}")

    fm = tile(field("fmask")).astype(np.float32)
    gn = np.zeros((R, ENC_N, np.asarray(field("geom_node")).shape[-1]), np.float32)
    cond, op = tile(field("cond")), tile(field("op_multihot", "op"))
    fid, fbm = tile(field("fam_id", "fid")), tile(field("fbmask", "fbm", "fb_mask"))
    cells = np.repeat(cells_np, R, 0)

    pr_parts = []
    for c0 in range(0, R, CHUNK_R):
        c1 = min(R, c0 + CHUNK_R)
        Rc = c1 - c0
        fn = _CACHE.get(Rc) or make_roll(Rc)
        pr_parts.append(np.asarray(fn(cn[c0:c1], xw[c0:c1], tf.constant(fm[c0:c1]),
                                      tf.constant(gn[c0:c1]), tf.constant(cond[c0:c1]),
                                      tf.constant(op[c0:c1]), tf.constant(fid[c0:c1]),
                                      tf.constant(fbm[c0:c1]), tf.constant(u0[c0:c1]),
                                      field("roles"), tf.constant(cells[c0:c1]))))
    pr = np.concatenate(pr_parts, axis=1)                              # (10, R, NCQ, S)
    dt = time.time() - t0
    print(f"[timing] sample {SKIP+si}: {dt:.1f}s total | {dt/max(1,R):.2f}s per replica "
          f"| R={R} XLA={os.environ.get('XLA','1')}", flush=True)

    inv = [rot_inv_pred(pr[:, r], ELEMS[r]) for r in range(R)]
    pp = np.mean(inv, 0) * scale[None]
    y_cells = (flat[10:20][:, gi]).astype(np.float32)
    cm = np.asarray(b0["cmask"], np.float32)[0][None, None]
    num = np.sqrt((((pp - y_cells) * cm) ** 2).sum((1, 2)))
    den = np.sqrt(((y_cells * cm) ** 2).sum((1, 2))) + 1e-12
    rel = float((num / den).mean())
    out[f"base_{FAM}_{SPLIT}{SKIP+si}"] = rel
    if PANEL_B:
        # Fig. 1b displays one scalar volume per 3D family: the LAST ACTIVE channel
        # (fig1_dump.py picks `np.nonzero(cmask)[0][-1]`) at the PANEL_LEAD-th predicted
        # step, in physical units.  Writing the ensemble mean here is the only change
        # relative to the single-pass panel.
        _cmv = np.asarray(b0["cmask"], np.float32)[0]
        _ch = int(np.nonzero(_cmv)[0][-1])
        _n = (QB if QGRID else BOX)[0]
        _L = PANEL_LEAD - 1
        out[f"b3d_{FAM}_pred"] = pp[_L, :, _ch].reshape(_n, _n, _n).astype(np.float32)
        out[f"b3d_{FAM}_gt"] = y_cells[_L, :, _ch].reshape(_n, _n, _n).astype(np.float32)
        print(f"[b3d] {FAM}: ch{_ch} lead{PANEL_LEAD} vol {_n}^3 "
              f"rel {np.linalg.norm(out[f'b3d_{FAM}_pred']-out[f'b3d_{FAM}_gt'])/np.linalg.norm(out[f'b3d_{FAM}_gt']):.4f}",
              flush=True)
    if int(os.environ.get("SAVE_PRED", "0")):
        out[f"pred_{FAM}_{SPLIT}{SKIP+si}"] = pp.astype(np.float32)
        out[f"gt_{FAM}_{SPLIT}{SKIP+si}"] = y_cells.astype(np.float32)
    print(f"[{FAM}] s{SKIP+si} base={rel*100:.2f}% (in-graph R{R})", flush=True)

os.makedirs(OUT, exist_ok=True)
np.savez_compressed(os.path.join(OUT, f"infer3d_{TAG}.npz"), **out)
print("INFER3D_DONE", flush=True)
