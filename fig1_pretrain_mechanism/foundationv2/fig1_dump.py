"""Figure-1 panel data dump (MOTION-NCS). Parameterized by checkpoint — rerun with a newer
ckpt to regenerate every panel from the same script.

  python fig1_dump.py /corpus/results/motion_m1/ckpt_best.npz /code-vol/motion_fv2/figs_out

Emits fig1_data.npz with:
  b: GT vs prediction final-frame fields for 5 regimes (smooth/shock/turbulent/wave/3D slice)
  c: knockout error matrix (mechanism x family, PHYS space) + baseline + contribution norms
  d: incom_ns final-frame fields with the transport carrier on/off (+GT) for spectra
  e: vortex-stretching head contribution field on 3D-Turb (mid-z slice) and its 2D counterpart
"""
import json
import os
import sys

import numpy as np
import tensorflow as tf

from core.pretrain import load_ckpt
from data import loader
from data.registry import FAMILIES, NUM_SLOTS, SEMANTIC_SLOTS, MOTION_FAMILIES
from v3.data_ar import make_example_ar, stack_batch
from v3.model import V3Model
from v3.policy import ar_mode
from v3.train import BOX_TABLE_2D, BOX_TABLE_3D, rollout_loss, singleshot_loss

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/corpus/results/motion_m1/ckpt_best.npz"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/code-vol/motion_fv2/figs_out"
# m1b architecture flags (m2: d=896, depth=12, d_w=48 — pass via env when regenerating)
D = int(os.environ.get("FIG_D", 384)); DEPTH = int(os.environ.get("FIG_DEPTH", 8))
DW = int(os.environ.get("FIG_DW", 16)); NP_ = int(os.environ.get("FIG_NP", 16))
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(1234)

FAMS = list(MOTION_FAMILIES)
F3D = [f for f in FAMS if FAMILIES[f].K == 3]
T3 = "pdebench3d_cns_Turb_M1.0_Eta1e-08_Zeta1e-08"
M10 = "pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08"
VOL_N = 64                       # display volume resolution for 3D isosurfaces

def bdims(fam):
    spec = FAMILIES[fam]
    r = (BOX_TABLE_2D if spec.K == 2 else BOX_TABLE_3D).get(fam, 64 if spec.K == 2 else 32)
    return (int(r),) * spec.K

# ---------------------------------------------------------------- model
model = V3Model(S=NUM_SLOTS, d=D, depth=DEPTH, d_w=DW, n_p=NP_, d_cond=128,
                dims2=(64, 64), dims3=(32, 32, 32), t_in=10, transport=True,
                n_fams=len(FAMILIES), n_semantic=SEMANTIC_SLOTS,
                gate_cond=1, dict_decode=0, fam_head=1)
model.warm_build()
V = model.trainable_variables
load_ckpt(V, CKPT, d_w=DW, names=[v.name for v in V])
print(f"[fig1] loaded {CKPT}: {sum(int(np.prod(v.shape)) for v in V)/1e6:.1f}M params", flush=True)

KEYS = ("coords_node", "x_win", "fmask", "geom_node", "y_node", "ic_q", "coords_q",
        "geom_q", "y_q", "tstep_mask", "cmask", "fbmask", "op_multihot", "cond",
        "fam_id", "sdf_box")

D4G = os.environ.get("D4_G", "")     # "", "fx", "fy", "r180" — dense-family symmetry TTA hook
def _d4(su, g=None):
    g = D4G if g is None else g
    if not g or g == "id":
        return su
    f = np.array(su["fields"], np.float32, copy=True)     # (T, X, Y, S)
    if g in ("fx", "r180"):
        f = f[:, ::-1]; f[..., 0] *= -1.0                  # mirror x: vx sign
    if g in ("fy", "r180"):
        f = f[:, :, ::-1]; f[..., 1] *= -1.0               # mirror y: vy sign
    return {**{k: su[k] for k in su.keys()}, "fields": np.ascontiguousarray(f)}

def batches_for(fam, n, k_fut=10):
    out = []
    for s in loader.units_for(fam, "val"):
        s = _d4(s)
        ex = make_example_ar(s, 10, k_fut, 4, 2048, 8192, rng, box_dims=bdims(fam))
        b = stack_batch([ex])
        if int(b["K"]) == 3:
            # 3D windows/targets are node-subsampled; keep mid-z plane slices of the RAW
            # trajectory for display (anchor: all channels normalized; GT: display channel).
            f = np.asarray(s["fields"], np.float32)
            dims = tuple(int(x) for x in s["dims"])
            zi = dims[2] // 2
            ti = min(10, f.shape[0] - 1)
            sc = b["scale"][0]
            b["_anchor_plane"] = (f[ti - 1, :, :, zi, :] / sc[None, None, :]).astype(np.float32)
            b["_gt_planes"] = f[ti:ti + k_fut, :, :, zi, :].astype(np.float32)   # physical units
            st = dims[0] // VOL_N
            b["_anchor_vol"] = (f[ti - 1, ::st, ::st, ::st, :] / sc[None, None, None, :]).astype(np.float32)
            b["_gt_vols"] = f[ti:ti + k_fut, ::st, ::st, ::st, :].astype(np.float32)
        out.append(b)
        if len(out) >= n:
            break
    return out

def tensors(b):
    return tuple(tf.constant(b[k]) for k in KEYS)

def statics(b):
    gd = tuple(b.get("dims", ())) or None
    return (int(b["K"]), bool(b["steady"]), bool(b["dense2d"]),
            tuple(int(r) for r in b["roles"]), int(b["k_seg"]), gd,
            tuple(int(v) for v in b["box_dims"]), bool(b["surf"]))

@tf.function(reduce_retracing=True)
def loss_ar(ts, sc, K, steady, dense2d, roles, k_seg, grid_dims, box_dims, surf):
    return rollout_loss(model, *ts, K, steady, dense2d, roles, k_seg, grid_dims,
                        box_dims, surf, clamp=1e9, scale=sc)

@tf.function(reduce_retracing=True)
def loss_ss(ts, sc, K, steady, dense2d, roles, k_seg, grid_dims, box_dims, surf):
    return singleshot_loss(model, *ts, K, roles, k_seg, box_dims, clamp=1e9, scale=sc)

def fam_err(bs):
    """Mean PHYS rel-L2 over pre-built batches (index [1] of the [norm, phys] pair)."""
    vals = []
    for b in bs:
        fn = loss_ss if b["mode"] == "ss" else loss_ar
        vals.append(float(fn(tensors(b), tf.constant(b["scale"]), *statics(b))[1]))
    return float(np.mean(vals))

# ---------------------------------------------------------------- panel b + d + e rollout helper
def predict_fields(b, transport_off=False, ret_all=False):
    """Eager rollout; returns (gt_field, pred_field) at the final valid frame on the node grid
    (2D: full dims; 3D: decoded on a mid-z plane 128x128)."""
    K = int(b["K"]); dims = tuple(b["dims"]); mode = b["mode"]
    cn = tf.constant(b["coords_node"]); xw = tf.constant(b["x_win"]); fm = tf.constant(b["fmask"])
    gn = tf.constant(b["geom_node"]); cond = tf.constant(b["cond"]); op = tf.constant(b["op_multihot"])
    fid = tf.constant(b["fam_id"]); sdb = tf.constant(b["sdf_box"]); bx = tuple(b["box_dims"])
    tv = b["tstep_mask"][0]; last = int(np.max(np.nonzero(tv)[0])) if tv.max() > 0 else 0
    # FIG_LEAD selects which predicted step the panel shows (1-based; 0/unset = final frame).
    # The panel is a qualitative display, so the lead it is drawn at must be stated in the
    # caption -- a one-step panel shows the operator, not the rollout.
    _L = int(os.environ.get("FIG_LEAD", 0))
    if _L > 0:
        last = min(_L - 1, last)
    vol = b.get("_want_vol", False)
    if K == 2:
        q = cn; gq = gn
    elif vol:  # decode the full display volume for 3D isosurfaces
        n = VOL_N
        ax = (np.arange(n) + 0.5) / n
        g3 = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
        q = tf.constant(g3[None].astype(np.float32))
        gq = tf.zeros([1, n ** 3, 8])
    else:  # decode on a mid-z plane at native xy resolution
        n = 128
        ax = (np.arange(n) + 0.5) / n
        gxy = np.stack(np.meshgrid(ax, ax, indexing="ij"), -1).reshape(-1, 2)
        q = tf.constant(np.concatenate([gxy, np.full((n * n, 1), 0.5)], -1)[None].astype(np.float32))
        gq = tf.zeros([1, n * n, 8])
    u_prev = tf.constant(b["x_win"][:, -1]) if K == 2 else None
    if K == 3:
        # anchor values at the plane: interpolate GT anchor frame (nearest z slice)
        pass
    if mode == "ss":
        icq = tf.constant(b["x_win"][:, -1]) if K == 2 else None
        preds = model.step_ss(cn, xw, fm, gn, K, list(b["roles"]), cond, op, q, gq,
                              interp_anchor(b, q, K), kf=int(b["k_seg"]), fam_id=fid,
                              sdf_box=sdb, box_dims=bx)
        pred = preds[:, last].numpy()
    else:
        # unified free-running AR: queries = nodes (feedback) ++ display points; the window is
        # refreshed from the node predictions exactly as in training (v3/train rollout_loss).
        u_prev_q = interp_anchor(b, q, K)
        win, fmask = xw, fm
        prev_grid = tf.reshape(xw[:, -1], [1, *dims, NUM_SLOTS]) if (K == 2 and b["dense2d"]) else None
        if transport_off:
            model.disp.kernel.assign(tf.zeros_like(model.disp.kernel))
            model.disp.bias.assign(tf.zeros_like(model.disp.bias))
            model.tgate.kernel.assign(tf.zeros_like(model.tgate.kernel))
        # ret_all (K=2): the display grid IS the node grid, so skip the duplicated display
        # queries entirely -- halves the per-step query count (peak-memory relief for the
        # 128^2-native turbulence boxes) and reads the node predictions directly.
        skipq = ret_all and K == 2
        qall = cn if skipq else tf.concat([cn, q], 1)
        gall = gn if skipq else tf.concat([gn, gq], 1)
        up = xw[:, -1] if skipq else tf.concat([xw[:, -1], u_prev_q], 1)
        Nn = cn.shape[1]
        for k in range(last + 1):
            pred_all = model.step(cn, win, fmask, gn, K, list(b["roles"]), cond, op, qall, gall,
                                  up, steady=False, prev_grid=prev_grid,
                                  grid_dims=dims if b["dense2d"] else None, fam_id=fid,
                                  sdf_box=sdb, box_dims=bx,
                                  fbmask=tf.constant(b["fbmask"]))
            up = pred_all
            pn = pred_all[:, :Nn]
            win = tf.concat([win[:, 1:], pn[:, None]], 1)
            fmask = tf.concat([fmask[:, 1:], tf.ones_like(fmask[:, :1])], 1)
            if K == 2 and b["dense2d"]:
                prev_grid = tf.reshape(pn, [1, *dims, NUM_SLOTS])
        pred = (pred_all if skipq else pred_all[:, Nn:]).numpy()
    cm = b["cmask"][0].astype(bool)
    if ret_all and K == 2:                   # every active channel, physical units (for spectra)
        chs = np.nonzero(cm)[0]
        scv = b["scale"][0][chs].astype(np.float32)
        gt = b["y_node"][0, last][:, chs].reshape(*dims, -1) * scv
        pf = pred[0][:, chs].reshape(*dims, -1) * scv
        return gt.astype(np.float32), pf.astype(np.float32)
    ch = int(np.nonzero(cm)[0][-1])          # display channel: last active (scalar/pressure-like)
    sc = b["scale"][0][ch]
    if K == 2:
        gt = b["y_node"][0, last][:, ch].reshape(dims) * sc
        pf = pred[0][:, ch].reshape(dims) * sc
    elif vol:
        n = VOL_N
        gt = b["_gt_vols"][last][:, :, :, ch]         # physical units
        pf = pred[0][:, ch].reshape(n, n, n) * sc
    else:
        n = 128
        gt = gt_plane(b, last, ch)                    # already in physical units
        pf = pred[0][:, ch].reshape(n, n) * sc
    return gt.astype(np.float32), pf.astype(np.float32)

def interp_anchor(b, q, K):
    """Anchor frame values at query coords."""
    if K == 3 and b.get("_want_vol", False):
        return tf.constant(b["_anchor_vol"].reshape(1, -1, b["_anchor_vol"].shape[-1]))
    if K == 3:
        n = 128
        a = b["_anchor_plane"]                       # (X,Y,S) normalized, mid-z
        ii = np.minimum(((np.arange(n) + 0.5) / n * a.shape[0]).astype(int), a.shape[0] - 1)
        jj = np.minimum(((np.arange(n) + 0.5) / n * a.shape[1]).astype(int), a.shape[1] - 1)
        return tf.constant(a[np.ix_(ii, jj)].reshape(1, n * n, -1))
    dims = tuple(b["dims"])
    a = b["x_win"][0, -1].reshape(*dims, NUM_SLOTS)
    qn = q[0].numpy()
    idx = tuple(np.minimum((qn[:, k] * dims[k]).astype(int), dims[k] - 1) for k in range(K))
    return tf.constant(a[idx][None].astype(np.float32))

def gt_plane(b, t, ch):
    g = b["_gt_planes"][t][:, :, ch]                 # physical units already
    n = 128
    ii = np.minimum(((np.arange(n) + 0.5) / n * g.shape[0]).astype(int), g.shape[0] - 1)
    jj = np.minimum(((np.arange(n) + 0.5) / n * g.shape[1]).astype(int), g.shape[1] - 1)
    return g[np.ix_(ii, jj)]

out = {}
# ---------------------------------------------------------------- panel d (v2): head-knockout spectra
# ONLY_KO=1: full-model rollout vs single-head knockouts (bank gate + cond-gate zeroed --
# exactly the panel-c intervention) on turbulence families. All active channels are dumped
# in physical units so the renderer can form energy spectra / phase coherence on any
# channel. A superset of heads is dumped; the renderer chooses which to show. Writes
# fig1d_ko.npz next to fig1_data.npz and exits (no other panel is touched).
if os.environ.get("ONLY_KO"):
    import gc
    GN = list(model.banks._gnames)

    # Graph-mode free-running rollout on the NODE grid (queries = nodes). The eager
    # predict_fields loop OOMs on the 128^2-native turbulence boxes; the panel-c loss path
    # proves the same rollout fits in graph mode. Gate knockouts mutate variables between
    # calls, which a traced graph picks up without retracing.
    @tf.function(reduce_retracing=True)
    def roll_nodes(cn, xw, fm, gn, cond, op, fid, sdb, fbm,
                   K, roles, dims, dense2d, bx, nsteps):
        win, fmask = xw, fm
        prev_grid = tf.reshape(xw[:, -1], [1, *dims, NUM_SLOTS]) if dense2d else None
        up = xw[:, -1]
        pred = up
        for _ in range(nsteps):
            pred = model.step(cn, win, fmask, gn, K, list(roles), cond, op, cn, gn, up,
                              steady=False, prev_grid=prev_grid,
                              grid_dims=dims if dense2d else None, fam_id=fid,
                              sdf_box=sdb, box_dims=bx, fbmask=fbm)
            up = pred
            win = tf.concat([win[:, 1:], pred[:, None]], 1)
            fmask = tf.concat([fmask[:, 1:], tf.ones_like(fmask[:, :1])], 1)
            if dense2d:
                prev_grid = tf.reshape(pred, [1, *dims, NUM_SLOTS])
        return pred

    def ko_fields(b):
        dims = tuple(b["dims"]); K = int(b["K"])
        tv = b["tstep_mask"][0]; last = int(np.max(np.nonzero(tv)[0])) if tv.max() > 0 else 0
        _L = int(os.environ.get("FIG_LEAD", 0))
        if _L > 0:
            last = min(_L - 1, last)
        pred = roll_nodes(tf.constant(b["coords_node"]), tf.constant(b["x_win"]),
                          tf.constant(b["fmask"]), tf.constant(b["geom_node"]),
                          tf.constant(b["cond"]), tf.constant(b["op_multihot"]),
                          tf.constant(b["fam_id"]), tf.constant(b["sdf_box"]),
                          tf.constant(b["fbmask"]), K,
                          tuple(int(r) for r in b["roles"]), dims,
                          bool(b["dense2d"]), tuple(int(v) for v in b["box_dims"]),
                          last + 1).numpy()
        cm = b["cmask"][0].astype(bool)
        chs = np.nonzero(cm)[0]
        scv = b["scale"][0][chs].astype(np.float32)
        yn = b["y_node"][0, last][:, chs]
        pn = pred[0][:, chs]
        if bool(b["dense2d"]):                       # dense grid -> (X,Y,C) for spectra/contours
            yn = yn.reshape(*dims, -1); pn = pn.reshape(*dims, -1)
        return (yn * scv).astype(np.float32), (pn * scv).astype(np.float32), chs

    def _inv2d(f, g, chs):
        """Inverse symmetry transform of a dense (X,Y,C-active) field + velocity signs."""
        if not g or g == "id":
            return f
        f = np.array(f, copy=True)
        chs = list(chs)
        if g in ("fx", "r180"):
            f = f[::-1]
            if 0 in chs: f[..., chs.index(0)] *= -1.0
        if g in ("fy", "r180"):
            f = f[:, ::-1]
            if 1 in chs: f[..., chs.index(1)] *= -1.0
        return f
    heads = [h for h in os.environ.get(
        "KO_HEADS",
        "spectral,shock,diffusion,gradvec,buoyancy,advection,compressible,wave").split(",")
        if h in GN]
    fams = os.environ.get("KO_FAMS", "pdearena_ns,incom_ns").split(",")
    nko = int(os.environ.get("KO_N", 4))
    ko = {}
    SYMS = [g for g in os.environ.get("KO_SYM", "").split(",") if g]
    for fam in fams:
        if SYMS:
            # OFFICIAL dense-family symmetry ensemble (08-11): roll each legal transform of
            # the SAME unit through the full model.step graph path, inverse-transform the
            # dense predictions, average. Same npz schema -> the d/e renderers run unchanged.
            units = []
            for s_u in loader.units_for(fam, "val"):
                units.append(s_u)
                if len(units) >= nko:
                    break
            acc = {"gt": [], "full": []}
            acc.update({m: [] for m in heads})
            for s_u in units:
                per = {c: [] for c in ["full"] + heads}
                gt0 = None
                seed_u = int(rng.integers(2 ** 31))       # SAME window/draws for every g --
                for g in SYMS:                            # averaging different windows is garbage
                    rng_g = np.random.default_rng(seed_u)
                    ex = make_example_ar(_d4(s_u, g), 10, 10, 4, 2048, 8192, rng_g,
                                         box_dims=bdims(fam))
                    b = stack_batch([ex])
                    assert bool(b["dense2d"]), f"KO_SYM requires a dense family, got {fam}"
                    gt, pf, chs = ko_fields(b)
                    if g == "id":
                        gt0 = gt
                    per["full"].append(_inv2d(pf, g, chs))
                    if int(os.environ.get("KO_SYM_DIAG", "0")) and gt0 is not None:
                        _n0 = np.linalg.norm(gt0) + 1e-12
                        cands = " ".join(
                            f"{gc}:{100 * np.linalg.norm(_inv2d(pf, gc, chs) - gt0) / _n0:.2f}"
                            for gc in ("id", "fx", "fy", "r180"))
                        print(f"[ko-sym-diag] {fam} g={g} inv-cands({cands})%  "
                              f"gt-drift={100 * np.linalg.norm(gt - _inv2d(gt0, g, chs) if g != 'id' else gt - gt0) / _n0:.2f}%",
                              flush=True)
                    for m in heads:
                        g0 = model.banks.gates[m].numpy()
                        model.banks.gates[m].assign(tf.zeros_like(model.banks.gates[m]))
                        gc0 = None
                        if model.banks.gate_cond:
                            gc0 = model.banks.gcond[m].kernel.numpy()
                            model.banks.gcond[m].kernel.assign(
                                tf.zeros_like(model.banks.gcond[m].kernel))
                        _, pk, _ = ko_fields(b)
                        model.banks.gates[m].assign(g0)
                        if gc0 is not None:
                            model.banks.gcond[m].kernel.assign(gc0)
                        per[m].append(_inv2d(pk, g, chs))
                    gc.collect()
                assert gt0 is not None, "KO_SYM list must include 'id'"
                acc["gt"].append(gt0)
                for c in ["full"] + heads:
                    acc[c].append(np.mean(per[c], 0))
            g_arr = np.stack(acc["gt"])
            ko[f"{fam}__gt"] = g_arr
            for kname in ["full"] + heads:
                p_arr = np.stack(acc[kname])
                ko[f"{fam}__{kname}"] = p_arr
                rel = np.linalg.norm(p_arr - g_arr) / (np.linalg.norm(g_arr) + 1e-12)
                print(f"[ko-sym{len(SYMS)}] {fam} {kname}: rel-L2 {100*rel:.2f}%", flush=True)
            continue
        bs = batches_for(fam, nko)
        acc = {"gt": [], "full": []}
        acc.update({m: [] for m in heads})
        for b in bs:
            gt, pf, _ = ko_fields(b)
            gc.collect()
            acc["gt"].append(gt); acc["full"].append(pf)
            for m in heads:
                g0 = model.banks.gates[m].numpy()
                model.banks.gates[m].assign(tf.zeros_like(model.banks.gates[m]))
                gc0 = None
                if model.banks.gate_cond:
                    gc0 = model.banks.gcond[m].kernel.numpy()
                    model.banks.gcond[m].kernel.assign(tf.zeros_like(model.banks.gcond[m].kernel))
                _, pk, _ = ko_fields(b)
                model.banks.gates[m].assign(g0)
                if gc0 is not None:
                    model.banks.gcond[m].kernel.assign(gc0)
                acc[m].append(pk)
                gc.collect()
        g = np.stack(acc["gt"])
        ko[f"{fam}__gt"] = g
        for kname in ["full"] + heads:
            p = np.stack(acc[kname])
            ko[f"{fam}__{kname}"] = p
            rel = np.linalg.norm(p - g) / (np.linalg.norm(g) + 1e-12)
            print(f"[ko] {fam} {kname}: rel-L2 {100*rel:.2f}%", flush=True)
    oname = os.environ.get("KO_OUT", "fig1d_ko")
    np.savez_compressed(os.path.join(OUT, oname + ".npz"), **ko)
    json.dump({"ckpt": CKPT, "heads": heads, "families": fams,
               "lead": int(os.environ.get("FIG_LEAD", 0)), "n": nko},
              open(os.path.join(OUT, oname + "_meta.json"), "w"))
    print(f"[ko] saved -> {OUT}/{oname}.npz", flush=True)
    sys.exit(0)

# ---------------------------------------------------------------- panel b
# panel b/d family: incompressible NS replaces buoyant NS. Two reasons: it is the
# stronger family (3.0% vs 19.5% at m1t) and it removes the bylfa substitution the
# renderer applied to the buoyant strip, so every panel is one model, one weight set.
PBFAMS = ["ACE", "CE-RM", "incom_ns", "Wave-Layer"]
PB3D = [M10, T3]
for fam in PB3D:                       # 3D isosurface volumes (GT vs pred at VOL_N^3)
    b = batches_for(fam, 1)[0]
    b["_want_vol"] = True
    gt, pf = predict_fields(b)
    out[f"b3d_{fam}_gt"], out[f"b3d_{fam}_pred"] = gt, pf
    print(f"[b3d] {fam}: vol {gt.shape}, rel {np.linalg.norm(pf-gt)/np.linalg.norm(gt):.3f}", flush=True)
if os.environ.get("ONLY_B3D"):
    old = dict(np.load(os.path.join(OUT, "fig1_data.npz")))
    old.update(out)
    np.savez_compressed(os.path.join(OUT, "fig1_data.npz"), **old)
    json.dump({"ckpt": CKPT, "families": FAMS, "mechanisms": GN, "pb_fams": PBFAMS, "pb3d": PB3D},
              open(os.path.join(OUT, "fig1_meta.json"), "w"))
    print("[fig1] ONLY_B3D merge done", flush=True)
    sys.exit(0)
for fam in PBFAMS:
    b = batches_for(fam, 1)[0]
    gt, pf = predict_fields(b)
    out[f"b_{fam}_gt"], out[f"b_{fam}_pred"] = gt, pf
    print(f"[b] {fam}: field {gt.shape}, rel {np.linalg.norm(pf-gt)/np.linalg.norm(gt):.3f}", flush=True)

# ---------------------------------------------------------------- panel d (transport on/off)
bd = batches_for("incom_ns", 3)
d_gt, d_on, d_off = [], [], []
for b in bd:
    gt, on = predict_fields(b, transport_off=False)
    saved = [model.disp.kernel.numpy(), model.disp.bias.numpy(), model.tgate.kernel.numpy()]
    _, off = predict_fields(b, transport_off=True)
    model.disp.kernel.assign(saved[0]); model.disp.bias.assign(saved[1]); model.tgate.kernel.assign(saved[2])
    d_gt.append(gt); d_on.append(on); d_off.append(off)
out["d_gt"], out["d_on"], out["d_off"] = np.stack(d_gt), np.stack(d_on), np.stack(d_off)
print("[d] transport on/off fields dumped", flush=True)

# ---------------------------------------------------------------- panel c: knockout + contributions
GN = list(model.banks._gnames)
cache = {f: batches_for(f, 2) for f in FAMS}          # 2 samples/family for the draft figure
base = {f: fam_err(cache[f]) for f in FAMS}
print("[c] baseline done", flush=True)
KO = np.zeros((len(GN), len(FAMS)), np.float32)
for i, m in enumerate(GN):
    g0 = model.banks.gates[m].numpy()
    model.banks.gates[m].assign(tf.zeros_like(model.banks.gates[m]))
    gc0 = None
    if model.banks.gate_cond:
        gc0 = model.banks.gcond[m].kernel.numpy()
        model.banks.gcond[m].kernel.assign(tf.zeros_like(model.banks.gcond[m].kernel))
    for j, f in enumerate(FAMS):
        KO[i, j] = fam_err(cache[f]) - base[f]
    model.banks.gates[m].assign(g0)
    if gc0 is not None:
        model.banks.gcond[m].kernel.assign(gc0)
    print(f"[c] knockout {m}: max dErr {KO[i].max():.4f}", flush=True)
out["c_knockout"] = KO
out["c_base"] = np.array([base[f] for f in FAMS], np.float32)

# contribution norms: one eager step per family with the probe on
CN = np.zeros((len(GN), len(FAMS)), np.float32)
for j, f in enumerate(FAMS):
    b = cache[f][0]
    model.banks.probe = {}
    model.banks.probe_field = "vortex" if f in (T3, "pdearena_ns") else None
    try:
        predict_fields(b)
    except Exception as e:
        print(f"[c] probe {f} fell back ({e})", flush=True)
    pr = model.banks.probe or {}
    for i, m in enumerate(GN):
        CN[i, j] = pr.get(m, 0.0)
    if f == T3 and "vortex_field" in pr:
        vf = pr["vortex_field"]                     # (B,*dims,out)
        out["e_vortex3d"] = np.abs(vf[0]).mean(-1)[:, :, vf.shape[3] // 2].astype(np.float32)
    if f == "pdearena_ns" and "vortex_field" in pr:
        out["e_vortex2d"] = np.abs(pr["vortex_field"][0]).mean(-1).astype(np.float32)
    model.banks.probe = None
    model.banks.probe_field = None
out["c_contrib"] = CN
print("[c/e] contributions + vortex fields dumped", flush=True)

np.savez_compressed(os.path.join(OUT, "fig1_data.npz"), **out)
json.dump({"ckpt": CKPT, "families": FAMS, "mechanisms": GN, "pb_fams": PBFAMS, "pb3d": PB3D},
          open(os.path.join(OUT, "fig1_meta.json"), "w"))
print(f"[fig1] saved -> {OUT}/fig1_data.npz", flush=True)
