"""OFFICIAL W-ensemble inference (MOTION-NCS convention, 2026-08-11).

Two orthogonal averaging axes, applied per family according to what its data path
actually randomizes / its physics actually permits:

  SUBSAMPLE axis (non-dense families: 3D volumes, PoolBoil-class downstream). The
  encoder sees an enc_n point subsample, so K draws give K valid instances W on the
  SHARED box lattice -> averaged in W-space (the instance stays a solution).
  Dense 2D families encode the full grid: this axis is degenerate there.

  SYMMETRY axis (all families, group limited by physics). Latents are NOT
  transform-covariant, so this axis averages in FIELD space: transform the input
  unit, roll out, inverse-transform the prediction, average over the legal group.
  Legal group is derived from the registry spec: an axis is flippable iff its two
  boundaries are of the SAME type (periodic, or wall|wall), and the gravity axis
  (any family whose operator set contains "buoyancy"; role axis 1) is never
  flipped. Rotations are deferred (role re-wiring); flips compose freely.

Cost: k_sub encoder passes are one batched trunk call; the symmetry loop multiplies
rollouts by |G| (trajectory-level; measured pilots: 3D product -24%, dense-2D
symmetry-only -10..-24%).

Usage:
    we = WEns(model)
    gt, pred, chs = we.predict(spec, unit, rng, k_sub=8, syms=None)  # syms=None -> legal_syms
"""
import numpy as np
import tensorflow as tf

from data.registry import FAMILIES, NUM_SLOTS


def legal_syms(spec):
    """Legal flip subgroup for a family, from its bc tuple / operators / K."""
    K = spec.K
    bc = tuple(getattr(spec, "bc", ()) or ())
    ops = tuple(getattr(spec, "operators", ()) or ())
    grav_axis = 1 if "buoyancy" in ops else None          # role-1 = gravity by convention
    def axis_ok(a):
        if a == grav_axis:
            return False
        if not bc:                                        # unspecified -> periodic corpus default
            return True
        # bc is per-axis-pair when len==K (e.g. PoolBoil ("wall","open","periodic")):
        # an axis is flippable iff its boundary spec is symmetric under the flip.
        if len(bc) == K:
            return bc[a] in ("periodic", "wall")
        return all(b == "periodic" for b in bc)
    axes = [a for a in range(K) if axis_ok(a)]
    names = {0: "fx", 1: "fy", 2: "fz"}
    syms = ["id"]
    # all non-empty subsets of flippable axes, composed
    for mask in range(1, 1 << len(axes)):
        syms.append("".join(names[axes[i]] for i in range(len(axes)) if mask >> i & 1))
    return syms


def transform_unit(su, g, K):
    """Apply flip g to a unit sample dict (fields (T, *spatial, S); vx/vy/vz = slots 0/1/2)."""
    if not g or g == "id":
        return su
    f = np.array(su["fields"], np.float32, copy=True)
    for a, tok in enumerate(("fx", "fy", "fz")[:K]):
        if tok in g:
            f = np.flip(f, axis=1 + a)
            f[..., a] *= -1.0
    return {**{k: su[k] for k in su.keys()}, "fields": np.ascontiguousarray(f)}


def inverse_field(arr, g, slot_ids, K, spatial_axes):
    """Inverse-transform a prediction whose channel dim holds the compact slots
    `slot_ids`; `spatial_axes` are the array axes carrying x(,y(,z))."""
    if not g or g == "id":
        return arr
    arr = np.array(arr, copy=True)
    sl = list(slot_ids)
    for a, tok in enumerate(("fx", "fy", "fz")[:K]):
        if tok in g:
            arr = np.flip(arr, axis=spatial_axes[a])
            if a in sl:
                arr[..., sl.index(a)] *= -1.0
    return np.ascontiguousarray(arr)


class WEns:
    """Compiled ensemble-rollout paths bound to one model instance."""

    def __init__(self, model):
        self.model = model

        @tf.function(reduce_retracing=True)
        def _roll_dense(cn, xw, fm, gn, cond, op, fid, sdb, fbm,
                        K, roles, dims, bx, nsteps):
            win, fmask = xw, fm
            prev_grid = tf.reshape(xw[:, -1], [1, *dims, NUM_SLOTS])
            up = xw[:, -1]
            pred = up
            for _ in range(nsteps):
                pred = model.step(cn, win, fmask, gn, K, list(roles), cond, op, cn, gn,
                                  up, steady=False, prev_grid=prev_grid, grid_dims=dims,
                                  fam_id=fid, sdf_box=sdb, box_dims=bx, fbmask=fbm)
                up = pred
                win = tf.concat([win[:, 1:], pred[:, None]], 1)
                fmask = tf.concat([fmask[:, 1:], tf.ones_like(fmask[:, :1])], 1)
                prev_grid = tf.reshape(pred, [1, *dims, NUM_SLOTS])
            return pred

        @tf.function(reduce_retracing=True)
        def _roll_sub(cn, xw, fm, gn, cond, op, fid, fbm, u0,
                      K, roles, bx, ncell, cells, nsteps):
            """Stacked subsample replicas (k_sub, ...); W-mean each step; cell rollout +
            per-replica node feedback (one batched dec)."""
            from v3.model import interp_box
            keep = tf.concat([tf.ones([model.n_semantic]),
                              tf.zeros([model.S - model.n_semantic])], 0)[None, None]
            fb = tf.cast(fbm, tf.float32)[:, None, :]
            win, fmask = xw, fm
            u_prev = u0
            for _ in range(nsteps):
                Wp, e = model.trunk_W(cn, win, fmask, gn, K, list(roles), cond, op,
                                      box_dims=bx)
                zm = tf.reduce_mean(tf.reduce_mean(tf.cast(Wp, tf.float32), -1),
                                    0, keepdims=True)
                zc = tf.reshape(zm, [1, ncell, model.d_w])
                d_c = tf.cast(model.dec(zc, coords=cells, roles=list(roles),
                                        geom=tf.zeros([1, ncell, 8]), cond=e[:1],
                                        fam_id=fid[:1]), tf.float32)
                u_prev = u_prev * keep + d_c * fb[:1]
                zbox = tf.reshape(zm, (1,) + tuple(bx) + (model.d_w,))
                zboxK = tf.tile(zbox, [tf.shape(cn)[0]] + [1] * (len(bx) + 1))
                zn = tf.cast(interp_box(zboxK, cn, bx), tf.float32)
                d_n = tf.cast(model.dec(zn, coords=cn, roles=list(roles),
                                        geom=tf.cast(gn, tf.float32), cond=e,
                                        fam_id=fid), tf.float32)
                u_n = win[:, -1] * keep + d_n * fb
                win = tf.concat([win[:, 1:], u_n[:, None]], 1)
                fmask = tf.concat([fmask[:, 1:], tf.ones_like(fmask[:, :1])], 1)
            return u_prev
        self._roll_dense = _roll_dense
        self._roll_sub = _roll_sub

    def predict(self, spec, unit, rng, k_sub=8, syms=None, t_in=10, k_fut=10,
                lead=None, box_dims=None):
        """Official-convention prediction for one unit (AR families).

        Returns (gt, pred, slot_ids): dense fams -> (X,Y,C) grids at the final lead;
        non-dense fams -> (10, NC_box, C) cell-rollout arrays (physical units).
        syms=None derives the legal flip group from the spec; k_sub is ignored for
        dense fams (their encoder sees the full grid)."""
        from v3.data_ar import _cell_coords, make_example_ar, stack_batch
        fam = spec.name
        K = spec.K
        syms = legal_syms(spec) if syms is None else syms
        ex0 = make_example_ar(unit, t_in, k_fut, 4, 2048, 8192, rng, box_dims=box_dims)
        b0 = stack_batch([ex0])
        dense = bool(b0["dense2d"])
        chs = np.nonzero(b0["cmask"][0])[0]
        preds, gt0 = [], None
        if dense:
            dims = tuple(b0["dims"])
            tv = b0["tstep_mask"][0]
            last = int(np.max(np.nonzero(tv)[0])) if tv.max() > 0 else 0
            if lead:
                last = min(lead - 1, last)
            for g in syms:
                ex = ex0 if g == "id" else make_example_ar(
                    transform_unit(unit, g, K), t_in, k_fut, 4, 2048, 8192, rng,
                    box_dims=box_dims)
                b = stack_batch([ex])
                pred = self._roll_dense(
                    tf.constant(b["coords_node"]), tf.constant(b["x_win"]),
                    tf.constant(b["fmask"]), tf.constant(b["geom_node"]),
                    tf.constant(b["cond"]), tf.constant(b["op_multihot"]),
                    tf.constant(b["fam_id"]), tf.constant(b["sdf_box"]),
                    tf.constant(b["fbmask"]), K,
                    tuple(int(r) for r in b["roles"]), dims,
                    tuple(int(v) for v in b["box_dims"]), last + 1).numpy()
                scv = b["scale"][0][chs].astype(np.float32)
                pf = (pred[0][:, chs].reshape(*dims, -1) * scv)
                if g == "id":
                    gt0 = (b["y_node"][0, last][:, chs].reshape(*dims, -1) * scv)
                preds.append(inverse_field(pf, g, list(chs), K,
                                           spatial_axes=tuple(range(K))))
            return gt0.astype(np.float32), np.mean(preds, 0).astype(np.float32), chs
        # non-dense: subsample stack x symmetry loop, scored on the box-cell lattice
        bx = tuple(int(v) for v in b0["box_dims"])
        ncell = int(np.prod(bx))
        cells = tf.constant(_cell_coords(bx)[None].astype(np.float32))
        fields = np.asarray(unit["fields"], np.float32)
        flat = fields.reshape(fields.shape[0], -1, fields.shape[-1])
        nat = int(round(flat.shape[1] ** (1 / K)))
        st = max(1, nat // bx[0])
        ax = np.arange(0, nat, st)[:bx[0]]
        if K == 3:
            gi = (ax[:, None, None] * nat * nat + ax[None, :, None] * nat
                  + ax[None, None, :]).ravel()
        else:
            gi = (ax[:, None] * nat + ax[None, :]).ravel()
        sc = np.asarray(b0["scale"], np.float32)[0]
        y_cells = flat[t_in:t_in + k_fut][:, gi]
        for g in syms:
            ug = unit if g == "id" else transform_unit(unit, g, K)
            fg = np.asarray(ug["fields"], np.float32).reshape(flat.shape)
            exs = [make_example_ar(ug, t_in, k_fut, 4, 512, 8192, rng, box_dims=bx)
                   for _ in range(k_sub)]
            bs = [stack_batch([e]) for e in exs]
            cat = lambda k: tf.constant(np.concatenate([b[k] for b in bs], 0))
            u0 = tf.constant(fg[t_in - 1][gi][None] / sc[None, None], tf.float32)
            pr = self._roll_sub(cat("coords_node"), cat("x_win"), cat("fmask"),
                                cat("geom_node"), cat("cond"), cat("op_multihot"),
                                cat("fam_id"), tf.constant(bs[0]["fbmask"]), u0,
                                K, tuple(int(r) for r in bs[0]["roles"]), bx, ncell,
                                cells, k_fut).numpy()
            # NOTE: _roll_sub returns only the FINAL state; per-frame trajectories can be
            # added when a caller needs them. Inverse-flip on the box lattice:
            pf = pr[0].reshape(bx + (-1,))
            pf = inverse_field(pf, g, [], K, spatial_axes=tuple(range(K)))
            for a, tok in enumerate(("fx", "fy", "fz")[:K]):
                if tok in (g or ""):
                    pf[..., a] *= -1.0          # velocity slots on RAW slot layout
            preds.append(pf.reshape(ncell, -1))
        gt_last = y_cells[-1]
        pred = np.mean(preds, 0) * sc[None]
        return gt_last.astype(np.float32), pred.astype(np.float32), chs
