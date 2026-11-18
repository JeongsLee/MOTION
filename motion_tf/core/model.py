"""UniversalTPADA prototype: one weight set, 2D+3D, grid or points in, continuous out.

    input (grid frames | point cloud) --[PointEncoder | frames-as-channels]--> latent box
      --> LatentBoxCore (axis-factorized, K-agnostic weights)
      --> TendencyHead: W (B, *dims, D_w, N_p), zero-init
      --> KAxisTPADA synthesis (fixed tables per K; weights above are shared)
      --> ContinuousDecoder at (query points, query times); u = IC(y) + decode(z(y,t))

Hard IC is structural: the time map has G(0,·)=0 and the decoder is bias-free.
Per-K objects (synthesis tables, box node lattice) are constants; every trainable
variable is used by both K=2 and K=3 paths (asserted in tests).
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from .axops import PointwiseMLP  # noqa: F401  (re-export convenience)
from .banks import SteadyField, TendencyBanks
from .binenc import BinEncoder
from .latentbox import LatentBoxCore
from .pointio import ContinuousDecoder, PointEncoder
from .synthesis import KAxisTPADA


class UniversalTPADA(tf.Module):
    def __init__(self, box_cfgs, d=64, depth=2, heads=4, d_w=8, n_p=16, t_final=1.0,
                 c_out=6, c_in_grid=None, encoder="attn", rec_steps=1, name="universal",
                 spatial_ada=True, time_basis="legendre", spatial_basis="global",
                 n_patches=4, patch_order=3, active_banks=None, moe_experts=0, moe_topk=2,
                 dual_core=False, sl_steps=0):
        """box_cfgs: {2: (n, n), 3: (n, n, n)} latent-box dims per K.
        encoder: 'attn' (cross-attention) | 'bin' (GINO cell-binning, O(N), scalable).
        rec_steps: unsteady recursion depth (1 = single-shot; >1 chains trajectory segments).
        spatial_ada: Fourier synthesis variant — True = V-int (k=1 ADA modes), False = V-spec.
        time_basis: 'legendre' (default) | 'fourier' (anti-derivative Fourier time).
        spatial_basis: 'global' | 'local' (patch-Legendre compact-support spectral element)."""
        super().__init__(name=name)
        self.rec_steps = rec_steps
        self.sl_steps = int(sl_steps)            # semi-Lagrangian latent sub-steps (0/1 = single-window)
        self._geom_box = None                    # set by encode(); wall geometry for the BL bank
        self.dual = bool(dual_core)
        self._h1box = self._h2box = None         # stashed by encode(); consumed by ortho_aux()
        with self.name_scope:
            self.core = LatentBoxCore(d, depth=depth, heads=heads,
                                      moe_experts=moe_experts, moe_topk=moe_topk)
            # DUAL-CORE: a second, independent transformer stack producing an ORTHOGONAL hidden box
            # h2. A decorrelation aux pushes h1 _|_ h2 so the two spans are complementary; the banks
            # read both (h2 via zero-init ungated heads -> warm-safe). Dense (no MoE) to isolate the
            # decomposition lever from raw capacity.
            self.core2 = (LatentBoxCore(d, depth=depth, heads=heads) if self.dual else None)
            self.penc = BinEncoder(d) if encoder == "bin" else PointEncoder(d)
            self.grid_proj = tf.keras.layers.Dense(d, name="grid_in")  # frames-as-channels -> d
            # UNSTEADY: combo tendency banks (base small-init + zero-init gated mechanisms).
            # Base is small (NOT zero): hard IC comes from the time anti-derivative G(0)=0, not W=0.
            # PER-BASIS W: banks emit 2x width — first d_w*n_p -> Fourier panels, second -> Legendre
            # panels — so each basis carries its OWN spatial content (not just a gain on a shared W).
            # Warm-start: duplicate-load (both halves = old shared W) -> output unchanged at load.
            self.banks = TendencyBanks(2 * d_w * n_p, d, active=active_banks, dual=self.dual)
            self.dec = ContinuousDecoder(c_out)
            # STEADY: direct elliptic field head (no time, no tendency banks).
            # PER-BASIS field: 2x c_out — [Fourier field | Legendre field], same duplicate-load.
            self.steady = SteadyField(2 * c_out, d)
            # RECURSION feedback (zero-init -> identity at load): end-of-window latent state fed
            # back into the box features so the banks re-evaluate on the EVOLVED state (combo-rec).
            self.rec_proj = tf.keras.layers.Dense(d, kernel_initializer="zeros", name="rec_fb")
            # LOCAL per-patch INDEPENDENT W heads (spatial_basis='local', 'global IC read, local
            # write'): P independent Dense heads; each cell adds the head of its patch, SUMMED over
            # axes -> per-patch-distinct W generation from independent weights. Heads small-init +
            # a zero-init gate g_extra so the branch is OFF at load (warm-safe) yet has live gradient
            # (small heads -> dL/dg_extra != 0, avoids the both-zero dead saddle). K-agnostic: the P
            # heads are shared across axes and 2D/3D (each axis selects among the same P heads).
            self.local_heads = (spatial_basis == "local")
            self._n_patches = int(n_patches)
            if self.local_heads:
                self.pheads = [tf.keras.layers.Dense(
                    2 * d_w * n_p, name=f"phead{p}",
                    kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3))
                    for p in range(int(n_patches))]
                self.g_extra = tf.Variable(tf.zeros([2 * d_w * n_p]), name="g_extra")
        self.d_w, self.n_p = d_w, n_p
        self.synth = {K: KAxisTPADA(dims, n_p, t_final=t_final, name=f"tp{K}d",
                                    spatial_ada=spatial_ada, time_basis=time_basis,
                                    spatial_basis=spatial_basis, n_patches=n_patches,
                                    patch_order=patch_order)
                      for K, dims in box_cfgs.items()}

    def encode(self, coords, feats, K, roles=None):
        """point cloud -> latent box features h (B, *dims, d). Stashes the raw geom box (wall SDF/
        normals binned to the box) for the boundary-layer bank, which needs the wall-normal."""
        h, self._geom_box = self.penc(coords, feats, dims=self.synth[K].dims, roles=roles)
        self._h1box = self.core(h, roles=roles)
        self._h2box = self.core2(h, roles=roles) if self.dual else None
        return self._h1box

    def moe_aux(self):
        """MoE load-balance aux loss (0 if core is not MoE). Add to the training objective."""
        return self.core.aux()

    def ortho_aux(self):
        """Decorrelation between the two hidden boxes h1, h2 (dual_core): flatten to (M,d), z-normalize
        each feature dim, then penalize the whole d x d cross-correlation matrix (VICReg-style). Drives
        the two sources to orthogonal subspaces. 0 if not dual or a box isn't populated yet."""
        if not self.dual or self._h1box is None or self._h2box is None:
            return tf.constant(0.0)
        d = self._h1box.shape[-1]
        z1 = tf.reshape(self._h1box, [-1, d])
        z2 = tf.reshape(self._h2box, [-1, d])
        z1 = (z1 - tf.reduce_mean(z1, 0)) / (tf.math.reduce_std(z1, 0) + 1e-5)
        z2 = (z2 - tf.reduce_mean(z2, 0)) / (tf.math.reduce_std(z2, 0) + 1e-5)
        M = tf.cast(tf.shape(z1)[0], tf.float32)
        C = tf.matmul(z1, z2, transpose_a=True) / M           # (d,d) cross-correlation
        return tf.reduce_mean(tf.square(C))

    def _w(self, h, dims, K, roles=None, op_multihot=None, steady=False):
        """-> (W_fourier, W_legendre), each (B, *dims, d_w, n_p): contiguous halves of the banks
        output, so a duplicate-load of the old shared-W weights makes both halves identical.
        LOCAL towers add per-patch independent-head W (gated OFF at load by g_extra)."""
        W = self.banks(h, K=K, roles=roles or list(range(K)), op_multihot=op_multihot,
                       geom_box=self._geom_box, h2=self._h2box, steady=steady)
        if self.local_heads:
            W = W + self.g_extra * self._patch_W(h, dims, K)
        Wf, Wl = tf.split(W, 2, axis=-1)
        shp = tf.concat([tf.shape(h)[:1], dims, [self.d_w, self.n_p]], 0)
        return tf.reshape(Wf, shp), tf.reshape(Wl, shp)

    def _patch_W(self, h, dims, K):
        """Per-patch independent-head contribution: stack the P head outputs, then for each axis
        select each cell's patch-head (one-hot over P) and sum over axes -> (B, *dims, 2*d_w*n_p).
        Cell in patch (p_1,..,p_K) gets sum_a phead_{p_a}(h) — per-patch-distinct, independent W."""
        P = self._n_patches
        Hp = tf.stack([ph(h) for ph in self.pheads], axis=-2)      # (B, *dims, P, out)
        acc = 0.0
        for a in range(K):
            n = int(dims[a])
            idx = (tf.range(n) * P) // n                           # (n_a,) patch per cell along a
            oh = tf.one_hot(idx, P)                                # (n_a, P)
            bshape = [1] * (K + 1) + [P, 1]
            bshape[1 + a] = n                                      # (1,..,n_a,..,1, P, 1)
            acc = acc + tf.reduce_sum(Hp * tf.reshape(oh, bshape), axis=-2)
        return acc

    def steady_field_at(self, h, K, query, op_multihot=None):
        """STEADY prediction: box features -> elliptic field -> value at query points.
        No IC anchor, no time integration. DUAL BASIS: Fourier + Legendre-ADA (both open).
        op_multihot is unused here (ensemble-interface compat: the tower gate consumes it)."""
        field_box = self.steady(h)                                # (B,*dims,2*c_out) per-basis
        f_f, f_l = tf.split(field_box, 2, axis=-1)                # [Fourier field | Legendre field]
        return (self.synth[K].field_at_ada(f_f, query)            # (B,Q,c_out) Fourier-ADA (V-int)
                + self.synth[K].field_at_leg(f_l, query))         # + Legendre-ADA (own content)

    def from_grid(self, frames, roles=None):
        """frames (B, T_in, n_1..n_K, C) -> (latent A, ic field). Grid = degenerate encoder."""
        K = frames.shape.rank - 3
        dims = frames.shape[2:2 + K]
        B = tf.shape(frames)[0]
        x = tf.transpose(frames, [0] + list(range(2, 2 + K)) + [1, K + 2])
        x = tf.reshape(x, tf.concat([[B], dims, [frames.shape[1] * frames.shape[-1]]], 0))
        self._geom_box = None                                      # grid path has no wall geometry
        gx = self.grid_proj(x)
        h = self.core(gx, roles=roles)
        self._h1box = h
        self._h2box = self.core2(gx, roles=roles) if self.dual else None
        Wf, _ = self._w(h, dims, K, roles)                         # grid path is Fourier-only
        A = self.synth[K].coef(Wf)
        ic = frames[:, -1]                                         # (B, *dims, C) anchor
        return A, ic, K

    def from_points(self, coords, feats, K, roles=None, op_multihot=None, steady=False):
        """coords (B,N,K) in [0,1]^K, feats (B,N,F) -> (latent A, None). UNSTEADY tendency path.
        rec_steps>1: re-apply banks on the end-of-window evolved latent (zero-init feedback).
        steady: drop the transient banks (relaxation trajectory has no physical time)."""
        dims = self.synth[K].dims
        h = self.encode(coords, feats, K, roles=roles)
        Wf, Wl = self._w(h, dims, K, roles, op_multihot, steady=steady)
        A = self.synth[K].coef(Wf)
        for _ in range(self.rec_steps - 1):
            z_end = self.synth[K].grid_traj(A, np.array([1.0], np.float32))[:, 0]   # (B,*dims,d_w)
            h = h + self.rec_proj(z_end)                                            # feed evolved state back
            Wf, Wl = self._w(h, dims, K, roles, op_multihot, steady=steady)
            A = self.synth[K].coef(Wf)
        A_leg = self.synth[K].coef_leg(Wl)                         # DUAL BASIS: Legendre's OWN W
        return A, A_leg, K

    def from_points_sl(self, coords, feats, K, roles=None, op_multihot=None):
        """SEMI-LAGRANGIAN latent integration. Split [0,1] into sl_steps sub-windows; at each,
        re-evaluate the banks on the EVOLVED latent (advection u=vel(h) follows the pathline) and
        keep that segment's Fourier coefficients. Returns (A_list, A_leg, K): A_list = per-segment
        Fourier coefs, A_leg = single-window Legendre (kept unchanged for warm-safety). The banks
        being re-evaluated along the trajectory is the fix the advection-dominated NS/turbulence
        laggards need — a single-shot analytic integration freezes the operator over the window and
        cannot follow curving pathlines. rec_proj is zero-init so at load every segment is identical
        and the query-time telescoping (point_traj_sl_g) recovers the single-window trajectory."""
        dims = self.synth[K].dims
        h0 = self.encode(coords, feats, K, roles=roles)
        N = self.sl_steps
        ts = np.linspace(0.0, 1.0, N + 1).astype(np.float32)       # sub-window boundaries in [0,1]
        _, Wl = self._w(h0, dims, K, roles, op_multihot)
        A_leg = self.synth[K].coef_leg(Wl)                         # Legendre: single-window from h_enc
        A_list = []
        h, zbox = h0, None
        for k in range(N):
            Wf, _ = self._w(h, dims, K, roles, op_multihot)
            A_k = self.synth[K].coef(Wf)
            A_list.append(A_k)
            inc = (self.synth[K].grid_traj(A_k, ts[k + 1:k + 2])[:, 0]   # box increment over [t_k,t_{k+1}]
                   - self.synth[K].grid_traj(A_k, ts[k:k + 1])[:, 0])    # (B,*dims,d_w)
            zbox = inc if zbox is None else zbox + inc
            h = h0 + self.rec_proj(zbox)                           # re-evaluate operator at evolved state
        return A_list, A_leg, K

    def point_traj_sl_g(self, A_list, A_leg, K, pts, Gq_seg, Gq, ic_at_pts, op_multihot=None,
                        q_chunk=64, geom_q=None):
        """Compose the semi-Lagrangian query trajectory. Gq_seg (B,Q,N,Np) = per-segment time-map
        DIFFERENCE qtm(min(t_q,t_{k+1})) - qtm(min(t_q,t_k)); Gq (B,Q,Np) = single-window map for
        the Legendre path. z_sl(t_q) = Σ_k point_modes(A_k)·Gq_seg[:,:,k,:] (Fourier, telescopes to
        point_modes(A)·qtm(t_q) at load) + Legendre(A_leg, Gq). Warm-safe & exact.
        geom_q (B,Q,GEOM_W): per-query geometry for the decoder geometry-FiLM (None -> off)."""
        Q = int(pts.shape[1])
        outs = []
        for s in range(0, Q, q_chunk):
            e = min(s + q_chunk, Q)
            zf = 0.0
            for k, A_k in enumerate(A_list):
                zf = zf + self.synth[K].point_traj_perquery(A_k, pts[:, s:e], Gq_seg[:, s:e, k, :])
            z = zf + self.synth[K].point_traj_perquery_leg(A_leg, pts[:, s:e], Gq[:, s:e])
            gq = geom_q[:, s:e] if geom_q is not None else None
            outs.append(ic_at_pts[:, s:e] + self.dec(z, coords=pts[:, s:e], geom=gq))
        return tf.concat(outs, axis=1)

    def grid_traj(self, A, ic, K, times):
        """u (B, T, *dims, C) on the box centers; hard IC at times[...]=0."""
        z = self.synth[K].grid_traj(A, times)                      # (B, T, *dims, d_w)
        return ic[:, None] + self.dec(z)

    def point_traj(self, A, ic, K, pts, times, ic_at_pts=None):
        """u (B, T, Q, C) at arbitrary points/times. IC anchor: caller-provided values
        at pts, or band-limited interpolation of the grid IC field."""
        z = self.synth[K].point_traj(A, pts, times)                # (B, T, Q, d_w)
        if ic_at_pts is None:
            if ic is None:
                raise ValueError("point-input runs must pass ic_at_pts")
            ic_at_pts = self.synth[K].field_at(ic, pts)            # (B, Q, C)
        return ic_at_pts[:, None] + self.dec(z)

    def point_traj_perquery(self, A, A_leg, K, pts, t_q, ic_at_pts, q_chunk=64, geom_q=None):
        """Eager convenience: builds Gq from numpy t_q then delegates. (Tests / single-example.)"""
        Gq = self.synth[K].query_time_map(t_q)                     # (B,Q,Np) tensor
        return self.point_traj_perquery_g(A, A_leg, K, pts, Gq, ic_at_pts, q_chunk=q_chunk, geom_q=geom_q)

    def point_traj_perquery_g(self, A, A_leg, K, pts, Gq, ic_at_pts, op_multihot=None,
                              q_chunk=64, geom_q=None):
        """Graph-friendly collocation eval with a PRECOMPUTED time map Gq (B,Q,Np) tensor.
        DUAL BASIS: latent trajectory = Fourier path + Legendre-ADA path (both open). Q is
        chunked so the 3D synthesis einsum intermediate stays bounded; pure TF for graph mode.
        geom_q (B,Q,GEOM_W): per-query geometry for the decoder geometry-FiLM (None -> off).
        op_multihot unused (ensemble-interface compat)."""
        Q = int(pts.shape[1])
        outs = []
        for s in range(0, Q, q_chunk):
            e = min(s + q_chunk, Q)
            z = (self.synth[K].point_traj_perquery(A, pts[:, s:e], Gq[:, s:e])          # Fourier
                 + self.synth[K].point_traj_perquery_leg(A_leg, pts[:, s:e], Gq[:, s:e]))  # Legendre-ADA
            gq = geom_q[:, s:e] if geom_q is not None else None
            outs.append(ic_at_pts[:, s:e] + self.dec(z, coords=pts[:, s:e], geom=gq))  # coord + geometry FiLM
        return tf.concat(outs, axis=1)


class BasisTowerEnsemble(tf.Module):
    """N parallel UniversalTPADA towers — per-tower basis variants (V-spec Fourier,
    V-int Fourier-ADA, ...) each a FULL weight copy warm-started from its own ckpt
    (sparse-upcycling style). A per-family linear gate on op_multihot decides the
    combination (all bases ON; the NN learns the mixture):

        u = ic + sum_i  g_i(op) * dec_i(z_i),   g init = 1/N (uniform mean)

    Interface mirrors UniversalTPADA so core.pretrain drives either transparently."""

    def __init__(self, towers, name="ens"):
        super().__init__(name=name)
        self.towers = towers
        self.d_w, self.n_p = towers[0].d_w, towers[0].n_p
        self.synth = towers[0].synth              # time maps/constants identical across towers
        with self.name_scope:
            self.gate = tf.keras.layers.Dense(
                len(towers), name="ens_gate", kernel_initializer="zeros",
                bias_initializer=tf.keras.initializers.Constant(1.0 / len(towers)))

    def encode(self, coords, feats, K, roles=None):
        return tuple(t.encode(coords, feats, K, roles=roles) for t in self.towers)

    def from_points(self, coords, feats, K, roles=None, op_multihot=None):
        As, Als = [], []
        for t in self.towers:
            A, A_leg, _ = t.from_points(coords, feats, K, roles=roles, op_multihot=op_multihot)
            As.append(A); Als.append(A_leg)
        return tuple(As), tuple(Als), K

    def point_traj_perquery_g(self, A, A_leg, K, pts, Gq, ic_at_pts, op_multihot=None,
                              q_chunk=64):
        g = self.gate(op_multihot) if op_multihot is not None else \
            tf.fill([tf.shape(pts)[0], len(self.towers)], 1.0 / len(self.towers))
        u = ic_at_pts
        zero_ic = tf.zeros_like(ic_at_pts)
        for i, t in enumerate(self.towers):
            d = t.point_traj_perquery_g(A[i], A_leg[i], K, pts, Gq, zero_ic, q_chunk=q_chunk)
            u = u + g[:, i, None, None] * d
        return u

    def point_traj_perquery(self, A, A_leg, K, pts, t_q, ic_at_pts, op_multihot=None):
        Gq = self.synth[K].query_time_map(t_q)
        return self.point_traj_perquery_g(A, A_leg, K, pts, Gq, ic_at_pts, op_multihot)

    def steady_field_at(self, hs, K, query, op_multihot=None):
        g = self.gate(op_multihot) if op_multihot is not None else \
            tf.fill([tf.shape(query)[0], len(self.towers)], 1.0 / len(self.towers))
        u = 0.0
        for i, t in enumerate(self.towers):
            u = u + g[:, i, None, None] * t.steady_field_at(hs[i], K, query)
        return u
