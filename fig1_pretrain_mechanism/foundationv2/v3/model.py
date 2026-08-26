"""V3 model — one AR step of the bylfa-paradigm universal operator (DESIGN_V3 §2.2).

  window at nodes -> BinEncoder (kernel scatter, geometry-gated) -> latent box
  -> FiLM(e_cond)-modulated axial core (K-agnostic; same weights K=2/K=3)
  -> TendencyBanks (12 mechanisms, op-token masked) -> W panels (n_p over the sub-step)
  -> exact panel anti-derivative to the segment end: z = mean_p W_p  (ADA, hard z(0)=0)
  -> native-res decode at query points: multilinear latent interp + bias-free MLP
     x (1 + coord-Fourier FiLM) x (1 + native geometry FiLM)      [core/pointio.py]
  u_next(q) = u_prev(q) + delta(q)  (+ gated backward-warp transport on dense 2D grids)

One weight set serves every (K, steady, dense2d) bucket; per-bucket tf.function traces
differ only in shapes/python flags, never in variables.
"""
from __future__ import annotations

import itertools

import numpy as np
import tensorflow as tf

from core.axops import (AxialAttention, AxialConv, MoEPointwiseMLP, PointwiseMLP,
                        RankFreeLN)
from core.banks import TendencyBanks
from core.dictdec import anchor_features, n_dict_slots
from core.binenc import BinEncoder
from core.pointio import ContinuousDecoder

from .cond import CondEncoder

ADV_OP = 1                # data.symbolic.OPERATOR_VOCAB index of "advection"


def interp_box(box, q, dims):
    """Multilinear interpolation of box (B,*dims,C) at q (B,Q,K) in [0,1]^K -> (B,Q,C).
    Cell centers at (i+0.5)/n; edges clamped (safe for periodic and walled alike)."""
    K = len(dims)
    c = [q[..., a] * dims[a] - 0.5 for a in range(K)]                    # continuous index
    i0 = [tf.floor(x) for x in c]
    fr = [x - f for x, f in zip(c, i0)]
    i0 = [tf.cast(f, tf.int32) for f in i0]
    B = tf.shape(q)[0]
    flat = tf.reshape(box, [B, -1, box.shape[-1]])                       # (B,ncell,C)
    strides = [int(np.prod(dims[a + 1:])) for a in range(K)]
    out = 0.0
    for off in itertools.product((0, 1), repeat=K):
        idx = 0
        w = 1.0
        for a in range(K):
            ia = tf.clip_by_value(i0[a] + off[a], 0, dims[a] - 1)
            idx = idx + ia * strides[a]
            w = w * (fr[a] if off[a] else (1.0 - fr[a]))
        out = out + tf.cast(w[..., None], flat.dtype) * tf.gather(flat, idx, batch_dims=1)
    return out


class V3Model(tf.keras.Model):
    def __init__(self, S=9, d=384, depth=8, d_w=32, n_p=8, d_cond=128,
                 dims2=(64, 64), dims3=(32, 32, 32), t_in=10, transport=True,
                 n_fams=0, n_slices=0, moe_experts=0, moe_topk=2, moe_capacity=0.0,
                 gate_cond=0, gate_experts=0, n_semantic=0, d_shape=0, dict_shift=0,
                 dict_decode=1, fam_head=0, name="v3", **kw):
        super().__init__(name=name, **kw)
        self.S, self.d, self.depth = S, d, depth
        # Slots [n_semantic, S) are MEMORY: recurrent state, not a physical field.
        self.n_semantic = int(n_semantic) if n_semantic else int(S)
        self.d_w, self.n_p, self.t_in = d_w, n_p, t_in
        self.box_dims = {2: tuple(dims2), 3: tuple(dims3)}
        self.transport = bool(transport)
        self.n_fams = int(n_fams)
        self.enc = BinEncoder(d, n_slices=n_slices, name="enc")
        self.cond = CondEncoder(d_cond, d, depth, name="cond")
        # SPARSE UPCYCLING (the standard dense->large recipe: Komatsuzaki et al. 2023). The block
        # MLP becomes E experts with a top-k router; each expert is seeded from the trained dense
        # MLP, so the model's function is preserved at load and capacity grows ~E-fold. This is
        # also what broke the previous generation's wall (dense 18.83 -> E16 18.09) where plain
        # depth growth did nothing. Expert names carry an `_e{e}` infix so the upcycle loader can
        # map each of them back to the one dense source tensor.
        self.moe_experts = int(moe_experts)
        self.blocks = []
        for i in range(depth):
            mlp = (MoEPointwiseMLP(d, n_experts=self.moe_experts, top_k=moe_topk,
                                   capacity=moe_capacity,
                                   name=f"b{i}_mlp") if self.moe_experts
                   else PointwiseMLP(d, name=f"b{i}_mlp"))
            self.blocks.append((AxialConv(d, name=f"b{i}_conv"),
                                AxialAttention(d, name=f"b{i}_attn"), mlp))
        self.ln_out = RankFreeLN(d, name="ln_out")
        self.banks = TendencyBanks(d_w * n_p, d, name="banks",
                                   gate_cond=gate_cond, gate_experts=gate_experts)
        # ITEM G: native anchor dictionary (core/dictdec.py). n_dict=0 restores the r19 decode.
        self.dict_shift = bool(dict_shift)
        self.n_dict = n_dict_slots(self.dict_shift) if int(dict_decode) else 0
        self.dec = ContinuousDecoder(S, hidden=96, n_fams=self.n_fams, n_dict=self.n_dict,
                                     n_dict_out=self.n_semantic, fam_only=bool(fam_head),
                                     name="dec")
        # GINO lever: the dense SDF volume enters the box as an extra channel through a zero-init
        # gate (inert for the 24 families that have none, warm-safe for the rest).
        self.sdf_proj = tf.keras.layers.Dense(d, activation="gelu", name="sdf_proj")
        self.sdf_gate = tf.keras.layers.Dense(d, use_bias=False, kernel_initializer="zeros",
                                              name="sdf_gate")
        # GLOBAL SHAPE CODE (the MARIO lever). Every other route by which geometry enters this
        # model is LOCAL — per-point SDF/normal/curvature appended to a feature vector, or a
        # scatter-mean that smears the surface across cells. Nothing anywhere answers "what shape
        # is this". For drivaernet that is fatal: every sample is a DIFFERENT car, so the model
        # has to re-infer the body from local distances each time, through a box that averages
        # them. It is the only family that never moved across the whole session (30.6 at r10,
        # 31.4 at r16) under slot splits, memory channels, MoE folding and gate specialisation.
        # MARIO's answer is to encode the shape ONCE into a compact code and let it MODULATE the
        # field network, keeping shape and coordinate separate. The material is already here:
        # sdf_box is computed and currently only added to the box as a local term. A couple of
        # K-agnostic axial convs plus a global mean turn it into that code, which then joins the
        # symbolic embedding e and so reaches every FiLM, every bank gate and the decoder.
        self.d_shape = int(d_shape)
        if self.d_shape:
            self.sh_in = tf.keras.layers.Dense(self.d_shape, activation="gelu", name="sh_in")
            self.sh_c1 = AxialConv(self.d_shape, name="sh_c1")
            self.sh_c2 = AxialConv(self.d_shape, name="sh_c2")
            self.sh_out = tf.keras.layers.Dense(d_cond, use_bias=False,
                                                kernel_initializer="zeros", name="sh_out")
        if n_slices:
            # SURFACE-MANIFOLD DECODE PATH: queries cross-attend to the encoder's slice tokens,
            # bypassing the volumetric box entirely. This is what gives co-dimension-1 families
            # (3D car surfaces, 2D airfoil curves) native-resolution structure — the box is
            # ~97% empty for them. Zero-init output gate => inert at load, warm-safe.
            # Explicit cross-attention (not keras MultiHeadAttention) because the decode needs a
            # DISTANCE BIAS on the logits, which the keras layer cannot take.
            self.sf_h, self.sf_dk = 4, max(d // 4, 8)
            self.sf_q = tf.keras.layers.Dense(self.sf_h * self.sf_dk, name="sf_q")
            # STATE-AWARE READ. The encoder already assigns points to slices the way Transolver
            # does — softmax(sl_assign(tok)) over a token that carries coordinates, field values
            # AND geometry, i.e. by physical state. The read-back did not match: the query was
            # built from coordinates alone plus a distance bias, so a slice formed by physics was
            # being retrieved by position. That inconsistency is the likeliest reason the whole
            # surface path sits gated off (sf_out 0.30, bl_gate 0.117, sk_gate 0.081, against a
            # box boundary-layer gate of 1.91). The query now also sees the interpolated latent
            # state at the query point, through a zero-init projection so nothing changes at load.
            self.sf_qs = tf.keras.layers.Dense(self.sf_h * self.sf_dk, use_bias=False,
                                               kernel_initializer="zeros", name="sf_qs")
            self.sf_k = tf.keras.layers.Dense(self.sf_h * self.sf_dk, name="sf_k")
            self.sf_v = tf.keras.layers.Dense(self.sf_h * self.sf_dk, name="sf_v")
            self.sf_out = tf.keras.layers.Dense(d_w, use_bias=False, kernel_initializer="zeros",
                                                name="sf_out")
            # LOCALITY. With global attention the decode is a smooth combination of all M tokens,
            # so surface detail cannot survive no matter how many tokens there are — measured on
            # drivaernet: mean and amplitude land exactly (|dDC| .02, |dstd| .03) while the spatial
            # correlation is only .91-.95, i.e. the field is right but blurred. A learned
            # inverse-length scale turns the query->token attention into a local one: the logit
            # gets -beta * |q - centroid|^2, so a query reads the tokens on its own patch of the
            # manifold. beta is softplus-parameterised and starts near 1/(typical spacing)^2.
            self.sf_beta = tf.Variable(4.0, dtype=tf.float32, name="sf_beta")
            # NATIVE-RESOLUTION BOUNDARY-LAYER BANK. There is already a `boundary_layer` mechanism
            # in TendencyBanks and its gate is open (0.137, mature-bank level), yet the three
            # turbulent external-aerodynamics families are the ones that lag (airfrans 24.8,
            # drivaernet 30.4, shapenet 13.7) while the inviscid / elastic / laminar steady families
            # sit at 5.8-16.9. The reason is that the box bank differences the wall-normal direction
            # on a 64^2 / 32^3 lattice, and at Re~1e6 the layer is thinner than one cell — a
            # derivative across a cell that contains the whole structure carries no information, and
            # no amount of capacity fixes it (21M -> 92M moved drivaernet 31.6 -> 30.4).
            #
            # So this bank does not live on the box. It is PARAMETRIC in the wall distance, the way
            # wall functions are: the value at a point is the wall state times a profile in y+,
            #     W_bl(x) = head( s(x_wall) (x) profile(d(x)) ),  x_wall = x - d*n
            # with the wall state read from the surface-manifold tokens at the FOOT POINT and the
            # profile spanned by [d, log(1+d), exp(-d/tau), d^2] (viscous sublayer + log layer).
            # Being a tendency it enters the time integration like any other mechanism; being
            # parametric in d it is resolution-free. Zero-init gate -> inert at load; d=0 (surface-
            # only families) or geom=0 (grids) -> degenerates to the plain surface read / nothing.
            self.bl_prof = tf.keras.layers.Dense(d_w, activation="gelu", name="bl_prof")
            self.bl_head = tf.keras.layers.Dense(d_w, name="bl_head")
            self.bl_gate = tf.keras.layers.Dense(d_w, use_bias=False, kernel_initializer="zeros",
                                                 name="bl_gate")
            self.bl_tau = tf.Variable(0.1, dtype=tf.float32, name="bl_tau")
            # SURFACE-KINEMATICS BANK. The existing `kinematic` bank takes invariants of the
            # velocity-gradient tensor and `geometry` takes the curvature of the LATENT iso-
            # surfaces — neither touches the differential geometry of the actual body. Yet that is
            # what sets surface pressure: along a curved wall the normal pressure gradient is
            # dp/dn ~ rho*U^2*kappa, and the TANGENTIAL pressure gradient decides whether the
            # boundary layer separates. Curvature was being fed as a raw geom channel with no
            # mechanism able to form those products. Features are the principal curvatures and
            # their coupling to the tangential kinetic energy; the head is native-resolution
            # (queries, not the box) and zero-init gated.
            self.sk_vel = tf.keras.layers.Dense(3, name="sk_vel")     # tangential velocity proxy
            self.sk_head = tf.keras.layers.Dense(d_w, activation="gelu", name="sk_head")
            self.sk_gate = tf.keras.layers.Dense(d_w, use_bias=False, kernel_initializer="zeros",
                                                 name="sk_gate")
        if self.transport:
            # backward-warp transport (adv_bank mechanism): displacement from the latent box,
            # zero-init -> identity warp at start; per-slot gate from e_cond, zero-init, and
            # masked by the advection operator token. Dense-2D-grid buckets only.
            self.disp = tf.keras.layers.Dense(2, kernel_initializer="zeros", name="disp")
            self.tgate = tf.keras.layers.Dense(S, kernel_initializer="zeros",
                                               use_bias=False, name="tgate")

    def warm_build(self, S=None, cond_dim=None):
        """Create EVERY variable (both K branches + transport) with tiny synthetic inputs, so
        the variable set is complete before ckpt load / optimizer capture / tf.function trace."""
        from .cond import COND_DIM
        cd = cond_dim or COND_DIM
        S = self.S if S is None else S      # was pinned at 9; the slot split made that a mismatch
        for K in (2, 3):
            N, Q = 16, 8
            cn = tf.random.uniform([1, N, K])
            win = tf.zeros([1, self.t_in, N, S])
            fm = tf.ones([1, self.t_in])
            gn = tf.zeros([1, N, 8])
            cond = tf.zeros([1, cd])
            from data.symbolic import OPERATOR_VOCAB as _OV
            op = tf.ones([1, len(_OV)])
            cq = tf.random.uniform([1, Q, K])
            gq = tf.zeros([1, Q, 8])
            up = tf.zeros([1, Q, S])
            pg = tf.zeros([1, 4, 4, S]) if K == 2 else None
            fid = tf.zeros([1], tf.int32) if self.n_fams else None
            sb = tf.zeros([1, *self.box_dims[K], 1])
            self.step(cn, win, fm, gn, K, list(range(K)), cond, op, cq, gq, up,
                      steady=False, prev_grid=pg, grid_dims=(4, 4) if K == 2 else None,
                      fam_id=fid, sdf_box=sb, prev2_grid=pg)   # prev2 builds the item-H slot too

    def step_ss(self, coords_node, win, fmask, geom_node, K, roles, cond, op_multihot,
                coords_q, geom_q, u_prev_q, kf, fam_id=None, sdf_box=None, box_dims=None):
        """SINGLE-SHOT mode (MOTION-NCS, 2026-08-06): ONE trunk evaluation whose panels span the
        WHOLE horizon; every future frame is decoded from the exact partial-panel anti-derivative.

        This is the per-family temporal-mode split of PLAN_NCS.md: smooth families (ACE,
        diff_react, Wave-Layer) measurably prefer seeing the global temporal structure in one
        segment over a 10-step re-anchored rollout — for second-order-in-time physics the
        re-anchor discards the velocity state — and the trunk (encoder + core + banks) runs once
        instead of kf times, which is where the wall-clock lives. n_p = 16 panels over a 10-frame
        horizon is 1.6 panels/frame, the measured adequacy threshold (07-20 N_p sweep).

        z(t) at frame fraction t = k/kf is the anti-derivative of the piecewise-constant panel
        tendency: z(t) = (sum_{p<m} W_p + f*W_m)/n_p with t*n_p = m + f. At t=1 this reduces to
        mean_p W_p — bit-identical to the AR step's readout, so both modes share one calculus.
        Panel interpolation to the query points commutes with the (linear) partial sums, so the
        box -> query interp happens ONCE at width d_w*n_p, not kf times.

        No AR feedback, no transport warp (the ss families carry no advection operator: the
        warp gate is op-masked off anyway), no surface path (grid families only), no item-G
        dictionary (removed in MOTION-NCS). Hard IC: z(0) = 0 and the decoder is bias-free, so
        u(t->0) -> u_prev exactly. Returns (B, kf, Q, S)."""
        dims = tuple(box_dims) if box_dims else self.box_dims[K]
        feats = self.feats_from_win(win, fmask, geom_node)
        box, geom_box, _ = self.enc(coords_node, feats, dims=dims, roles=roles)
        if sdf_box is not None:
            box = box + self.sdf_gate(self.sdf_proj(sdf_box))
            geom_box = tf.concat([tf.cast(sdf_box, geom_box.dtype), geom_box[..., 1:]], -1)
        e = tf.cast(self.cond(cond), box.dtype)
        h = self.core(box, e, roles)
        W = self.banks(h, K=K, roles=roles, op_multihot=op_multihot,
                       geom_box=geom_box, steady=False, e=e)          # (B,*dims,d_w*n_p)
        Wq = interp_box(W, coords_q, dims)                            # (B,Q,d_w*n_p)
        Wq = tf.reshape(Wq, [tf.shape(Wq)[0], -1, self.d_w, self.n_p])
        cum = tf.cumsum(Wq, axis=-1) / float(self.n_p)                # z at panel ends
        if self.n_semantic < self.S:                                  # memory slots: no anchor sum
            keep = tf.concat([tf.ones([self.n_semantic], u_prev_q.dtype),
                              tf.zeros([self.S - self.n_semantic], u_prev_q.dtype)], 0)
            u_prev_q = u_prev_q * keep[None, None, :]
        outs = []
        for k in range(1, kf + 1):
            t = k / float(kf) * self.n_p                              # frame time in panel units
            m = int(np.ceil(t)) - 1                                   # panel containing t
            f = t - m                                                 # fraction of panel m, (0,1]
            z_k = (f / float(self.n_p)) * Wq[..., m]
            if m > 0:
                z_k = z_k + cum[..., m - 1]
            delta = self.dec(z_k, coords=coords_q, roles=roles, geom=geom_q,
                             cond=e, fam_id=fam_id, dfeat=None)
            outs.append(u_prev_q + tf.cast(delta, u_prev_q.dtype))
        return tf.stack(outs, 1)                                      # (B,kf,Q,S)

    def _surface_read(self, slices, coords_q, roles, state=None):
        """LOCAL cross-attention from query points to the surface-manifold tokens.

        The tokens carry the manifold's own spatial structure (the volumetric box cannot: a
        2-manifold fills ~3% of a 3D box). Attention is biased by -beta*|q - centroid|^2 so a
        query reads the tokens on ITS patch of the surface. Without that bias the read is a
        smooth global mixture and fine surface detail is unrecoverable however many tokens exist
        — the measured drivaernet failure mode is exactly that: mean and amplitude correct
        (|dDC| .02, |dstd| .03) but spatial correlation only .91-.95."""
        sl, cen = slices                                                 # (B,M,d), (B,M,K)
        B = tf.shape(coords_q)[0]
        Q = tf.shape(coords_q)[1]
        M = tf.shape(sl)[1]
        H, dk = self.sf_h, self.sf_dk
        q = self.sf_q(self.enc.coord(coords_q, roles=roles))
        if state is not None:
            q = q + self.sf_qs(tf.cast(state, q.dtype))       # physical state joins the query
        qh = tf.reshape(q, [B, Q, H, dk])
        kv = sl + self.enc.coord(cen, roles=roles)                       # position-tagged tokens
        kh = tf.reshape(self.sf_k(kv), [B, M, H, dk])
        vh = tf.reshape(self.sf_v(kv), [B, M, H, dk])
        logit = tf.einsum("bqhd,bmhd->bhqm", qh, kh) / (dk ** 0.5)
        cq32 = tf.cast(coords_q, tf.float32)                             # geometry in f32; the
        cen32 = tf.cast(cen, tf.float32)                                 # token centroids may be
        d2 = tf.reduce_sum(tf.square(cq32[:, :, None, :] - cen32[:, None, :, :]), -1)  # bf16
        d2 = tf.cast(tf.nn.softplus(tf.cast(self.sf_beta, tf.float32)) * d2, logit.dtype)
        logit = logit - d2[:, None]                                      # locality bias
        att = tf.nn.softmax(logit, -1)
        out = tf.einsum("bhqm,bmhd->bqhd", att, vh)
        return tf.reshape(out, [B, Q, H * dk])

    def _bl_bank(self, slices, coords_q, geom_q, roles, e):
        """Wall-normal boundary-layer tendency at NATIVE resolution (see __init__ for why this
        cannot be a box bank). Returns (B,Q,d_w) to be added to the interpolated tendency."""
        K = int(coords_q.shape[-1])
        d = geom_q[..., 0:1]                                    # normalized signed wall distance
        n = geom_q[..., 2:2 + K]                                # wall normal (from grad SDF)
        nn = n / (tf.norm(n, axis=-1, keepdims=True) + 1e-6)
        foot = coords_q - d * nn                                # projection onto the wall
        s_w = self._surface_read(slices, foot, roles)            # wall state, native resolution
        ad = tf.abs(d)
        tau = tf.cast(tf.nn.softplus(self.bl_tau) + 1e-3, ad.dtype)   # raw tf.Variable: no autocast
        prof = self.bl_prof(tf.concat([d, ad / tau, tf.math.log1p(ad / tau),
                                       tf.exp(-ad / tau), ad * ad], -1))   # wall-law basis
        prof = tf.cast(prof, s_w.dtype)                          # geometry f32 -> latent dtype
        return self.bl_gate(self.bl_head(s_w[..., :self.d_w] * prof))

    def _surface_kinematics(self, s_w, coords_q, geom_q):
        """Curvature x tangential-flow coupling at the query points (see __init__)."""
        K = int(coords_q.shape[-1])
        gq = tf.cast(geom_q, s_w.dtype)                         # raw f32 input meets latent state
        n = gq[..., 2:2 + K]
        nn = n / (tf.norm(n, axis=-1, keepdims=True) + 1e-6)
        H = gq[..., 5:6]                                        # mean curvature
        G = gq[..., 6:7]                                        # gaussian curvature
        disc = tf.sqrt(tf.maximum(H * H - G, 0.0))
        k1, k2 = H + disc, H - disc                             # principal curvatures
        u = self.sk_vel(s_w)[..., :K]                           # velocity proxy from the wall state
        ut = u - tf.reduce_sum(u * nn, -1, keepdims=True) * nn  # tangential component
        q = tf.reduce_sum(ut * ut, -1, keepdims=True)           # tangential kinetic energy
        feat = tf.concat([H, G, k1, k2, q, k1 * q, k2 * q, H * q], -1)
        return self.sk_gate(self.sk_head(tf.concat([s_w, feat], -1)))

    def feats_from_win(self, win, fmask, geom_node):
        """win (B,T_in,N,S), fmask (B,T_in), geom_node (B,N,GEOM_W) -> (B,N,F) fixed layout
        matching core.train_step.build_feats: [frames time-major per node | frame mask | geom]."""
        B = tf.shape(win)[0]
        N = tf.shape(win)[2]
        x = tf.transpose(win, [0, 2, 1, 3])                              # (B,N,T,S)
        x = tf.reshape(x, [B, N, self.t_in * self.S])
        fm = tf.tile(fmask[:, None, :], [1, N, 1])                       # (B,N,T)
        return tf.concat([x, fm, geom_node], -1)

    def core(self, h, e, roles):
        for i, (cv, at, ml) in enumerate(self.blocks):
            h = cv(h, roles=roles)
            h = at(h, roles=roles)
            h = ml(h)
            h = self.cond.modulate(i, e, h)
        return self.ln_out(h)

    def moe_aux(self):
        """Switch-Transformer load-balance loss, averaged over blocks (0 when MoE is off).

        It was already being computed per block and then never used, and diag_moe.py shows the
        consequence: after r1-r10 the 16 experts still sit 2.2% from their own mean, the top-1 gate
        share is 0.52 (the two picks are averaged, not chosen), and load per expert runs from 0.00
        to 1.00 instead of the uniform 0.125. An unbalanced router is also what makes a
        capacity-based sparse dispatch inexact, so this is the prerequisite for making a higher
        latent resolution affordable rather than a 4x slowdown."""
        if not self.moe_experts:
            return tf.constant(0.0)
        return tf.add_n([b[2].aux for b in self.blocks]) / float(len(self.blocks))

    def trunk_W(self, coords_node, win, fmask, geom_node, K, roles, cond, op_multihot,
                box_dims=None):
        """First half of step(): window -> latent box -> panel tendencies. Returns
        (Wp (B,*dims,d_w,n_p), e) so a caller can EDIT the panels (test-time pull-back
        correction of the detached instance) and re-decode via dec()/interp_box.
        Plain grid path only (no sdf/shape/slices) — the 3D transient buckets."""
        dims = tuple(box_dims) if box_dims else self.box_dims[K]
        feats = self.feats_from_win(win, fmask, geom_node)
        box, geom_box, _ = self.enc(coords_node, feats, dims=dims, roles=roles)
        e = tf.cast(self.cond(cond), box.dtype)
        h = self.core(box, e, roles)
        W = self.banks(h, K=K, roles=roles, op_multihot=op_multihot,
                       geom_box=geom_box, steady=False, e=e)
        Wp = tf.reshape(W, tf.concat([tf.shape(W)[:-1], [self.d_w, self.n_p]], 0))
        return Wp, e

    def step(self, coords_node, win, fmask, geom_node, K, roles, cond, op_multihot,
             coords_q, geom_q, u_prev_q, steady=False, prev_grid=None, grid_dims=None,
             fam_id=None, sdf_box=None, box_dims=None, fbmask=None, surf=False,
             prev2_grid=None, ret_dudt=False):
        """ONE AR step: predict the next frame at coords_q.
        u_prev_q (B,Q,S): previous frame values at the query points (hard anchor).
        prev_grid (B,H,W,S) for dense-2D transport; None elsewhere.
        box_dims overrides the default latent resolution for this call — the core carries no
        dims-shaped weights, so one weight set serves every resolution."""
        dims = tuple(box_dims) if box_dims else self.box_dims[K]
        feats = self.feats_from_win(win, fmask, geom_node)
        box, geom_box, slices = self.enc(coords_node, feats, dims=dims, roles=roles)
        if sdf_box is not None:                                          # dense shape context
            box = box + self.sdf_gate(self.sdf_proj(sdf_box))
            geom_box = tf.concat([tf.cast(sdf_box, geom_box.dtype),          # ch0 := SDF
                                  geom_box[..., 1:]], -1)                    # (feeds BL bank)
        e = tf.cast(self.cond(cond), box.dtype)
        if self.d_shape and sdf_box is not None:                         # shape -> conditioning
            sh = self.sh_c2(self.sh_c1(self.sh_in(tf.cast(sdf_box, box.dtype)), roles=roles),
                            roles=roles)
            sh = tf.reduce_mean(sh, axis=list(range(1, len(dims) + 1)))  # (B,d_shape) global pool
            e = e + tf.cast(self.sh_out(sh), e.dtype)                    # zero-init: warm-safe
        h = self.core(box, e, roles)
        W = self.banks(h, K=K, roles=roles, op_multihot=op_multihot,
                       geom_box=geom_box, steady=steady, e=e)   # (B,*dims,d_w*n_p);
        #                                                 e = equation -> mechanism emphasis
        Wp = tf.reshape(W, tf.concat([tf.shape(W)[:-1], [self.d_w, self.n_p]], 0))
        z = tf.reduce_mean(Wp, axis=-1)                                  # exact panel ADA at t=1
        zq = interp_box(z, coords_q, dims)                               # (B,Q,d_w)
        if surf:
            # FORCE THE MANIFOLD PATH. Two routes reach the query: the volumetric box, which is
            # 97% empty for a surface and needs a per-shape embed/project pair learned on top,
            # and the slice tokens, which live on the manifold and need neither. The box wins
            # anyway because 20 grid families train it while only 6 mesh families train the
            # slices — measured, the surface gates stayed shut (sf_out 0.30, bl_gate 0.117,
            # sk_gate 0.081 against a box boundary-layer gate of 1.91). Cutting the box outright
            # would collapse these families to persistence, since zq is also how the physics
            # banks reach the decoder. Stopping the GRADIENT instead keeps the forward pass
            # intact and sends every bit of these families' learning signal into the slice path,
            # which is the only route that can actually represent a car surface.
            zq = tf.stop_gradient(zq)
        if slices is not None:                                           # surface-manifold path
            s_q = self._surface_read(slices, coords_q, roles, state=zq)
            zq = zq + self.sf_out(s_q)
            zq = zq + self._bl_bank(slices, coords_q, geom_q, roles, e)   # native-res BL tendency
            zq = zq + self._surface_kinematics(s_q, coords_q, geom_q)     # curvature x flow
        # ITEM G: the native anchor dictionary at the query points. Requires a native grid for
        # the anchor, so it is active exactly on the dense grid buckets; point/mesh buckets keep the
        # r19 decode (dfeat=None -> the zero-init coefficient head contributes nothing).
        dfeat = None
        if self.n_dict and prev_grid is not None and grid_dims is not None:
            ns = self.n_semantic
            F = anchor_features(prev_grid[..., :ns],
                                None if prev2_grid is None else prev2_grid[..., :ns],
                                K=len(grid_dims), shift=self.dict_shift)  # (B,*grid,ns,J)
            F = tf.cast(F, zq.dtype)
            ncell = int(np.prod(grid_dims))
            Ff = tf.reshape(F, [tf.shape(F)[0], ncell, ns * self.n_dict])
            nq = coords_q.shape[1]
            if nq is not None and nq >= ncell:
                # On a dense bucket the first `ncell` queries ARE the cells, in the same C-order as
                # the grid (data_ar builds coords_node = _cell_coords(dims) with ni = arange(N)), so
                # interpolating them would be an identity at 4x the gather traffic. Only the
                # collocation tail needs it.
                tail = (interp_box(Ff, coords_q[:, ncell:], tuple(grid_dims))
                        if nq > ncell else None)
                dq = Ff if tail is None else tf.concat([Ff, tail], 1)
            else:
                dq = interp_box(Ff, coords_q, tuple(grid_dims))
            dfeat = tf.reshape(dq, [tf.shape(dq)[0], -1, self.n_semantic, self.n_dict])
        dudt = None
        if ret_dudt:
            # ANALYTIC RATE AT AN INTERIOR COLLOCATION TIME (Fig2 arm-2 v3). Data labels exist
            # only at integer frames; the physics residual is therefore evaluated at a RANDOM
            # panel-interior time tau in (0,1] where no label exists, so it can never compete
            # with the data loss (PINN collocation in time). ADA makes this analytic: z(tau) is
            # the partial panel mean and dz/dt(tau) = W at that panel exactly. du/dt then comes
            # from one forward-mode JVP through the small decoder (coord/geom/cond branches are
            # constant in t). Plain grid path only (surf=False, slices=None) -- 3D buckets.
            m = tf.random.uniform([], minval=self.n_p // 2, maxval=self.n_p, dtype=tf.int32)
            cum = tf.cumsum(Wp, axis=-1) / float(self.n_p)     # z at panel ends
            z_tau = tf.gather(cum, m, axis=-1)                 # (B,*dims,d_w) at tau=m/n_p
            w_tau = tf.gather(Wp, m, axis=-1)
            zq_tau = tf.cast(interp_box(z_tau, coords_q, dims), zq.dtype)
            wq_tau = tf.cast(interp_box(w_tau, coords_q, dims), zq.dtype)
            with tf.autodiff.ForwardAccumulator(zq_tau, wq_tau) as fwd_acc:
                delta_tau = self.dec(zq_tau, coords=coords_q, roles=roles, geom=geom_q,
                                     cond=e, fam_id=fam_id, dfeat=dfeat)   # (B,Q,S) at tau
            dudt = fwd_acc.jvp(delta_tau)
            delta = delta_tau                                  # fields AT tau (for the flux)
            if fbmask is not None:
                dudt = dudt * tf.cast(fbmask, dudt.dtype)[:, None, :]
        else:
            delta = self.dec(zq, coords=coords_q, roles=roles, geom=geom_q,
                             cond=e, fam_id=fam_id, dfeat=dfeat)         # (B,Q,S)
        # The step BOUNDARY is float32: this value is the AR feedback, the hard IC anchor and the
        # loss input, and under a mixed policy accumulating 10 segments in bfloat16 would let the
        # rollout drift on rounding alone. Only the interior runs reduced.
        # fbmask decides which channels may carry state back into this family's own window; see
        # v3/data_ar.py for why it is neither all-ones nor cmask. Masking everything the family
        # does not output costs more than it saves (Poisson-Gauss 4.6 -> 35.0 on r10's own
        # weights): the model uses its unused channels as latent scratch state.
        d = tf.cast(delta, u_prev_q.dtype)
        if fbmask is not None:
            d = d * tf.cast(fbmask, d.dtype)[:, None, :]
        # HARD IC ANCHORING APPLIES TO PHYSICAL SLOTS ONLY. u_next = u_prev + delta is what keeps
        # a predicted FIELD tied to its initial condition, and it is the whole reason the rollout
        # does not wander. Applied to the memory channels it means something else entirely: the
        # state accumulates as a running sum over segments and is free to drift, because nothing
        # scores it. Measured over r15's first 4500 steps that is exactly what happened — the four
        # families that depend on memory swung instead of converging (Poisson-Gauss 4.6 -> 9.4 ->
        # 4.3 -> 7.7, Wave-Layer 27.3 <-> 34.1) while the other 22 improved or held. Memory is now
        # WRITTEN FRESH each segment; the window still carries the last t_in of those writes, so
        # it is still recurrent state, just a bounded one.
        if self.n_semantic < self.S:
            keep = tf.concat([tf.ones([self.n_semantic], d.dtype),
                              tf.zeros([self.S - self.n_semantic], d.dtype)], 0)
            u_prev_q = u_prev_q * keep[None, None, :]
        u_next = u_prev_q + d
        if self.transport and prev_grid is not None and K == 2:
            db = self.disp(h)                                            # (B,*dims,2) zero-init
            dq = tf.cast(interp_box(db, coords_q, dims), coords_q.dtype)
            warped = interp_box(prev_grid, coords_q - dq, grid_dims)     # backward warp sample
            g = tf.cast(tf.tanh(self.tgate(e))[:, None, :], u_next.dtype)   # (B,1,S) zero-init
            g = g * op_multihot[:, None, ADV_OP:ADV_OP + 1]              # advection families only
            u_next = u_next + g * (tf.cast(warped, u_next.dtype) - u_prev_q)
        if ret_dudt:
            return u_next, tf.cast(dudt, u_next.dtype)
        return u_next
