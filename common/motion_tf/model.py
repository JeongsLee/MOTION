"""Physics-Operator Mixture foundation operator (DESIGN.md §5).

    dz/dt|_i = Π_{BC,Ω}[ Σ_k α_k(family) · B_k(z_i, ctx) ]      (learned operator splitting)

Pipeline (2-D, scalar field; multi-channel/3-D are dimensional lifts, §4.6):

  1. Shared geometry/coord feature lift   (x, y, metric[, SDF]) → h_geom        (once)
  2. Family gate  α_k = FamilyGate(descriptor)                                  (once)
  3. Multi-branch g-feedback rollout over N_p panels:
        z_i (cumulative) → each expert B_k → gated sum W_i → z_{i+1}=z_i+W_i·dt
  4. Shared ADA basis integrates {W_i} → continuous trajectory, hard IC.

Encoder consumes explicit (x, y) + metric and uses relational conv experts — never a
pointwise coord→W INR (DESIGN.md §4.5). Grid is structured (possibly stretched); the
metric channels carry the spacing (§4.6).
"""
from __future__ import annotations
import numpy as np
import tensorflow as tf

from .basis.adaf_basis import FourierPanelBasis
from .basis.lpa_basis import LegendrePanelBasis
from .experts.base import periodic_pad_2d, d_dx, d_dy, laplacian, minmod_dx, minmod_dy
from .experts.convective import ConvectiveExpert
from .experts.diffusive import DiffusiveExpert
from .experts.reaction import ReactionExpert
from .experts.elliptic import EllipticExpert
from .experts.shock import ShockExpert
from .experts.helmholtz import HelmholtzExpert
from .experts.wave import WaveExpert
from .experts.forcing import ForcingExpert
from .experts.unified import UnifiedExpert
from .conditioner import FamilyGate, FieldGate


_EXPERT_REGISTRY = {
    "convective": ConvectiveExpert,
    "diffusive": DiffusiveExpert,
    "reaction": ReactionExpert,
    "elliptic": EllipticExpert,
    "shock": ShockExpert,
    "helmholtz": HelmholtzExpert,
    "wave": WaveExpert,
    "forcing": ForcingExpert,
    "unified": UnifiedExpert,
}


class AttnRouter(tf.keras.layers.Layer):
    """Transformer (patch-attention) router — the VERIFIED data→weights mapping (DESIGN.md §router).

    Replaces the 1×1-conv sigmoid router. Rationale: in a multi-PDE model the physics inductive bias
    belongs in the *experts* (operator computation), but the *routing* (how input information flows
    into the mixture) needs a well-conditioned, gradient-stable, verified architecture — which the
    field has converged on = transformers (PROSE-FD; Poseidon/scOT-Swin). So the router is a small
    transformer; the experts stay physics-aware.

    h_geom (B,H,W,d_geom) → patch-embed → tokens → L×[LN·MHSA·LN·MLP, residual] → per-token expert
    logits → bilinear-upsample to the router grid → **softmax over experts**. Softmax makes Σ_k route_k
    = 1 (a convex mix), which structurally cures the independent-sigmoid divergence (all experts→1 →
    ΣW K× too big → NaN) and damps the elliptic branch-collapse. Equation-label-free → transfer-friendly.
    `patch` must divide H,W (e.g. patch=8 on 128², patch=2 on a 32² latent grid → 16×16=256 tokens)."""
    def __init__(self, n_experts, d_model=192, depth=2, heads=4, patch=8, mlp_ratio=2,
                 route_act="softmax", encode_dim=0, name="attn_router", **kw):
        super().__init__(name=name, **kw)
        # encode_dim>0 → ENCODER mode: emit per-pixel FEATURES (no expert softmax). Used by the unified
        # single-expert design where the transformer is a feature encoder, not a router (avoids the
        # observed route collapse-to-one-expert). encode_dim=0 → legacy K-expert softmax routing.
        self.encode_dim = int(encode_dim)
        self.K = int(n_experts); self.d_model = int(d_model); self.patch = int(patch)
        self.depth = int(depth); self.heads = int(heads); self.mlp_ratio = int(mlp_ratio)
        # routing activation: 'softmax' (Σ_k=1, convex competition — bounds ΣW) OR 'sigmoid' (independent
        # 0-1 per expert, NO sum constraint → overlapping physical experts can co-activate fully; ΣW kept
        # bounded by neg-bias init + zero-init experts + W_scale + clipnorm instead of the sum=1 device).
        self.route_act = str(route_act)
        self.patch_embed = tf.keras.layers.Conv2D(d_model, self.patch, strides=self.patch,
                                                  padding="valid", name="patch_embed")
        self.blocks = []
        for i in range(self.depth):
            self.blocks.append(dict(
                ln1=tf.keras.layers.LayerNormalization(epsilon=1e-5, name=f"ln1_{i}"),
                mha=tf.keras.layers.MultiHeadAttention(num_heads=self.heads,
                                                       key_dim=max(1, d_model // self.heads),
                                                       name=f"mha_{i}"),
                ln2=tf.keras.layers.LayerNormalization(epsilon=1e-5, name=f"ln2_{i}"),
                fc1=tf.keras.layers.Dense(d_model * self.mlp_ratio, activation="gelu", name=f"fc1_{i}"),
                fc2=tf.keras.layers.Dense(d_model, name=f"fc2_{i}"),
            ))
        # sigmoid mode: neg-bias init → routes start ≈0.12 (proven router-v2 stabilizer; bias=0 diverged)
        hbias = tf.keras.initializers.Constant(-2.0) if self.route_act == "sigmoid" else "zeros"
        self.head = tf.keras.layers.Dense(self.encode_dim or self.K,
                                          bias_initializer=("zeros" if self.encode_dim else hbias),
                                          name="route_head")

    def build(self, input_shape):
        H, W = int(input_shape[1]), int(input_shape[2])
        self.out_h, self.out_w = H, W
        self.gh, self.gw = H // self.patch, W // self.patch
        self.pos = self.add_weight(name="pos", shape=(1, self.gh * self.gw, self.d_model),
                                   initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, h, bias=None):                              # (B,H,W,d_geom) → (B,H,W,K), Σ_K=1
        x = self.patch_embed(h)                                # (B,gh,gw,d_model)
        B = tf.shape(x)[0]
        x = tf.reshape(x, [B, self.gh * self.gw, self.d_model]) + tf.cast(self.pos, x.dtype)
        for blk in self.blocks:
            y = blk["ln1"](x)
            x = x + blk["mha"](y, y)                            # self-attention (global over tokens)
            x = x + blk["fc2"](blk["fc1"](blk["ln2"](x)))       # MLP
        logits = tf.reshape(self.head(x), [B, self.gh, self.gw, self.encode_dim or self.K])
        logits = tf.image.resize(logits, [self.out_h, self.out_w], method="bilinear")
        if self.encode_dim:
            return logits                                      # ENCODER mode: per-pixel features (no softmax)
        if bias is not None:                                   # descriptor → per-family per-expert logit PRIOR
            logits = logits + tf.cast(bias, logits.dtype)[:, None, None, :]   # added DIRECTLY to logits (not drowned)
        if self.route_act == "sigmoid":
            return tf.sigmoid(logits)                          # independent per-expert (no sum=1)
        return tf.nn.softmax(logits, axis=-1)                  # convex per-pixel mix over experts


class RouteFiLM(tf.keras.layers.Layer):
    """FiLM conditioning of the expert input by the routing distribution (Poseidon adaLN / Unisolver
    deep-conditions style; DESIGN.md §11 / refs/PDE-FM_architectures.md). Instead of ADDING a route
    embedding, modulate h_geom by (γ,β) derived from HOW the router allocated across all experts:
    domain-wise (global-pooled route → global γ,β) + point-wise (per-pixel route → spatial γ,β). Lets a
    shared expert disambiguate the family-regime it serves and specialize. zero-init → identity at start
    (preserves the validated stable start), learns the modulation."""
    def __init__(self, d_geom, name="route_film", **kw):
        super().__init__(name=name, **kw)
        self.d = int(d_geom)
        self.point = tf.keras.layers.Conv2D(2 * self.d, 1, kernel_initializer="zeros", name="film_point")
        self.domain = tf.keras.layers.Dense(2 * self.d, kernel_initializer="zeros", name="film_domain")

    def call(self, route, hg):                                 # route,hg cast to compute dtype upstream
        gp, bp = tf.split(self.point(route), 2, axis=-1)       # point-wise (B,h,w,d)
        pooled = tf.reduce_mean(route, axis=[1, 2])            # (B,K) domain-wise pool
        gd, bd = tf.split(self.domain(pooled), 2, axis=-1)     # (B,d)
        gamma = gp + gd[:, None, None, :]
        beta = bp + bd[:, None, None, :]
        return (1.0 + gamma) * hg + beta                       # identity when γ=β=0


class RegimeFiLM(tf.keras.layers.Layer):
    """FiLM conditioning of h_geom by the FIELD REGIME (amplitude / length-scale stats from
    _regime_features), so ONE shared expert+decoder can MODULATE its behaviour per regime — e.g. predict a
    VARYING density for compressible (ρ has high std) vs a CONSTANT density for incompressible (ρ≈const) —
    WITHOUT a router (RouteFiLM is inert in the unified config). The dummy-supervision teaches 4 families
    ρ=const and 1 (com_ns) ρ=varying; a shared decoder collapses to the majority → com_ns regresses. A
    per-sample (domain) γ,β derived from the input regime lets the decoder split by regime instead. zero-init
    final → identity at start (preserves the stable start), learns the modulation."""
    def __init__(self, d_geom, name="regime_film", **kw):
        super().__init__(name=name, **kw)
        self.d = int(d_geom)
        self.h = tf.keras.layers.Dense(self.d, activation="gelu", name="regime_film_h")
        self.out = tf.keras.layers.Dense(2 * self.d, kernel_initializer="zeros", name="regime_film_out")

    def call(self, regime, hg):                                # regime:(B,4C) field stats, hg:(B,H,W,d)
        g, b = tf.split(self.out(self.h(regime)), 2, axis=-1)  # (B,d) per-sample γ,β
        return (1.0 + g[:, None, None, :]) * hg + b[:, None, None, :]   # identity when out=0


class GridMP(tf.keras.layers.Layer):
    """Grid message passing for the ROUTER (DESIGN.md §11): GraphSAGE-style neighbor aggregation on the
    regular grid (periodic, multi-dilation), enriching per-pixel features with multi-scale local
    connectivity BEFORE the routing decision. Motivation: even on a homogeneous periodic domain the
    local regime (Re ∝ amplitude·length-scale, flow direction) varies in space; conv is translation-
    invariant (one stencil everywhere) but multi-dilation neighbor aggregation gives the router a
    multi-scale length-scale/Re proxy per location → regime-aware routing. SAGE form (separate self +
    neighbor transforms) over fixed grid-graph edges. residual + zero-init out → identity at start."""
    def __init__(self, d, depth=2, dilations=(1, 2, 4), name="grid_mp", **kw):
        super().__init__(name=name, **kw)
        self.dils = tuple(dilations)
        self.blocks = []
        for i in range(int(depth)):
            self.blocks.append(dict(
                ln=tf.keras.layers.LayerNormalization(epsilon=1e-5, name=f"mp_ln_{i}"),
                self_t=tf.keras.layers.Conv2D(d, 1, name=f"mp_self_{i}"),
                neigh_t=tf.keras.layers.Conv2D(d, 1, name=f"mp_neigh_{i}"),
                out=tf.keras.layers.Conv2D(d, 1, kernel_initializer="zeros", name=f"mp_out_{i}"),
            ))

    def _agg(self, h):                                          # periodic multi-dilation 4-neighbor mean
        accs = []
        for d in self.dils:
            n = (tf.roll(h, d, 1) + tf.roll(h, -d, 1) + tf.roll(h, d, 2) + tf.roll(h, -d, 2)) * 0.25
            accs.append(n)
        return tf.add_n(accs) / float(len(accs))

    def call(self, h):
        for b in self.blocks:
            x = b["ln"](h)
            m = b["out"](tf.nn.gelu(b["self_t"](x) + b["neigh_t"](self._agg(x))))  # SAGE: self + neighbor
            h = h + m                                           # residual (zero-init out → identity start)
        return h


class PanelAttn(tf.keras.layers.Layer):
    """Bidirectional self-attention over the N_p PANEL axis, applied PER SPATIAL LOCATION, to couple the
    independently-computed panel tendencies W (pw_batched path). Replaces the SEQUENTIAL W-recursion with a
    SINGLE parallel pass: every panel's W is conditioned on all other panels' W at the same pixel (mutual
    self-consistency, bidirectional since W is a latent — no teacher-forcing / causal order needed). A learned
    panel-positional embedding gives the attention the panel-index/time ordering. Zero-init output proj →
    starts as identity (== pw_batched), learns the inter-panel coupling. Batchable → keeps the pw_batched speed."""

    def __init__(self, dim, n_panels, heads=4, d_attn=0, layers=1, mlp_ratio=2, name="panel_attn", **kw):
        super().__init__(name=name, **kw)
        self.dim = int(dim); self.Np = int(n_panels); self.heads = int(heads); self.L = int(layers)
        self.da = int(d_attn) or int(dim)
        if self.da % self.heads != 0:                            # round up to a head-divisible attn width
            self.da = self.heads * ((self.da + self.heads - 1) // self.heads)
        self.dh = self.da // self.heads
        # project the oc-wide W to the (wider) attention width once; project back at the end (zero-init).
        self.in_proj = tf.keras.layers.Dense(self.da, name="pa_in") if self.da != self.dim else None
        self.pos = self.add_weight(name="pa_pos", shape=(self.Np, self.da),
                                   initializer="zeros", trainable=True)                    # panel positional
        self.blocks = []                                         # L full transformer blocks (attn + FFN)
        for i in range(self.L):
            self.blocks.append(dict(
                ln1=tf.keras.layers.LayerNormalization(epsilon=1e-5, name=f"pa_ln1_{i}"),
                q=tf.keras.layers.Dense(self.da, name=f"pa_q_{i}"),
                k=tf.keras.layers.Dense(self.da, name=f"pa_k_{i}"),
                v=tf.keras.layers.Dense(self.da, name=f"pa_v_{i}"),
                po=tf.keras.layers.Dense(self.da, name=f"pa_po_{i}"),               # attn output proj
                ln2=tf.keras.layers.LayerNormalization(epsilon=1e-5, name=f"pa_ln2_{i}"),
                fc1=tf.keras.layers.Dense(self.da * int(mlp_ratio), activation="gelu", name=f"pa_fc1_{i}"),
                fc2=tf.keras.layers.Dense(self.da, name=f"pa_fc2_{i}")))
        self.o = tf.keras.layers.Dense(self.dim, kernel_initializer="zeros", name="pa_o")  # zero-init → identity

    def _mha(self, blk, h):                                      # self-attn over the Np panel axis
        q = blk["q"](h) + tf.cast(self.pos[None], h.dtype)
        k = blk["k"](h) + tf.cast(self.pos[None], h.dtype)
        v = blk["v"](h)

        def split(t):                                            # (.,Np,da) → (.,heads,Np,dh)
            t = tf.reshape(t, [tf.shape(t)[0], self.Np, self.heads, self.dh])
            return tf.transpose(t, [0, 2, 1, 3])
        qh, kh, vh = split(q), split(k), split(v)
        att = tf.matmul(tf.cast(qh, tf.float32), tf.cast(kh, tf.float32), transpose_b=True) * (self.dh ** -0.5)
        att = tf.nn.softmax(att, axis=-1)                        # fp32 softmax (stability)
        oh = tf.matmul(tf.cast(att, vh.dtype), vh)               # (.,heads,Np,dh)
        o = tf.reshape(tf.transpose(oh, [0, 2, 1, 3]), [tf.shape(q)[0], self.Np, self.da])
        return blk["po"](o)

    def call(self, x):                                          # x: (B, Np, H, W, dim) → same
        sh = tf.shape(x); B, H, W = sh[0], sh[2], sh[3]
        xt = tf.transpose(x, [0, 2, 3, 1, 4])                   # (B,H,W,Np,dim)
        seq = tf.reshape(xt, [B * H * W, self.Np, self.dim])    # per-pixel sequence over panels
        h = self.in_proj(seq) if self.in_proj is not None else seq
        for blk in self.blocks:                                 # pre-LN transformer blocks (attn + FFN)
            h = h + self._mha(blk, blk["ln1"](h))
            h = h + blk["fc2"](blk["fc1"](blk["ln2"](h)))
        o = self.o(h)                                           # (.,Np,dim) zero-init → identity at start
        o = tf.reshape(o, [B, H, W, self.Np, self.dim])
        return x + tf.transpose(o, [0, 3, 1, 2, 4])             # (B,Np,H,W,dim) residual


def _bilinear_warp(img, flow):
    """Backward bilinear warp with PERIODIC wrap. img (N,H,W,C), flow (N,H,W,2)=(dx,dy) in CELL units;
    out[n,i,j] = img[n] sampled at (i+dy, j+dx). Periodic (pdearena/NS domains are periodic). fp32 inside."""
    img = tf.cast(img, tf.float32)
    N = tf.shape(img)[0]; H = tf.shape(img)[1]; Wd = tf.shape(img)[2]
    Hf = tf.cast(H, tf.float32); Wf = tf.cast(Wd, tf.float32)
    gy, gx = tf.meshgrid(tf.range(H), tf.range(Wd), indexing="ij")     # (H,W)
    gx = tf.cast(gx, tf.float32)[None]; gy = tf.cast(gy, tf.float32)[None]   # (1,H,W)
    sx = tf.math.floormod(gx + flow[..., 0], Wf)                       # (N,H,W) periodic
    sy = tf.math.floormod(gy + flow[..., 1], Hf)
    x0 = tf.floor(sx); y0 = tf.floor(sy)
    wx = (sx - x0)[..., None]; wy = (sy - y0)[..., None]
    x0i = tf.cast(x0, tf.int32) % Wd; x1i = (x0i + 1) % Wd
    y0i = tf.cast(y0, tf.int32) % H; y1i = (y0i + 1) % H
    bnd = tf.reshape(tf.range(N), [-1, 1, 1]) * tf.ones_like(x0i)      # (N,H,W)
    g = lambda yy, xx: tf.gather_nd(img, tf.stack([bnd, yy, xx], -1))  # (N,H,W,C)
    top = g(y0i, x0i) * (1 - wx) + g(y0i, x1i) * wx
    bot = g(y1i, x0i) * (1 - wx) + g(y1i, x1i) * wx
    return top * (1 - wy) + bot * wy


class AdvPanelCouple(tf.keras.layers.Layer):
    """ADVECTIVE coupling of ADA W-panels (turbulence lever, 2026-07-18). Panel k's tendency W_k(x)
    receives the PREVIOUS panel W_{k-1} sampled at the backward-advected departure point x - u·t_k
    (semi-Lagrangian; u = IC velocity, state slots 0,1). Injects the advective space-time correlation a
    SEPARABLE ADA basis lacks (pdearena turbulence). Unlike AR there is NO prediction feedback → no error
    accumulation. mix conv is ZERO-INIT → exact no-op at load (warm-safe); scale init small-nonzero so
    d(loss)/d(mix)=warped ≠ 0 (live gradient, avoids the both-zero dead saddle)."""

    def __init__(self, ch, n_panels, name="adv_panel", **kw):
        super().__init__(name=name, **kw)
        self.Np = int(n_panels)
        self.tk = [float(k) / float(max(1, self.Np - 1)) for k in range(self.Np)]  # normalized panel times
        self.mix = tf.keras.layers.Conv2D(ch, 1, kernel_initializer="zeros", name="apc_mix")
        self.scale = self.add_weight(name="apc_scale", shape=(), trainable=True,
                                     initializer=tf.constant_initializer(0.25))    # velocity → cell displacement

    def call(self, w5, u):
        # w5 (B,Np,H,W,C) panel tendencies; u (B,H,W,2) IC velocity. Returns (B,Np,H,W,C).
        dt = w5.dtype
        H = int(w5.shape[2]); Wd = int(w5.shape[3]); C = int(w5.shape[4])
        prev = tf.reshape(w5[:, :-1], [-1, H, Wd, C])                  # (B*(Np-1),H,W,C) panels 0..Np-2
        tkv = tf.constant(self.tk[1:], tf.float32)                     # (Np-1,) times of panels 1..Np-1
        u32 = tf.cast(u, tf.float32)[:, None]                          # (B,1,H,W,2)
        flow = -u32 * self.scale * tkv[None, :, None, None, None]      # (B,Np-1,H,W,2)
        warped = _bilinear_warp(prev, tf.reshape(flow, [-1, H, Wd, 2]))  # (B*(Np-1),H,W,C)
        mixed = self.mix(tf.cast(warped, dt))                         # zero-init → 0 at load
        mixed = tf.reshape(mixed, [tf.shape(w5)[0], self.Np - 1, H, Wd, C])
        coupled = w5[:, 1:] + tf.cast(mixed, dt)
        return tf.concat([w5[:, :1], coupled], axis=1)


class WindowedTCU(tf.keras.layers.Layer):
    """Windowed temporal/coarse cross-attention UPSAMPLE (replaces bilinear). Each output sub-pixel is a
    softmax-attention over a small win x win window of the coarse state S (kv keys) applied to the current
    panel W (values). S is the coarse/previous-timestep latent (in the recursive rollout zc is the evolved
    state) -> fine detail is synthesized by attending to the local coarse neighborhood (short-time, local:
    small-scale features have short correlation time + local spatial correlation). Periodic padding. W is
    ~0 at start (expert zero-init) -> output ~0 -> stable start regardless of attention init."""
    def __init__(self, oc, LF, win=3, d=32, name="wtcu", **kw):
        super().__init__(name=name, **kw)
        self.oc=int(oc); self.LF=int(LF); self.win=int(win); self.d=int(d)
        self.kproj=tf.keras.layers.Conv2D(self.d, 1, name="k")
        self.qproj=tf.keras.layers.Conv2D(self.d*self.LF*self.LF, 1, name="q")

    def call(self, W, S):
        cd=W.dtype; win=self.win; LF=self.LF; d=self.d; oc=self.oc; p=win//2
        Wf=tf.cast(W, tf.float32); Sf=tf.cast(S, tf.float32)
        B=tf.shape(Wf)[0]; H=Wf.shape[1]; Wd=Wf.shape[2]
        k=tf.cast(self.kproj(Sf), tf.float32); q=tf.cast(self.qproj(Wf), tf.float32)   # fp32 attention (softmax stable)
        kpad=periodic_pad_2d(k, p); vpad=periodic_pad_2d(Wf, p)              # periodic borders
        pk=tf.image.extract_patches(kpad,[1,win,win,1],[1,1,1,1],[1,1,1,1],'VALID')   # (B,H,W,win*win*d)
        pv=tf.image.extract_patches(vpad,[1,win,win,1],[1,1,1,1],[1,1,1,1],'VALID')   # (B,H,W,win*win*oc)
        pk=tf.reshape(pk,[B,H,Wd,win*win,d])
        pv=tf.reshape(pv,[B,H,Wd,win*win,oc])
        q=tf.reshape(q,[B,H,Wd,LF*LF,d])
        att=tf.einsum('bhwsd,bhwnd->bhwsn', q, pk) * (1.0/(d**0.5))          # (B,H,W,LF²,win²)
        att=tf.nn.softmax(att, axis=-1)
        out=tf.einsum('bhwsn,bhwno->bhwso', att, pv)                        # (B,H,W,LF²,oc)
        out=tf.reshape(out,[B,H,Wd,LF,LF,oc])
        out=tf.transpose(out,[0,1,3,2,4,5])                                 # (B,H,LF,W,LF,oc)
        out=tf.reshape(out,[B,H*LF,Wd*LF,oc])
        return tf.cast(out, cd)


class PhysicsOperatorMixture(tf.Module):
    def __init__(self, cfg, name="pomix"):
        super().__init__(name=name)
        self.cfg = cfg
        H = W = cfg.Nx

        # --- shared ADA temporal basis (Fourier + Legendre, trainable mix) ---
        self.basis_adaf = FourierPanelBasis(T_seg=cfg.T_final, Nt=cfg.Nt, N_p=cfg.N_p,
                                            n_modes=cfg.n_modes, n_integrations=1,
                                            return_derivative=False, dtype=tf.float32)
        self.basis_lpa = LegendrePanelBasis(T_seg=cfg.T_final, Nt=cfg.Nt, N_p=cfg.N_p,
                                            max_order=cfg.max_order, n_integrations=1,
                                            return_derivative=False, gamma=1.0,
                                            dtype=tf.float32)
        self.mix_logit = tf.Variable(0.0, trainable=True, name="mix_logit")
        self.W_scale = tf.Variable(float(getattr(cfg, "w_scale_init", 0.05)),
                                   trainable=True, name="W_scale")  # gentler start ↓ → less early overshoot

        # --- shared geometry/coord feature lift (relational, §4.5) ---
        d_geom = cfg.d_geom
        n_geo = int(getattr(cfg, "geo_layers", 2))
        self.geo = [tf.keras.layers.Conv2D(d_geom, 3, padding="valid",
                                           activation="gelu", name=f"geo{i}")
                    for i in range(n_geo)]

        # --- experts (selectable for ablation; multi-channel via out_ch) ---
        self.n_channels = int(getattr(cfg, "n_channels", 1))
        self.expert_names = list(cfg.experts)
        # split_basis: experts emit 2·C panel values — first C drive the Fourier (cos/sin) basis,
        # last C the Legendre basis; each basis integrates its OWN W (no longer shared). A learned
        # per-pixel/per-channel map then decides WHERE to trust cos/sin vs Legendre. (False = legacy
        # shared-W + single scalar mix.)
        self.split_basis = bool(getattr(cfg, "split_basis", True))
        # latent_decoder (NTO-ADA HiLatent): represent the dynamics in D LATENT channels (not the C
        # physical channels). Experts output per-pixel latent W (D), the basis integrates the latent W
        # with ic=0 (residual), and a per-pixel NO-BIAS MLP decodes the latent trajectory → C physical δ;
        # field = IC + δ. no-bias → decode(0)=0 → field(0)=IC exact (hard-IC preserved). Decouples the
        # integration dimensionality from physical channels (more room for the dynamics) and — proven in
        # NTO-ADA NS — sharply improves generalization (ood_k 36.7→17.5%). Forces split_basis off (single
        # dual-basis on the latent W). The physics experts/g-feedback rollout are otherwise unchanged,
        # operating on the D-channel latent state (z0 = a learned 1×1 encode of the physical IC).
        self.latent_decoder = bool(getattr(cfg, "latent_decoder", False))
        self.n_latent = int(getattr(cfg, "n_latent", 16))
        if self.latent_decoder:
            self.split_basis = False
            out_per = self.n_latent
        else:
            out_per = (2 * self.n_channels) if self.split_basis else self.n_channels
        # learned_upsample: replace the bilinear (low-pass) coarse->full resize of the per-panel W with
        # bilinear + a LEARNED high-freq residual (sub-pixel conv, zero-init → starts == bilinear, learns
        # to synthesize high-freq). Gated by cfg (default False = bilinear, existing behavior unchanged).
        self.learned_upsample = bool(getattr(cfg, "learned_upsample", False))
        # pure_learned_upsample: DROP the bilinear base entirely → the coarse→full W upsample is a SINGLE
        # learned sub-pixel conv (depth_to_space), so the low-pass bilinear ceiling that smears high-freq
        # (hurts turbulent PDEArena) is removed. NON-zero (glorot) init: there is no bilinear base to anchor
        # the start, so w_up must produce signal from step 0; stability comes from W_scale (≈0.05, W is small
        # pre-upsample) + the residual decode (field = IC + δ → W≈small ⇒ field≈IC start). Overrides the
        # bilinear+residual mode below. Gated; default OFF.
        self.pure_upsample = bool(getattr(cfg, "pure_learned_upsample", False))
        # buoyancy_bank (by2): time-consistent Boussinesq source bank. Fixes the dormant unified_buoyancy
        # branch's three gaps: (1) per-trajectory magnitude beta = 1 + zero-init Dense(GAP(h_geom)) -- the
        # conditioned set VARIES the buoyancy coefficient (ForcingExpert-style global readout; 1+zero avoids
        # the both-zero saddle); (2) BOTH axis gradients of the scalar (vertical-axis convention insurance);
        # (3) source evaluated on the TRANSPORTED scalar c(tau) (adv_bank backward characteristics reused)
        # instead of the stale anchored IC. Residual = beta * tau * head([c_t, dr c_t, dc c_t]) -> vel slots.
        self.buoyancy_bank = bool(getattr(cfg, "buoyancy_bank", False))
        if self.buoyancy_bank:
            self.bb_h = tf.keras.layers.Conv2D(int(getattr(cfg, "buoyancy_bank_hidden", 32)), 1,
                                               activation="gelu", name="bb_h")
            self.bb_out = tf.keras.layers.Conv2D(2, 1, kernel_initializer="zeros", name="bb_out")
            self.bb_coef = tf.keras.layers.Dense(1, kernel_initializer="zeros", name="bb_coef")
        # lag_channel: forward-Lagrangian particle transport bank (v1). Particles seeded on a stride-S grid
        # carry the passive-scalar slot along basis-represented trajectories Delta(tau) = alpha*u0*tau +
        # sum_m A_m Phi_m(tau) (all Phi(0)=0 -> hard-IC); differentiable normalized Gaussian splat back to
        # the grid; per-family descriptor gate (small const init -> live grads, ~identity at start).
        self.lag_channel = bool(getattr(cfg, "lag_channel", False))
        if self.lag_channel:
            self.lag_stride = int(getattr(cfg, "lag_stride", 2))
            self.lag_modes = int(getattr(cfg, "lag_modes", 6))
            self.lag_sigma = float(getattr(cfg, "lag_sigma", 1.2))
            self.lag_native = bool(getattr(cfg, "lag_native", False))   # slot-2 output = splat (not residual)
            # lag_gate_mode: "desc" = B-I gate (descriptor pattern x temporal variance over the input
            # window); "slots" = IVP/transfer gate (slot-aliveness from the IC; valid at T_in=1). See call().
            self.lag_gate_mode = str(getattr(cfg, "lag_gate_mode", "desc"))
            self.lag_h1 = tf.keras.layers.Conv2D(64, 1, activation="gelu", name="lag_h1")
            self.lag_Ad = tf.keras.layers.Conv2D(2 * self.lag_modes, 1, kernel_initializer="zeros", name="lag_Ad")
            self.lag_Ac = tf.keras.layers.Conv2D(self.lag_modes, 1, kernel_initializer="zeros", name="lag_Ac")
            self.lag_gate = tf.keras.layers.Dense(1, use_bias=False,
                kernel_initializer=tf.keras.initializers.Constant(0.02), name="lag_gate")
            self.lag_alpha = tf.Variable(1.0, dtype=tf.float32, name="lag_alpha")
            # lag_omega (v2): particles also carry vorticity w (materially conserved in 2D incompressible
            # flow); splat w(tau) with the SAME trajectories/weights, invert by FFT-Poisson (periodic 128^2,
            # exact, differentiable: lap psi = -w, u = (d psi/dy, -d psi/dx)) -> Lagrangian VELOCITY residual.
            self.lag_omega = bool(getattr(cfg, "lag_omega", False))
            if self.lag_omega:
                self.lag_Aw = tf.keras.layers.Conv2D(self.lag_modes, 1, kernel_initializer="zeros", name="lag_Aw")
                self.lag_gate_w = tf.keras.layers.Dense(1, use_bias=False,
                    kernel_initializer=tf.keras.initializers.Constant(0.02), name="lag_gate_w")
        # tcu_upsample: windowed cross-attention upsample (kv = coarse/prev-timestep state). Replaces bilinear.
        self.tcu = bool(getattr(cfg, "tcu_upsample", False))
        if self.tcu and int(getattr(cfg, "latent_factor", 1)) > 1:
            _ocw = (2 * self.n_channels) if self.split_basis else (self.n_latent if self.latent_decoder else self.n_channels)
            self.wtcu = WindowedTCU(_ocw, int(cfg.latent_factor), win=int(getattr(cfg, "tcu_win", 3)), d=int(getattr(cfg, "tcu_dim", 32)))
        if (self.learned_upsample or self.pure_upsample) and int(getattr(cfg, "latent_factor", 1)) > 1:
            _LF = int(cfg.latent_factor)
            _wu_init = "glorot_uniform" if self.pure_upsample else "zeros"   # pure: no bilinear base → must init non-zero
            self.w_up = tf.keras.layers.Conv2D(out_per * _LF * _LF, 3, padding="same",
                                               kernel_initializer=_wu_init, name="w_up")
        # shock_adapt: data-derived analog of NTO-ADA's ν-scaled PE — a shock SENSOR (|∇ IC-velocity|)
        # drives a per-pixel/channel amplification of the learned high-freq upsample residual, so the
        # resolution ADAPTS (sharper) where shocks are. zero-init → starts as a no-op (w_hf unchanged).
        self.shock_adapt = bool(getattr(cfg, "shock_adapt", False))
        if self.shock_adapt and self.learned_upsample and int(getattr(cfg, "latent_factor", 1)) > 1:
            self.shock_scale = tf.keras.layers.Conv2D(out_per, 3, padding="same",
                                                      kernel_initializer="zeros", name="shock_scale")
        # ms_lup: MULTI-SCALE latent upsample. Instead of ONE coarse->128 upsample, build a pyramid of
        # scales s: pool W to s^2 then upsample s->128, and combine. Each branch imposes spatial correlation
        # at length ~128/s; combining synthesizes structure across scales (turbulence energy cascade). The
        # correlation the single coarse->fine upsample gave, now at MULTIPLE length scales. config-gated (list).
        # msada: MULTI-SCALE ADA. Run the expert bank INDEPENDENTLY at each scale s in {4,8,16,32} (pool the
        # STATE+encoder-ctx to s^2, experts compute W_s at s^2 = genuine coarse-scale dynamics), smooth-upsample
        # (bilinear) each W_s to the latent grid, learned-weighted-combine. Each scale carries structure at its
        # own length (~128/s); combining = turbulence energy cascade. (vs the broken ms_lup which pooled the
        # already-computed fine W = redundant blur.) Shared expert weights across scales (multi-grid). Then the
        # standard latent->128 upsample finishes. config-gated (list).
        # msdec: MULTI-SCALE DECODE. On the FINAL 128^2 W, synthesize spatial correlation across scales:
        # for each s, avg-pool W to s^2, learned 3x3 filter, SMOOTH (bilinear) upsample back to 128^2, combine.
        # Imposes correlation at length ~128/s. Include s=Nx (identity) to preserve fine detail => contains the
        # plain (no-synthesis) case, so it cannot be worse. NO differential operators here (no dilation limit)
        # so coarse scales (4,8) are fine, unlike running the physics expert coarse. Smooth upsample (bilinear),
        # NOT depth_to_space => no block artifacts. Pairs with latent_factor=1 (expert operators at full 128^2,
        # which alone loses the upsample's correlation; msdec adds it back multi-scale). config-gated (list).
        self._msdec = list(getattr(cfg, "msdec", []) or [])
        if self._msdec:
            self.msdec_c = [tf.keras.layers.Conv2D(out_per, 3, padding="same", name=f"msdec_c{s}")
                            for s in self._msdec]
            self.msdec_mix = tf.keras.layers.Conv2D(out_per, 1, kernel_initializer="glorot_uniform", name="msdec_mix")
        self._msada = list(getattr(cfg, "msada_scales", []) or [])
        if self._msada:
            self._msada_w = tf.Variable((1.0 / len(self._msada)) * tf.ones((len(self._msada),), tf.float32),
                                        trainable=True, name="msada_w")
            if bool(getattr(cfg, "msada_learned_up", False)):           # per-scale LEARNED decode (decode_s <- physics_s)
                self.msada_dec = [tf.keras.layers.Conv2D(out_per, 3, padding="same", name=f"msada_dec{_s}")
                                  for _s in self._msada]
        self._ms = list(getattr(cfg, "ms_lup", []) or [])
        if self._ms:
            self.ms_up = []
            for _s in self._ms:
                _r = cfg.Nx // int(_s)
                self.ms_up.append(tf.keras.layers.Conv2D(out_per * _r * _r, 3, padding="same",
                                  kernel_initializer="glorot_uniform", name=f"ms_up_{_s}"))
            self.ms_mix = tf.keras.layers.Conv2D(out_per, 1, kernel_initializer="glorot_uniform", name="ms_mix")

        # enc_expand: replace the LOSSY avg-pool of the state to the latent grid with a DEPTHWISE-EXPANDING
        # downsample. avg-pool (low-pass) destroys the turbulent high-freq before the experts see it; here we
        # KEEP the avg-pooled state (slots 0,1 intact → operator ω/div/u·∇ unchanged) AND concat a depthwise
        # strided conv (per-channel, depth_multiplier=M) that packs the LF×LF detail into 16·M extra feature
        # channels. The coarse-grid expert input becomes OVERCOMPLETE (32²×(16+16M) > 128²) → high-freq
        # survives. Output W dim (oc) and the zero-init start are unchanged. General (no PDE/equation key).
        self.enc_expand = bool(getattr(cfg, "enc_expand", False))
        if self.enc_expand and int(getattr(cfg, "latent_factor", 1)) > 1:
            _LFe = int(cfg.latent_factor); _Me = int(getattr(cfg, "enc_expand_mult", 4))
            self.dw_down = tf.keras.layers.DepthwiseConv2D(_LFe, strides=_LFe, padding="valid",
                                                           depth_multiplier=_Me, name="enc_dw_down")
        # img_refine: per-snapshot 2D image-to-image high-freq CORRECTOR at the output (time handled by the
        # ADA basis; this refines each frame's spatial high-freq). Full-res dilated-conv (1,2,4) residual,
        # zero-init out → starts as identity, learns to synthesize high-freq (esp. velocity). Gated by cfg.
        self.img_refine = bool(getattr(cfg, "img_refine", False))
        if self.img_refine:
            dR = int(getattr(cfg, "refine_dim", 48))
            self.ref_d1 = tf.keras.layers.Conv2D(dR, 3, padding="valid", dilation_rate=1, activation="gelu", name="ref_d1")
            self.ref_d2 = tf.keras.layers.Conv2D(dR, 3, padding="valid", dilation_rate=2, activation="gelu", name="ref_d2")
            self.ref_d4 = tf.keras.layers.Conv2D(dR, 3, padding="valid", dilation_rate=4, activation="gelu", name="ref_d4")
            self.ref_out = tf.keras.layers.Conv2D(self.n_channels, 3, padding="valid",
                                                  kernel_initializer="zeros", name="ref_out")
        eh = getattr(cfg, "expert_hidden", None)
        ekw = {"out_ch": out_per}
        if eh:
            ekw["hidden"] = eh
        self.experts = []
        for n in self.expert_names:
            kw = dict(ekw)
            if n == "elliptic":                       # iterated global-coupling option (NS Δ⁻¹/pressure)
                kw["rounds"] = int(getattr(cfg, "elliptic_rounds", 1))
                kw["dilations"] = tuple(getattr(cfg, "elliptic_dilations", (1, 2, 4, 8)))
                kw["multigrid"] = bool(getattr(cfg, "elliptic_multigrid", False))  # V-cycle global coupling
                kw["mg_levels"] = int(getattr(cfg, "elliptic_mg_levels", 3))       # restrict depth (32→16→8→4)
                kw["mg_hidden"] = int(getattr(cfg, "elliptic_mg_hidden", 0)) or None  # V-cycle width (0→lean)
            if n == "convective":                     # nonlinear self-advection u·∇z from state velocity
                kw["self_advect"] = bool(getattr(cfg, "conv_self_advect", False))
                kw["upwind"] = bool(getattr(cfg, "conv_upwind", False))   # stable upwind (central diverged)
                kw["muscl"] = bool(getattr(cfg, "conv_muscl", False))     # 2nd-order TVD (low diffusion)
            if n == "unified":                         # ablation flag: drop the physics-operator bank
                kw["use_ops"] = bool(getattr(cfg, "unified_ops", True))
                kw["fixedops"] = bool(getattr(cfg, "unified_fixedops", False))  # +FIXED exact PDE-operator branch (finetune)
                kw["use_shock"] = bool(getattr(cfg, "unified_shock", False))   # +shock-capturing feature
                kw["multigrid"] = bool(getattr(cfg, "unified_multigrid", False))  # NONLOCAL branch (V-cycle)
                kw["mg_levels"] = int(getattr(cfg, "unified_mg_levels", 3))
                kw["mg_hidden"] = int(getattr(cfg, "unified_mg_hidden", 0)) or None
                kw["xattn"] = bool(getattr(cfg, "unified_xattn", False))   # op↔transformer cross-attention
                kw["xattn_dim"] = int(getattr(cfg, "unified_xattn_dim", 256))
                kw["fno"] = bool(getattr(cfg, "unified_fno", False))       # FNO spectral global branch
                kw["fno_modes"] = int(getattr(cfg, "unified_fno_modes", 12))
                kw["fno_width"] = int(getattr(cfg, "unified_fno_width", 64))
                kw["reaction"] = bool(getattr(cfg, "unified_reaction", False))   # +pointwise R(u)+f branch (reaction-diffusion bias)
                kw["reaction_hidden"] = int(getattr(cfg, "unified_reaction_hidden", 32))
                kw["wave"] = bool(getattr(cfg, "unified_wave", False))           # +cross-channel ∇ coupling (wave/hyperbolic bias)
                kw["wave_hidden"] = int(getattr(cfg, "unified_wave_hidden", 32))
                kw["adapter"] = bool(getattr(cfg, "unified_adapter", False))     # +generic fresh local-operator adapter (transfer)
                kw["adapter_hidden"] = int(getattr(cfg, "unified_adapter_hidden", 48))
                kw["hibank"] = bool(getattr(cfg, "unified_hibank", False))        # +higher-order FD ops (Δ²,∂²xy,∂³,|∇|) — general-PDE vocabulary
                kw["buoyancy"] = bool(getattr(cfg, "unified_buoyancy", False))     # +scalar→velocity source (Boussinesq buoyancy)
                kw["buoyancy_hidden"] = int(getattr(cfg, "unified_buoyancy_hidden", 32))
                kw["advbias"] = bool(getattr(cfg, "unified_advbias", False))         # +learned upwind advection-bias head
                kw["advbias_hidden"] = int(getattr(cfg, "unified_advbias_hidden", 32))
                kw["gradx"] = bool(getattr(cfg, "unified_gradx", False))             # +horizontal ∂_x gradient/shear head
                kw["gradx_hidden"] = int(getattr(cfg, "unified_gradx_hidden", 32))
                kw["heads_off"] = tuple(getattr(cfg, "unified_heads_off", ()))       # FT-time selective head pruning (vars kept)
                kw["hsplit"] = int(getattr(cfg, "unified_hsplit", 0))                 # per-group latent differentiation (0=off, 3=groups)
                kw["hsplit_hidden"] = int(getattr(cfg, "unified_hsplit_hidden", 8))   # per-group bottleneck middle width
                kw["hsplit_shared"] = bool(getattr(cfg, "unified_hsplit_shared", False))  # param-matched control (one shared adapter)
                kw["hsplit_ch"] = int(self.n_latent) if self.latent_decoder else int(self.n_channels)  # operator-input latent
                kw["learnable_deriv"] = bool(getattr(cfg, "unified_learnable_deriv", False))  # learnable-FD stencils
            self.experts.append(_EXPERT_REGISTRY[n](d_geom, **kw))
        if self.split_basis:
            # per-pixel/per-channel Fourier-vs-Legendre selector from the shared geom features
            self.mix_head = tf.keras.layers.Conv2D(self.n_channels, 1, name="mix_head")
        # adaptive_mix: in the latent (split_basis=False) path the Fourier↔Legendre blend is ONE global scalar
        # (mix_logit) → it cannot specialize per family/region (turbulent pdearena wants Legendre/aperiodic,
        # waves want Fourier; the scalar gets stuck mid). Predict a PER-PIXEL mix logit-offset from the encoder
        # features instead: m(x) = sigmoid(mix_logit + mix_head_lat(h)). zero-init → m(x)=sigmoid(mix_logit)
        # everywhere at load (no change), then learns spatial Fourier/Legendre allocation. Single-shot, cheap.
        self.adaptive_mix = bool(getattr(cfg, "adaptive_mix", False)) and not self.split_basis
        if self.adaptive_mix:
            _cw_mix = self.n_latent if self.latent_decoder else self.n_channels   # match Cw (channels each basis integrates)
            self.mix_head_lat = tf.keras.layers.Conv2D(_cw_mix, 1, kernel_initializer="zeros", name="mix_head_lat")
        # warp_head (semi-Lagrangian): the per-pixel ADA integral CANNOT represent a feature TRANSPORTING across
        # cells (a buoyant plume rising lower→upper cell) — each pixel integrates its own fixed-t=0 W with no
        # spatial coupling, so a cell whose W(0)≈0 never receives the arriving plume. Fix: predict a per-pixel
        # displacement d(x,t)=v0·t+½a0·t² (v0,a0 from the encoder, zero-init) and Lagrangian-TRANSPORT the IC by
        # BACKWARD-WARPING it: base(x,t)=IC(x−d(x,t)) (bilinear, periodic). This MOVES features (unlike
        # extrap_residual which adds a ramp to the value). zero-init → d=0 → identity warp = current base at load.
        self.warp_head = bool(getattr(cfg, "warp_head", False))
        # warp_off: build warp vars (positional-load compatible) but skip the semi-Lagrangian transport in
        # the forward (base stays the constant-IC anchor) — FT-time pruning for tasks without transport.
        self.warp_off = bool(getattr(cfg, "warp_off", False))
        # warp_steps: 1 (default) = single warp by a 2nd-order Taylor displacement d=v0·t+½a0·t² (legacy).
        # >1 = VELOCITY-COUPLED COMPOSITIONAL semi-Lagrangian: trace the backward characteristic in K sub-steps
        # through a predicted velocity field u(x) (warp_v), sampling u at the moving departure point each step
        # (d_k = d_{k-1} + u(x−d_{k-1})·t/K) → CURVED path. Pure grid-sample (bounded → cannot diverge like the
        # bank-recursion combo-tf), single-shot (K cheap warps, no while_loop / no bank re-eval). zero-init→identity.
        self.warp_steps = int(getattr(cfg, "warp_steps", 1))
        if self.warp_head:
            self.warp_v = tf.keras.layers.Conv2D(2, 1, kernel_initializer="zeros", name="warp_v")  # velocity / displacement field
            self.warp_a = tf.keras.layers.Conv2D(2, 1, kernel_initializer="zeros", name="warp_a")  # accel (Taylor path, warp_steps=1)
        # wave_bank (trainable 2nd-order wave propagator): the warp head transports values (advection) and the
        # ADA integral damps — neither can form an EXPANDING/INTERFERING wavefront (u_tt = ∇·(c²∇u), 2nd-order
        # in time, conservative). Residual = truncated Taylor of the exact propagator
        #   u(t) = cos(t√L)u0 + t·sinc(t√L)v0,  L = c̃²Δ:
        #   Σ_k α_k·t^{2k}·L^k u0 + Σ_k β_k·t^{2k+1}·L^k v0 + γ·t²·(∇c̃²·∇u0)
        # with c̃² a learned per-pixel speed² (from IC+encoder; absorbs dx/dt scaling — Δ is PIXEL-space), v0 a
        # learned hidden initial velocity (u_t unobservable from a single-frame IC), and the γ term the
        # variable-coefficient interface part of ∇·(c²∇u) (layered-c reflection). α/β/γ zero-init → residual
        # ≡ 0 at load (name-keyed warm start unchanged); wb_c/wb_v keep default init so the coefficient
        # gradients are alive (∂res/∂β ∝ L^k v0 would be 0 forever under a zero-init v0 head).
        self.wave_bank = bool(getattr(cfg, "wave_bank", False))
        self.wave_bank_K = int(getattr(cfg, "wave_bank_K", 3))
        if self.wave_bank:
            self.wb_c = tf.keras.layers.Conv2D(1, 1, name="wb_c")                    # per-pixel speed² (softplus in use)
            self.wb_v = tf.keras.layers.Conv2D(self.n_channels, 1, name="wb_v")      # hidden initial velocity v0
            _K = self.wave_bank_K
            self.wb_alpha = tf.Variable(tf.zeros((_K, self.n_channels)), trainable=True, name="wb_alpha")
            self.wb_beta = tf.Variable(tf.zeros((_K + 1, self.n_channels)), trainable=True, name="wb_beta")
            self.wb_gamma = tf.Variable(tf.zeros((self.n_channels,)), trainable=True, name="wb_gamma")
        # ace_bank (trainable reaction-diffusion propagator, Allen-Cahn family): the generic ReactionExpert is a
        # pointwise MLP on the INSTANTANEOUS state — it cannot express the LONG-TIME saturating flow of the
        # bistable kinetics nor the curvature-driven interface motion that dominates Allen-Cahn. Residual =
        #   γ_r·(flow(u0,t) − u0) + Σ_k δ_k·t^k·Δ^k u0 + γ_κ·t·κ|∇u0|
        # where flow is the CLOSED FORM of u_t = a·u − (s·a)·u³:  u(t)=u0·e^{at}/√(1+s·u0²(e^{2at}−1))
        # (a rate, s saturation — learned per-channel), the δ sum is the diffusion semigroup e^{ε²tΔ} Taylor
        # (PIXEL-space Δ, scale absorbed in δ), and κ|∇u| the mean-curvature front speed. Gates γ_r/δ/γ_κ
        # zero-init → residual ≡ 0 at load; a init 1 (NOT 0: flow(a=0)=u0 would zero γ_r's gradient forever).
        self.ace_bank = bool(getattr(cfg, "ace_bank", False))
        self.ace_bank_K = int(getattr(cfg, "ace_bank_K", 2))
        if self.ace_bank:
            _C = self.n_channels
            self.ab_a = tf.Variable(tf.ones((_C,)), trainable=True, name="ab_a")          # kinetics rate
            self.ab_s = tf.Variable(tf.zeros((_C,)), trainable=True, name="ab_s")         # saturation (softplus in use)
            self.ab_gr = tf.Variable(tf.zeros((_C,)), trainable=True, name="ab_gr")       # kinetics gate
            self.ab_delta = tf.Variable(tf.zeros((self.ace_bank_K, _C)), trainable=True, name="ab_delta")
            self.ab_gk = tf.Variable(tf.zeros((_C,)), trainable=True, name="ab_gk")       # curvature gate
        # loose_bank (physics VOCABULARY + learned combiner): the rigid wave/ace banks fix the functional form
        # (analytic propagator, scalar gains) — if the data's effective dynamics deviate, the cheapest rel-L1
        # solution is gate≈0 (observed: wave_bank tied its base). Here physics enters only as FEATURES
        # (u0 powers / Δ,Δ² / c̃²-weighted Laplacians / hidden v0 / |∇u|, curvature κ|∇u|, ∇c̃²·∇u0) and the
        # COMBINATION is learned, separable in time:  residual(x,t) = Σ_m w_m(t)·B_m(x), with B = 1×1-conv MLP
        # of the feature stack and w(t) a small MLP of a Fourier lead-time embedding. The analytic propagators
        # live INSIDE this hypothesis space (w can recover Taylor-in-t, B the operator combos) but the model is
        # free to leave it (damping, dispersion, nonlinear media). lb_tw zero-init → residual ≡ 0 at load;
        # basis/feature heads keep default init so ∂res/∂w = B ≠ 0 bootstraps the branch.
        self.loose_bank = bool(getattr(cfg, "loose_bank", False))
        self.loose_bank_M = int(getattr(cfg, "loose_bank_M", 8))
        if self.loose_bank:
            _C = self.n_channels
            self.lb_c2 = tf.keras.layers.Conv2D(1, 1, name="lb_c2")                       # c̃² (softplus in use)
            self.lb_v = tf.keras.layers.Conv2D(_C, 1, name="lb_v")                        # hidden initial velocity
            self.lb_hp = tf.keras.layers.Conv2D(16, 1, activation="gelu", name="lb_hp")   # encoder-feature squeeze
            self.lb_h1 = tf.keras.layers.Conv2D(64, 1, activation="gelu", name="lb_h1")   # combiner hidden
            self.lb_bm = tf.keras.layers.Conv2D(self.loose_bank_M * _C, 1, name="lb_bm")  # basis maps B_m(x)
            self.lb_t1 = tf.keras.layers.Dense(32, activation="gelu", name="lb_t1")       # time-weight MLP
            self.lb_tw = tf.keras.layers.Dense(self.loose_bank_M, kernel_initializer="zeros",
                                               bias_initializer="zeros", name="lb_tw")    # zero-init → no-op
        # phase_bank (LOOSE SPECTRAL-PHASE vocabulary): the loose_bank residual Σ w_m(t)B_m(x) is SEPARABLE in
        # (x,t) — a traveling wave u(x−ct) is not low-rank separable — and the rigid wave_bank Taylor-truncates
        # the phase, so neither spans the Wave-Layer failure mode (accumulating phase drift). This arm supplies
        # NON-SEPARABLE phase carriers as a learnable VOCABULARY on the radial wavenumber κ=|k|:
        # {sin/cos(c_i κτ), e^{−g_i κτ}, e^{−g_i κ²τ}} with LEARNED speeds c_i / rates g_i, combined by a
        # zero-init MLP into (κ,τ)-multipliers (G1,G2) applied to û0(k) and a learned hidden v̂0(k) in rfft
        # space. The exact d'Alembert propagator (G1=cos(cκτ), G2∝sinc) lives INSIDE this space; dispersion,
        # damping and mode mixing are free. pb_g zero-init → residual ≡ 0 at load; ×τ ramp keeps IC exact.
        self.phase_bank = bool(getattr(cfg, "phase_bank", False))
        self.phase_bank_S = int(getattr(cfg, "phase_bank_S", 4))
        if self.phase_bank:
            _S = self.phase_bank_S
            self.pb_v = tf.keras.layers.Conv2D(self.n_channels, 1, name="pb_v")   # hidden initial velocity v0
            self.pb_h1 = tf.keras.layers.Dense(48, activation="gelu", name="pb_h1")
            self.pb_g = tf.keras.layers.Dense(2 * self.n_channels, kernel_initializer="zeros",
                                              bias_initializer="zeros", name="pb_g")  # (κ,τ)→(G1,G2) zero-init
            _c0 = np.log(np.pi * (4.0 ** np.arange(_S))).astype(np.float32)           # π,4π,16π,64π
            _g0 = np.log(0.5 * (4.0 ** np.arange(_S))).astype(np.float32)             # 0.5,2,8,32
            self.pb_logc = tf.Variable(_c0, trainable=True, name="pb_logc")
            self.pb_logg = tf.Variable(_g0, trainable=True, name="pb_logg")
        # adv_bank (LOOSE TRANSPORT vocabulary, pdearena diagnosis-matched): spectral analysis shows the
        # joint model's pdearena gap is TRANSPORT-PHASE misplacement (spectrum right, phases wrong) — the
        # task2 arch has NO transport mechanism (no warp) and separable heads cannot move features across
        # cells coherently over long leads. Residual = gated semi-Lagrangian transport of the IC through a
        # TIME-MODULATED learned velocity field
        #   ṽ(x,τ) = V0(x) + Σ_m w_m(τ)·V_m(x),   d(x,τ) traced in K bounded substeps,
        #   base += tanh(σ_av) ⊙ (IC(x−d(x,τ)) − IC(x)).
        # Velocity vocabulary {V_m} and temporal mixture w(τ) are LEARNED (default-init, live);
        # σ_av zero-init → residual ≡ 0 at load; d(x,0)=0 keeps the IC exact. Constant-velocity advection
        # lives inside the space (w≡0); time-varying chaotic transport is approximated by M temporal modes.
        # char_bank (CHARACTERISTICS + INTERFACE-REFLECTION vocabulary, Wave-Layer diagnosis-matched): the
        # loose_bank residual is (x,t)-separable and phase_bank is isotropic in κ=|k| — neither carries
        # DIRECTIONAL traveling fronts nor interface-localized secondary (reflected/transmitted) fronts,
        # which is exactly the layered-medium failure mode. Vocabulary = 4 one-way characteristic transports
        # of the IC at a LEARNED bounded local speed c̃(x) (semi-Lagrangian along ±x/±y — exact traveling
        # waves for piecewise-constant c̃) + the same carriers gated by a learned interface mask m(x) from
        # |∇c̃| (secondary fronts are born where the speed jumps; opposite-direction × mask products let the
        # combiner synthesize reflections). A zero-init 1×1-conv combiner mixes the (8C+2) carrier stack per
        # (x,τ); ×τ ramp keeps the IC exact. cb_g zero-init → residual ≡ 0 at load (name-keyed fresh vars).
        self.char_bank = bool(getattr(cfg, "char_bank", False))
        self.char_bank_smax = float(getattr(cfg, "char_bank_smax", 32.0))
        if self.char_bank:
            _C = self.n_channels
            self.cb_c = tf.keras.layers.Conv2D(1, 1, name="cb_c")    # c̃(x) local speed (softplus→tanh bounded)
            self.cb_m = tf.keras.layers.Conv2D(1, 1, name="cb_m")    # interface gate from |∇c̃| + features
            self.cb_h1 = tf.keras.layers.Conv2D(48, 1, activation="gelu", name="cb_h1")
            self.cb_g = tf.keras.layers.Conv2D(_C, 1, kernel_initializer="zeros",
                                               bias_initializer="zeros", name="cb_g")
        self.adv_bank = bool(getattr(cfg, "adv_bank", False))
        self.adv_bank_M = int(getattr(cfg, "adv_bank_M", 4))
        self.adv_bank_K = int(getattr(cfg, "adv_bank_K", 4))
        # adv_bank_off: create the adv_bank vars (positional ckpt-load compatible with an advb pretrain)
        # but SKIP the residual in the forward — A/B lever for "disable task-irrelevant vocabulary at
        # finetune": tests whether the advb transfer deficit is the ACTIVE mismatched residual path
        # (recoverable at FT time) or the pretrained representation itself (not recoverable).
        self.adv_bank_off = bool(getattr(cfg, "adv_bank_off", False))
        if self.adv_bank:
            _M = self.adv_bank_M
            # stabilized init (adv_bank_vstd>0 / adv_bank_tw0): under the CLAMPED rel-L1 pretrain loss a
            # live glorot velocity field pushes early errors past the clamp (gradient death spiral, run
            # diverged by ~10k); small-amplitude V + zero time-mixture keeps the bootstrap path (gate sees
            # the V0 branch) while starting as a near-identity transport. Phase-1 (normalized MSE) is fine
            # with the live default.
            _vstd = float(getattr(cfg, "adv_bank_vstd", 0.0))
            _vini = tf.keras.initializers.RandomNormal(stddev=_vstd) if _vstd > 0 else "glorot_uniform"
            _twini = "zeros" if bool(getattr(cfg, "adv_bank_tw0", False)) else "glorot_uniform"
            self.adv_bank_clip = float(getattr(cfg, "adv_bank_clip", 8.0))
            self.av_V = tf.keras.layers.Conv2D(2 * (_M + 1), 1, kernel_initializer=_vini,
                                               name="av_V")                    # V0 | M velocity basis fields
            self.av_t1 = tf.keras.layers.Dense(32, activation="gelu", name="av_t1")
            self.av_tw = tf.keras.layers.Dense(_M, kernel_initializer=_twini,
                                               bias_initializer="zeros", name="av_tw")  # temporal mixture w(τ)
            self.av_gate = tf.Variable(tf.zeros((self.n_channels,)), trainable=True, name="av_gate")
        # teacher_forced: BCAT-style cheap recursion. The recursive while_loop is sequential (slow) in BOTH
        # train and inference. AR transformers are cheap to TRAIN via teacher forcing (feed GT states, one
        # parallel pass) — only inference is sequential. Here: during TRAINING feed the GROUND-TRUTH states at
        # the panel times into the bank → all panels' W in ONE batched call (reuse _rollout_pw_batched with a
        # GT z_st override, no sequential loop, no fixedpoint); at INFERENCE (no target) fall back to the
        # sequential while_loop (the recursive model). Net: training ~single-shot speed, model stays recursive.
        self.teacher_forced = bool(getattr(cfg, "teacher_forced", False))
        # slot_structured: keep the latent CHANNEL-IDENTITY clean — the first C latent slots ARE the physical
        # channels (vx,vy,scalar,rho,p,height passed through), the remaining (D−C) are learned extra capacity.
        # Then the operators read CLEAN physical fields: advection's vel_slots (0,1)=real velocity, the
        # buoyancy/source branch reads real smoke (slot 2) — not an entangled 6→D learned mix. ic_enc then only
        # produces the (D−C) EXTRA channels. Requires latent_decoder + D>C.
        self.slot_structured = bool(getattr(cfg, "slot_structured", False))
        # history_augment: the no-recursion bank sees only z0 (frozen IC). Augment z0 with a learned encoding of
        # the last K input frames (a temporal conv) so the operators see the OBSERVED recent trend (velocity/
        # accel/higher-order) — richer than the fixed ∂u/∂t (banktd) and without the full-history bottleneck
        # (z0 = [current-frame encode | K-frame history features], history is ADDED not replacing the state).
        self.history_augment = bool(getattr(cfg, "history_augment", False))
        self.history_k = int(getattr(cfg, "history_k", 3))
        self.history_dim = int(getattr(cfg, "history_dim", 6))
        if self.latent_decoder:
            # z0 = learned 1×1 encode of the physical IC (C) → latent (D): the rollout's starting state.
            if self.history_augment:
                _enc_out = max(self.n_latent - self.history_dim, 1)        # current-frame block
                self.hist_enc = tf.keras.layers.Conv2D(self.history_dim, 3, padding="same", name="hist_enc")  # K-frame history
            elif self.slot_structured:
                _enc_out = max(self.n_latent - self.n_channels, 1)
            else:
                _enc_out = self.n_latent
            self.ic_enc = tf.keras.layers.Conv2D(_enc_out, 1, name="ic_enc")  # slot_structured → only the EXTRA block
            # per-pixel no-bias MLP decoder: latent trajectory (D) → physical correction δ (C). no-bias
            # so decode(0)=0 → with the residual ω̂=IC+δ and basis ic=0, the hard-IC is exact.
            dh = int(getattr(cfg, "decode_hidden", 32))
            self.decode_h = tf.keras.layers.Dense(dh, activation="gelu", use_bias=False, name="decode_h")
            self.decode_out = tf.keras.layers.Dense(self.n_channels, use_bias=False, name="decode_out")
            # omega_decode: velocity increment as a learned scalar vorticity-potential -> solenoidal u
            self.omega_decode = bool(getattr(cfg, "omega_decode", False))
            if self.omega_decode:
                self.decode_w = tf.keras.layers.Dense(1, use_bias=False, name="decode_w")
            # group_decode: per-desc-group readout (shared bank vocabulary, per-group grammar). Same
            # no-bias+gelu construction as decode_h/out -> zero-latent still decodes to 0 (hard-IC kept).
            self.group_decode = bool(getattr(cfg, "group_decode", False))
            if self.group_decode:
                self.gdec_h = [tf.keras.layers.Dense(dh, activation="gelu", use_bias=False,
                                                     name=f"gdec_h{g}") for g in range(3)]
                self.gdec_out = [tf.keras.layers.Dense(self.n_channels, use_bias=False,
                                                       name=f"gdec_out{g}") for g in range(3)]

        # --- operator gate (conditioner) ---
        # field_gate: route from the pooled field features h_geom (regime-aware + transfers to unseen
        # equations) instead of the equation-label descriptor; descriptor kept as a dropout-prior.
        self.field_gate = bool(getattr(cfg, "field_gate", False))
        if self.field_gate:
            self.gate = FieldGate(len(self.experts),
                                  desc_dropout=float(getattr(cfg, "gate_desc_dropout", 0.5)))
        else:
            self.gate = FamilyGate(len(self.experts))

        # input-router (alternative to output-gating): instead of α_k weighting expert OUTPUTS, a
        # per-pixel per-expert sigmoid route r_k(x)=σ(Conv(h_geom)) decides — FROM THE INPUT, once,
        # up front — how much of each expert's contribution flows in at each location. Independent
        # sigmoid (not softmax) → experts can be jointly active/inactive per pixel. Ties each expert's
        # learning signal to where it's actually routed (vs a global α that can starve low-gated
        # experts) and unifies the input pathway (one h_geom → router → experts). See project_gate_design.
        self.input_router = bool(getattr(cfg, "input_router", False))
        if self.input_router:
            # TRUE input dispatch (see _panel): route_k(x)=σ(Conv(h_geom)) scales how much of the
            # encoded input h_geom flows INTO expert k. Stability comes from the experts' zero-init
            # output (W starts ≈0), so we use a NEUTRAL bias (route≈0.5 → experts get moderate context
            # to start differentiating) — unlike output-gating which needed a negative bias to bound W.
            self.router = tf.keras.layers.Conv2D(len(self.experts), 1, name="router")

        # attn-router: the input router as a small TRANSFORMER (verified data→weights mapping) instead
        # of a 1×1 conv. softmax-over-experts → bounded ΣW. Experts unchanged. (DESIGN.md §router.)
        # unified_expert: single physics expert (no routing/collapse) + the transformer as a feature
        # ENCODER (encode_dim=d_geom, no expert softmax). The encoder features become the expert's
        # conditioning context. Requires experts=("unified",).
        self.unified_expert = bool(getattr(cfg, "unified_expert", False))
        self.attn_router = bool(getattr(cfg, "attn_router", False))
        if self.attn_router:
            self.router = AttnRouter(len(self.experts),
                                     d_model=int(getattr(cfg, "router_dim", 192)),
                                     depth=int(getattr(cfg, "router_depth", 2)),
                                     heads=int(getattr(cfg, "router_heads", 4)),
                                     patch=int(getattr(cfg, "router_patch", 8)),
                                     route_act=str(getattr(cfg, "router_act", "softmax")),
                                     encode_dim=(d_geom if self.unified_expert else 0))
        # router_descriptor: map the equation descriptor → a per-expert LOGIT bias added DIRECTLY to the
        # route logits (not concat to the 1024-ch input, where 6 desc channels drown). gives each family an
        # explicit per-expert prior → breaks the data-only diffusive-collapse while keeping field routing.
        self.router_descriptor = bool(getattr(cfg, "router_descriptor", False))
        if self.router_descriptor:
            self.desc_route_bias = tf.keras.layers.Dense(len(self.experts), name="desc_route_bias")
        # unified flag: any per-pixel input-dispatch router (legacy sigmoid conv OR attn) drives the
        # input-dispatch path in operator()/_panel.
        self._router_on = self.attn_router or self.input_router

        # router grid message passing: prepend GraphSAGE-style multi-scale neighbor aggregation to the
        # router input so routing is regime-aware (local Re/length-scale from spatial connectivity).
        self.router_mp = bool(getattr(cfg, "router_mp", False)) and self._router_on
        if self.router_mp:
            self.grid_mp = GridMP(d_geom, depth=int(getattr(cfg, "router_mp_depth", 2)),
                                  dilations=tuple(getattr(cfg, "router_mp_dilations", (1, 2, 4))))

        # route-context conditioning: feed each expert not only its dispatched input route_k·h_geom but
        # an embedding of the FULL routing distribution (how the router allocated across ALL experts), so
        # a SHARED expert can disambiguate which family-regime it serves (NS vs DR ...) and specialize
        # instead of finding a destructive compromise. Addresses the observed inter-family competition.
        # zero-init Conv2D → starts as a no-op (preserves the stable zero-W start). (DESIGN.md §11.)
        self.route_context = bool(getattr(cfg, "route_context", False)) and self._router_on
        if self.route_context:
            self.route_ctx = tf.keras.layers.Conv2D(d_geom, 1, kernel_initializer="zeros",
                                                    name="route_ctx")
        # route_film: stronger form of route-context — FiLM(γ,β) modulation (domain+point) instead of
        # additive embedding (Poseidon adaLN / Unisolver). Overrides route_context when both set.
        self.route_film = bool(getattr(cfg, "route_film", False)) and self._router_on
        if self.route_film:
            self.route_film_mod = RouteFiLM(d_geom)
        # regime_film: FiLM(γ,β) modulation of h_geom by the FIELD REGIME (no router needed → works in the
        # unified config). Lets one shared decoder split per regime (compressible ρ varies vs incompressible
        # ρ const) instead of collapsing to the dummy-supervised majority (the com_ns trade-off).
        self.regime_film = bool(getattr(cfg, "regime_film", False))
        if self.regime_film:
            self.regime_film_mod = RegimeFiLM(d_geom)

        # ① panel-step embedding (NTO-ADA recursive design): condition each rollout panel on its step
        # index i (sinusoidal) so experts know WHERE in the trajectory they are (early vs late dynamics).
        # Our g-feedback rollout was panel-agnostic (only the evolved z distinguished panels). zero-init.
        self.step_embed = bool(getattr(cfg, "step_embed", False))
        if self.step_embed:
            d_step = int(getattr(cfg, "step_dim", 32))
            pos = np.arange(cfg.N_p)[:, None]
            div = np.exp(np.arange(0, d_step, 2) * (-np.log(10000.0) / d_step))
            tbl = np.zeros((cfg.N_p, d_step), np.float32)
            tbl[:, 0::2] = np.sin(pos * div); tbl[:, 1::2] = np.cos(pos * div)
            self._step_tbl = tf.constant(tbl, tf.float32)
            self.step_proj = tf.keras.layers.Dense(d_geom, kernel_initializer="zeros", name="step_proj")
        # ② spatial-context W readout: the shared ADA basis integrates W per-pixel (location-agnostic),
        # so mix W spatially (depthwise, periodic) BEFORE the basis — each pixel's W then sees neighbors',
        # giving the readout the surrounding-field context the pointwise basis lacks (NS advective
        # coupling). residual + zero-init → identity at start. (NTO-ADA used per-location ADA so W carried
        # position implicitly; we share the basis → need explicit spatial context.)
        self.w_spatial = bool(getattr(cfg, "w_spatial", False))
        if self.w_spatial:
            self.w_dw = tf.keras.layers.DepthwiseConv2D(3, padding="valid",
                                                        depthwise_initializer="zeros", name="w_spatial")

        # ③ cross-expert W-feedback (recursive coupling ACROSS experts, not just within each expert):
        # at panel i every expert additionally sees an embedding of ALL experts' PREVIOUS-panel
        # contributions {W_k^{i-1}} — not only the summed/integrated state z. This is operator-splitting
        # WITH coupling: e.g. the elliptic/pressure expert can react to the convective+diffusive predicted
        # tendency (Chorin projection: pressure acts on the others' tentative dz/dt → incompressible NS).
        # It also hands each expert the instantaneous dz/dt (momentum), which z alone hides. The previous
        # panel's per-expert stack (B,h,w,oc,K) is carried in the rollout and embedded by a zero-init 1×1
        # conv → identity at start (preserves the validated zero-W start), learns coupling only if useful.
        self.w_feedback = bool(getattr(cfg, "w_feedback", False))
        if self.w_feedback:
            # zero-init conv THEN tanh → the feedback added to h_geom is BOUNDED to ±1 per channel no
            # matter how large the previous-panel W stack is. This BREAKS the positive-feedback
            # amplification loop (big W → big cp → bigger W → … → NaN) that diverged the un-bounded version
            # at step ~400, WITHOUT LayerNorm's small-variance (≈constant pixel) blow-up. tanh(0)=0 + zero-
            # init conv → identity at start (validated zero-W start preserved); even cp=inf → tanh→±1.
            self.wfb_proj = tf.keras.layers.Conv2D(d_geom, 1, kernel_initializer="zeros", name="wfb_proj")

        # ②' panel-axis attention (parallel replacement for the sequential W-recursion): in the pw_batched
        # path the N_p panels are computed independently — panel_attn then couples them with ONE bidirectional
        # self-attention over the panel axis (per pixel), so each panel's W sees all others. Captures the
        # inter-panel/self-consistency the Euler recursion gave, but in a SINGLE parallel pass (batchable →
        # keeps pw_batched speed). zero-init out → starts == pw_batched. Only active on the _pwb batched path.
        self.panel_attn = bool(getattr(cfg, "panel_attn", False))
        if self.panel_attn:
            _pa_dim = self.n_latent if self.latent_decoder else ((2 * self.n_channels) if self.split_basis else self.n_channels)
            self.panel_attn_layer = PanelAttn(_pa_dim, int(cfg.N_p),
                                              heads=int(getattr(cfg, "panel_attn_heads", 4)),
                                              d_attn=int(getattr(cfg, "panel_attn_dim", 0)),
                                              layers=int(getattr(cfg, "panel_attn_layers", 1)),
                                              mlp_ratio=int(getattr(cfg, "panel_attn_mlp_ratio", 2)))

        # panel_advect: ADVECTIVE W-panel coupling (semi-Lagrangian; panel k sees the IC-velocity-advected
        # previous panel). Attacks the space-time SEPARABILITY that limits turbulence (pdearena). zero-init.
        self.panel_advect = bool(getattr(cfg, "panel_advect", False))
        if self.panel_advect:
            _apc_ch = self.n_latent if self.latent_decoder else ((2 * self.n_channels) if self.split_basis else self.n_channels)
            self.apc_layer = AdvPanelCouple(_apc_ch, int(cfg.N_p))

        # recursive_gru: replace the FIXED Euler state-update (z+=W·dt) in the recursive rollout with a LEARNED
        # GRU-style gated update z' = z + u⊙cand (u=update gate, cand=zero-init candidate of [W, r⊙z]). Keeps
        # the physics bank (W=operators on z) but LEARNS the integrator (generalizes Euler/AB2's fixed scheme;
        # the gate can damp → stabilizes). zero-init candidate → z'=z at start (validated identity start). 1×1
        # convs at the state resolution. Sequential (recursive path only).
        self.recursive_gru = bool(getattr(cfg, "recursive_gru", False))
        if self.recursive_gru:
            _gc = self.n_latent if self.latent_decoder else ((2 * self.n_channels) if self.split_basis else self.n_channels)
            self.gru_u = tf.keras.layers.Conv2D(_gc, 1, activation="sigmoid", name="gru_u")   # update gate
            self.gru_r = tf.keras.layers.Conv2D(_gc, 1, activation="sigmoid", name="gru_r")   # reset gate
            self.gru_c = tf.keras.layers.Conv2D(_gc, 1, kernel_initializer="zeros", name="gru_c")  # candidate (zero→identity)

        # extrap_residual: upgrade the residual ansatz's ANCHOR from "ic (last frame, constant over t)" to a
        # learned TEMPORAL EXTRAPOLATION of the input history: base(t) = ic + v0·t + ½·a0·t², where v0
        # (observed velocity) and a0 (observed ACCELERATION = the buoyancy/forcing signature) are learned from
        # the input frame 1st/2nd differences. This captures the FORCED secular drift (smoke-buoyancy plume
        # rise) directly from the IC frames — the part the conservative operators + ADA basis represent poorly
        # — so ADA only models the deviation. zero-init → base=ic at start (== current). Multi-frame (Tin≥3).
        self.extrap_residual = bool(getattr(cfg, "extrap_residual", False))
        if self.extrap_residual:
            self.ex_proj = tf.keras.layers.Conv2D(2 * self.n_channels, 3, padding="same",
                                                  kernel_initializer="zeros", name="ex_proj")  # → (v0, a0)

        # learned_frontend: ORDER-B with LEARNED architecture-bias heads (not explicit operators). Apply a few
        # bias-typed CONV heads to the CLEAN raw IC (multi-dilation = transport bias, 1×1 = pointwise/source
        # bias) → learned physics-bias features → prepend to the encoder input → transformer. Combines the
        # archbias finding (learned bias > explicit ops) with Order B (bias at the raw front-end, before the
        # transformer's global mixing). Learned, not hardcoded; on raw fields, not the abstract latent.
        self.learned_frontend = bool(getattr(cfg, "learned_frontend", False))
        if self.learned_frontend:
            _lfd = int(getattr(cfg, "lfe_dim", 16))
            self.lfe_dil = [tf.keras.layers.Conv2D(_lfd, 3, dilation_rate=d, activation="gelu", name=f"lfe_d{d}")
                            for d in (1, 2, 4)]                       # transport / multi-scale bias
            self.lfe_pt = tf.keras.layers.Conv2D(_lfd, 1, activation="gelu", name="lfe_pt")  # pointwise / source bias

        # learned weighted sum: replace the plain W_scale·Σ_k B_k with Σ_k ω_{c,k}·B_k (per-channel,
        # per-expert learnable weights, init = W_scale → identical to the plain sum at start). Lets the
        # model down-weight / cancel REDUNDANT experts at the COMBINATION stage (the plain unweighted sum
        # double-counts overlapping operators); ω unconstrained → can go negative (subtract redundancy).
        self.learned_combine = bool(getattr(cfg, "learned_combine", False))
        if self.learned_combine:
            _oc = (2 * self.n_channels) if self.split_basis else self.n_channels
            _ws = float(getattr(cfg, "w_scale_init", 0.05))
            self.omega = tf.Variable(_ws * tf.ones((_oc, len(self.experts)), tf.float32),
                                     trainable=True, name="combine_w")

        self._dt = tf.constant(cfg.T_final / cfg.N_p, tf.float32)

        # static (x, y) coordinate grid + uniform cell size (computational-coordinate
        # metric, §4.6). Domain length is configurable: synthetic families use [0,2π)²,
        # the Poseidon ACE benchmark uses the unit square [0,1]².
        self._L = float(getattr(cfg, "domain_L", 2.0 * np.pi))
        self._dx = self._L / cfg.Nx
        xs = np.linspace(0.0, self._L, cfg.Nx, endpoint=False, dtype=np.float32)
        gx, gy = np.meshgrid(xs, xs, indexing="ij")
        self._coords = tf.constant(np.stack([gx, gy], -1)[None], tf.float32)  # (1,H,W,2)

        # spatial Fourier positional features: multi-frequency sin/cos of normalized (x,y) — an EXPLICIT
        # spatial-phase basis so the net can place advected structure at a precise POSITION (=phase),
        # not only via implicit grid row/col. Config-gated (spatial_fourier = #octaves; 0 = off).
        self._spatial_fourier = int(getattr(cfg, "spatial_fourier", 0))
        if self._spatial_fourier > 0:
            _xn = np.linspace(0.0, 1.0, cfg.Nx, endpoint=False, dtype=np.float32)
            _gxn, _gyn = np.meshgrid(_xn, _xn, indexing="ij")
            _ff = []
            for _j in range(self._spatial_fourier):
                _f = (2.0 ** _j) * np.pi
                _ff += [np.sin(_f * _gxn), np.cos(_f * _gxn), np.sin(_f * _gyn), np.cos(_f * _gyn)]
            self._coords_ff = tf.constant(np.stack(_ff, -1)[None], tf.float32)  # (1,H,W,4*sf)

        # spatial Fourier at the EXPERT BANK: inject the positional-phase basis DIRECTLY into the expert
        # context (latent grid), BYPASSING the deep conv encoder (NeRF/INR: PE must reach the output head
        # undegraded, not buried at the encoder input). config-gated (spatial_fourier_expert = #octaves).
        self._sfe = int(getattr(cfg, "spatial_fourier_expert", 0))
        if self._sfe > 0:
            _lf = int(getattr(cfg, "latent_factor", 1)); _latn = cfg.Nx // _lf
            _xl = np.linspace(0.0, 1.0, _latn, endpoint=False, dtype=np.float32)
            _gxl, _gyl = np.meshgrid(_xl, _xl, indexing="ij")
            _ffl = []
            for _j in range(self._sfe):
                _f = (2.0 ** _j) * np.pi
                _ffl += [np.sin(_f * _gxl), np.cos(_f * _gxl), np.sin(_f * _gyl), np.cos(_f * _gyl)]
            self._coords_ff_lat = tf.constant(np.stack(_ffl, -1)[None], tf.float32)  # (1,latn,latn,4*sfe)
            self.pos_proj = tf.keras.layers.Conv2D(int(cfg.d_geom), 1, kernel_initializer="zeros",
                                                   name="pos_proj_expert")

        # super_factor: run experts at SF x the latent grid (super-resolution), then learned-DOWNSAMPLE W
        # back to Nx. Reintroduces the structured spatial correlation the coarse->fine upsample gave (lost at
        # latent_factor=1) WHILE computing dynamics at higher resolution (LES / anti-aliasing). Depthwise conv
        # initialized to average-pooling (sane downsample at start), learns a per-channel filter. Requires
        # latent_factor=1 + fixedpoint_iters=0 + no teacher-forcing (AR path). config-gated (super_factor=1 off).
        self._sf = int(getattr(cfg, "super_factor", 1))
        if self._sf > 1:
            self.w_down = tf.keras.layers.DepthwiseConv2D(
                self._sf, strides=self._sf, padding="valid",
                depthwise_initializer=tf.keras.initializers.Constant(1.0 / (self._sf * self._sf)),
                name="w_down")

        # Build ALL submodules (geo, experts, gate) via a full dummy operator pass so that
        # trainable_variables is complete and order-stable — required for checkpoint
        # save/load (otherwise lazily-built gate/expert weights are missing at load time).
        dd = int(getattr(cfg, "desc_dim", 3))
        tin = int(getattr(cfg, "T_in", 1))
        u_dummy = (tf.zeros((1, tin, H, W, self.n_channels)) if tin > 1
                   else tf.zeros((1, H, W, self.n_channels)))
        self.operator(u_dummy, tf.zeros((1, dd)), tf.zeros((1, 4)))
        if self.latent_decoder:                          # build decode_h/decode_out (operator alone won't)
            self.decode_out(self.decode_h(tf.zeros((1, 1, 1, 1, self.n_latent))))
            if getattr(self, "omega_decode", False):
                self.decode_w(self.decode_h(tf.zeros((1, 1, 1, 1, self.n_latent))))
            if getattr(self, "group_decode", False):
                for _g in range(3):
                    self.gdec_out[_g](self.gdec_h[_g](tf.zeros((1, 1, 1, 1, self.n_latent))))
        if self.img_refine:                              # build refiner (used only in call(), not operator)
            _xr = tf.zeros((1, H, W, self.n_channels))
            _h = self.ref_d1(periodic_pad_2d(_xr, 1)); _h = self.ref_d2(periodic_pad_2d(_h, 2))
            _h = self.ref_d4(periodic_pad_2d(_h, 4)); self.ref_out(periodic_pad_2d(_h, 1))
        if self.extrap_residual:                         # build ex_proj (lives in call(), not operator)
            self.ex_proj(tf.zeros((1, H, W, 2 * self.n_channels)))
        if self.warp_head:                               # build warp_v/warp_a (live in call(); operator stashed _h_geom)
            self.warp_v(self._h_geom); self.warp_a(self._h_geom)
        if self.lag_channel:                             # build lag_* (live in call(); grad-accum snapshots vars pre-forward)
            _hsl = self._h_geom[:, :: self.lag_stride, :: self.lag_stride, :]
            _icl = tf.zeros((1, H // self.lag_stride, W // self.lag_stride, self.n_channels), dtype=_hsl.dtype)
            _hhl = self.lag_h1(tf.concat([_hsl, _icl], -1))
            self.lag_Ad(_hhl); self.lag_Ac(_hhl)
            self.lag_gate(tf.zeros((1, dd), dtype=self._h_geom.dtype))
            if self.lag_omega:
                self.lag_Aw(_hhl); self.lag_gate_w(tf.zeros((1, dd), dtype=self._h_geom.dtype))
        if self.buoyancy_bank:                           # build bb_* (live in call(); accum snapshots pre-forward)
            self.bb_out(self.bb_h(tf.zeros((1, H, W, 3), dtype=self._h_geom.dtype)))
            self.bb_coef(tf.zeros((1, int(self._h_geom.shape[-1])), dtype=self._h_geom.dtype))
        if self.wave_bank:                               # build wb_c/wb_v (live in call(); operator stashed _h_geom)
            _icd = tf.zeros((1, H, W, self.n_channels), dtype=self._h_geom.dtype)
            self.wb_c(tf.concat([_icd, self._h_geom], axis=-1)); self.wb_v(self._h_geom)
        if self.loose_bank:                              # build lb_* (live in call(); operator stashed _h_geom)
            self.lb_bm(self.lb_h1(self._loose_feats(tf.zeros((1, H, W, self.n_channels)))))
            self.lb_tw(self.lb_t1(tf.zeros((1, 7))))
        if self.phase_bank:                              # build pb_* (reads stashed _h_geom like lb_*)
            self.pb_v(self._h_geom[:1])
            self.pb_g(self.pb_h1(tf.zeros((1, 4 * self.phase_bank_S + 2))))
        if self.adv_bank:                                # build av_* (reads stashed _h_geom)
            self.av_V(self._h_geom[:1])
            self.av_tw(self.av_t1(tf.zeros((1, 7))))
        if self.char_bank:                               # build cb_* (reads stashed _h_geom like lb_*)
            _icd = tf.zeros((1, H, W, self.n_channels), dtype=self._h_geom.dtype)
            self.cb_c(tf.concat([_icd, self._h_geom[:1]], axis=-1))
            self.cb_m(tf.concat([_icd[..., :1], self._h_geom[:1]], axis=-1))
            self.cb_g(self.cb_h1(tf.zeros((1, H, W, 8 * self.n_channels + 2))))

    def _geom_features(self, geo_in):
        h = geo_in
        for layer in self.geo:
            h = layer(periodic_pad_2d(h, 1))
        return h

    def _regime_features(self, ic):
        """Per-channel regime statistics of the observed field (B,H,W,C) → (B, 4C) for the field gate.
        Equation-agnostic + regime-discriminative: amplitude (std) and length-scales (mean|∇u|, mean
        |Δu|) encode the dimensionless regime (|∇u|/std ∝ inverse length-scale ∝ Re-like). Computed
        on the last observed frame; cheap finite differences, no FFT (XLA-safe)."""
        x = tf.cast(ic, tf.float32)
        dx = self._dx
        mean = tf.reduce_mean(x, axis=[1, 2])                              # (B,C)
        std = tf.sqrt(tf.math.reduce_variance(x, axis=[1, 2]) + 1e-8)      # (B,C)
        grad = tf.reduce_mean(tf.sqrt(d_dx(x, dx) ** 2 + d_dy(x, dx) ** 2 + 1e-12), axis=[1, 2])
        curv = tf.reduce_mean(tf.abs(laplacian(x, dx, dx)), axis=[1, 2])   # (B,C)
        return tf.concat([mean, std, grad, curv], axis=-1)                # (B, 4C)

    def operator(self, u0, descriptor, coeffs, metric=None, geom_mask=None, training=False, target=None):
        """u0:(B,H,W)|(B,H,W,C)|(B,T_in,H,W,C)  geom_mask:(B,H,W,1) fluid=1/solid=0 (None→ones).
        Returns (W_flat:(B,H*W*C,N_p), ic_flat:(B,H*W*C))."""
        cfg = self.cfg
        if len(u0.shape) == 3:
            u0 = u0[..., None]                                         # (B,H,W,1)
        if len(u0.shape) == 5:                                        # (B,T_in,H,W,C) multi-frame
            C = int(u0.shape[-1]); Tin = int(u0.shape[1])
            # ic_first (full-trajectory supervision, NTO-ADA style): anchor z(0) at the FIRST input
            # frame so the continuous ADA trajectory spans the WHOLE window [0,T_final] (input interval
            # + future) and is supervised on ALL frames (the input frames become reconstruction targets
            # via the dynamics). Default (False) = legacy: z(0)=last input frame, future-only.
            ic = u0[:, 0] if getattr(self.cfg, "ic_first", False) else u0[:, -1]
            ctx_in = tf.reshape(tf.transpose(u0, [0, 2, 3, 1, 4]),
                                [tf.shape(u0)[0], cfg.Nx, cfg.Nx, Tin * C])  # frames→channels
            # time_deriv_feat: the physics bank is all SPATIAL (Δ,∇,−u·∇,ω,δ); add the OBSERVED initial time
            # derivative ∂u/∂t ≈ (last − prev input frame) as an EXPLICIT extra encoder-input channel-block so
            # the model gets the current rate-of-change directly (vs inferring it from the raw frame channels).
            # Reaches the expert via h_geom. Multi-frame only (T_in≥2); IVP T_in=1 has no time history.
            if bool(getattr(self.cfg, "time_deriv_feat", False)) and Tin >= 2:
                ctx_in = tf.concat([ctx_in, u0[:, -1] - u0[:, -2]], axis=-1)
            # bank_time_deriv: the operator-bank STATE z0 is normally just the last input frame, so the
            # spatial operators (Δ,∇,−u·∇) are BLIND to the input time-history — and buoyancy (dropped buo_y)
            # is an ACCELERATION that only shows up across frames. Feed the observed ∂u/∂t ≈ (last − prev)
            # INTO z0 (via ic_enc) so the bank state itself carries the rate-of-change → operators can build
            # a buoyancy/source tendency. (Distinct from time_deriv_feat, which only reaches the ENCODER.)
            ic_bank = ic
            if bool(getattr(self.cfg, "bank_full_history", False)):
                # FULL history: z0 encodes ALL Tin frames (Tin*C channels) → the operators act on a state that
                # summarizes the whole input trajectory (richest temporal info; ic_enc 1×1 mixes frames per
                # pixel). Richer than the 2-frame deriv, but z0 is now an abstract summary (advection of it is
                # heuristic, not a literal current-state operator). Overrides bank_time_deriv.
                ic_bank = tf.reshape(tf.transpose(u0, [0, 2, 3, 1, 4]),
                                     [tf.shape(u0)[0], cfg.Nx, cfg.Nx, Tin * C])   # (B,H,W,Tin*C)
            elif bool(getattr(self.cfg, "bank_time_deriv", False)) and Tin >= 2:
                # n consecutive backward first-differences Δ_k = u[-k]−u[-k-1] (k=1..n). n=1 → last + ∂u/∂t.
                # n≥2 → two+ velocity-rates at successive times so the bank can infer ∂²u/∂t² (ACCELERATION =
                # buoyancy), which one difference can't give. Needs Tin ≥ n+1.
                _ndv = int(getattr(self.cfg, "bank_deriv_n", 1))
                diffs = [u0[:, -k] - u0[:, -k - 1] for k in range(1, _ndv + 1) if Tin >= k + 1]
                ic_bank = tf.concat([ic] + diffs, axis=-1)                # (B,H,W,(1+#diffs)·C)
        else:                                                         # (B,H,W,C) single frame (IVP)
            C = int(u0.shape[-1]); ic = u0; ctx_in = u0; ic_bank = ic
        B = tf.shape(ic)[0]
        coords = tf.broadcast_to(self._coords, [B, cfg.Nx, cfg.Nx, 2])
        if metric is None:
            metric = tf.ones([B, cfg.Nx, cfg.Nx, 2], tf.float32) * self._dx
        elif len(metric.shape) == 2:                                  # (B,2) per-sample dx
            metric = metric[:, None, None, :] * tf.ones([B, cfg.Nx, cfg.Nx, 2], tf.float32)
        if geom_mask is None:
            geom_mask = tf.ones([B, cfg.Nx, cfg.Nx, 1], tf.float32)   # no geometry → Π = identity
        # no_coords: DROP the absolute (x,y) channels from the encoder input. On our uniform PERIODIC
        # grids the coord grid is a fixed spatial pattern, identical across every sample — it carries no
        # sample-discriminative signal and BREAKS the translation-equivariance that periodic-conv experts
        # would otherwise have (lets the net memorize absolute position, which can't transfer for a
        # translation-invariant PDE). Dropping it makes the whole encoder translation-equivariant. (metric
        # & geom_mask are spatially-uniform constants → kept; they don't break equivariance and geom_mask
        # still drives Π.) The general non-uniform/stretched-grid path (DESIGN.md §4.6) keeps coords.
        # ORDER-B physics front-end: compute the physics operators on the CLEAN physical IC (real velocity for
        # advection, real scalar for buoyancy) and PREPEND them to the encoder input, so the transformer
        # processes [raw fields + physics features] — the architecture-level inductive bias acts at the PHYSICAL
        # scale (raw fields) BEFORE the global learned mixing, rather than on the transformer's abstract latent
        # (which entangles the channels). Operators evaluated once on the IC (no recursion); ADA handles time.
        if getattr(cfg, "physics_frontend", False):
            mdx = metric[..., 0:1]; mdy = metric[..., 1:2]
            icp = tf.cast(ic, tf.float32)                                  # clean physical IC (B,H,W,C)
            ux = icp[..., 0:1]; uy = icp[..., 1:2]                         # REAL velocity slots
            lap = laplacian(icp, mdx, mdy)                                 # Δ (diffusion)
            gx = d_dx(icp, mdx); gy = d_dy(icp, mdy)                       # ∇
            adv = -(ux * minmod_dx(icp, mdx) + uy * minmod_dy(icp, mdy))   # −(u·∇) advection (real velocity)
            omega = d_dy(ux, mdy) - d_dx(uy, mdx)                          # vorticity
            div = d_dx(ux, mdx) + d_dy(uy, mdy)                            # divergence
            phys = tf.concat([lap, gx, gy, adv, omega, div], axis=-1)      # physics features on clean fields
            ctx_in = tf.concat([ctx_in, tf.cast(phys, ctx_in.dtype)], axis=-1)  # → into the encoder input
        if getattr(cfg, "learned_frontend", False):
            # LEARNED bias-head front-end (Order B): bias-typed convs on the clean raw IC → learned features
            icp = tf.cast(ic, tf.float32)
            t = icp
            for conv, d in zip(self.lfe_dil, (1, 2, 4)):
                t = conv(periodic_pad_2d(t, d))                            # transport / multi-scale (learned)
            pt = self.lfe_pt(icp)                                          # pointwise / source (learned)
            lf = tf.concat([t, pt], axis=-1)
            ctx_in = tf.concat([ctx_in, tf.cast(lf, ctx_in.dtype)], axis=-1)  # → into the encoder input
        if getattr(cfg, "no_coords", False):
            geo_in = tf.concat([metric, geom_mask, ctx_in], axis=-1)
        else:
            geo_in = tf.concat([coords, metric, geom_mask, ctx_in], axis=-1)  # (B,H,W,5+Tin*C)
        if getattr(self, "_spatial_fourier", 0) > 0:                            # + explicit spatial-phase basis
            geo_in = tf.concat([geo_in, tf.broadcast_to(self._coords_ff,
                                 [B, cfg.Nx, cfg.Nx, 4 * self._spatial_fourier])], axis=-1)
        h_geom = self._geom_features(geo_in)
        if self.regime_film:                                          # regime-conditioned FiLM (no router)
            h_geom = self.regime_film_mod(self._regime_features(ic), h_geom)

        # --- latent-grid: run the EXPENSIVE per-panel expert convs at a coarse latent resolution
        # (Nx/LF)² instead of 128², then upsample the panel dynamics W back to 128². This is the
        # scalability lever — the O(N_p) rollout activation (5 experts × N_p panels) is what blows up
        # memory at 150M; computing it at e.g. 32² cuts that ~LF² (16×) so fp32 150M N_p=64 fits with
        # no memory tricks. The state z, W storage, ADA basis, hard-IC and output stay at 128² (so the
        # hard-IC is exact; only the dynamics are computed coarse and bilinearly upsampled). LF=1 → no-op.
        LF = int(getattr(cfg, "latent_factor", 1))
        if LF > 1:
            h_geom_x = tf.nn.avg_pool2d(h_geom, LF, LF, "VALID")       # (B, Nx/LF, Nx/LF, d_geom)
            metric_x = tf.nn.avg_pool2d(metric, LF, LF, "VALID") * float(LF)  # coarse grid → dx×LF
        else:
            h_geom_x, metric_x = h_geom, metric

        if self._router_on:
            # optional grid message passing enriches the router input with multi-scale local connectivity
            r_in = self.grid_mp(h_geom_x) if self.router_mp else h_geom_x
            # router_descriptor: per-family per-expert logit bias from the descriptor, added DIRECTLY to the
            # route logits inside the router (not drowned in the 1024-ch input). Breaks diffusive-collapse.
            rbias = self.desc_route_bias(tf.cast(descriptor, tf.float32)) if self.router_descriptor else None
            # attn router returns softmax probs (Σ_K=1) directly; legacy conv router → independent sigmoid
            route = self.router(r_in, bias=rbias) if self.attn_router else tf.sigmoid(self.router(r_in))
            self._last_alpha = tf.reduce_mean(route, axis=[1, 2])      # (B,K) pooled — interp/load-balance
            alpha = None
        else:
            route = None
            alpha = (self.gate(self._regime_features(ic), descriptor, training=training)  # regime→α (B,K)
                     if self.field_gate else self.gate(descriptor))    # or label-conditioned (B,K)
            self._last_alpha = alpha

        # per-pixel/per-channel Fourier-vs-Legendre selector (split_basis only)
        mix_map = tf.sigmoid(self.mix_head(h_geom)) if self.split_basis else None  # (B,H,W,C)
        self._h_geom = h_geom                                          # stash for the warp head (used in call())
        # oc = expert output channels; Cw = channels each basis integrates (latent D, or physical C).
        if self.latent_decoder:
            oc = self.n_latent; Cw = self.n_latent
        else:
            oc = (2 * C) if self.split_basis else C; Cw = C
        self._ic_phys = tf.cast(ic, tf.float32)                        # physical IC for the residual decode
        if getattr(self, "shock_adapt", False):                        # shock sensor from IC velocity gradient
            icp = tf.reshape(self._ic_phys, [tf.shape(self._ic_phys)[0], cfg.Nx, cfg.Nx, -1])
            icv = icp[..., :2]                                         # velocity (vx,vy) @128²
            gx = icv - tf.roll(icv, 1, axis=2); gy = icv - tf.roll(icv, 1, axis=1)
            sens = tf.reduce_sum(tf.abs(gx) + tf.abs(gy), axis=-1, keepdims=True)  # |∇u| (B,Nx,Nx,1)
            self._shock_sens = sens / (tf.reduce_max(sens, axis=[1, 2], keepdims=True) + 1e-6)  # →[0,1]

        # compute dtype = 'bfloat16' under mixed_bfloat16 policy, else 'float32'. Cast every expert
        # input to it so the custom rollout math (fp32 state/derivatives ↔ bf16 conv outputs) never
        # mixes dtypes; the panel W is cast back to fp32 for accumulation + the fp32 ADA basis.
        cd = tf.keras.mixed_precision.global_policy().compute_dtype

        def _panel(zc, h, step_i, cp=None):
            zc = tf.cast(zc, cd)                                       # state at full 128²
            if LF > 1:
                zc_p = tf.nn.avg_pool2d(zc, LF, LF, "VALID")          # slot-preserving coarse state (operators)
                if self.enc_expand:                                    # + depthwise-expanded high-freq features
                    zc = tf.concat([zc_p, tf.cast(self.dw_down(zc), cd)], axis=-1)  # overcomplete; slots 0,1 intact
                else:
                    zc = zc_p
            hg = tf.cast(h_geom_x, cd); cf = tf.cast(coeffs, cd); mt = tf.cast(metric_x, cd)
            if self.step_embed:                                        # ① condition experts on panel index
                se = tf.gather(self._step_tbl, step_i)                # (d_step,)
                sp = tf.cast(self.step_proj(se[None, :]), cd)         # (1,d_geom), zero-init → 0 at start
                hg = hg + sp[:, None, None, :]                         # broadcast over space
            if self.w_feedback and cp is not None:                     # ③ all experts' previous-panel W
                # stop_gradient: the previous-panel W is a FORWARD conditioning signal only — do NOT
                # backprop through the 64-panel feedback recurrence (that BPTT path is what can explode;
                # the integrated-state z carry stays differentiable). wfb_proj still learns to USE the
                # (detached) previous W from the current panel's gradient.
                cpc = tf.reshape(tf.stop_gradient(tf.cast(cp, cd)),    # (B,lat,lat,oc,K) → (B,lat,lat,oc·K)
                                 [tf.shape(cp)[0], cfg.Nx // LF, cfg.Nx // LF, oc * len(self.experts)])
                hg = hg + tf.tanh(tf.cast(self.wfb_proj(cpc), cd))    # bounded ±1 → no amplification loop
            if self.unified_expert:
                # transformer-as-ENCODER: `route` holds per-pixel encoder FEATURES (encode_dim=d_geom, no
                # softmax). Every expert gets the SAME full encoder context (NO route gating → no collapse).
                # 1 expert → UnifiedExpert (all ops in one); ≥2 experts → all experts always-on, summed
                # (operator-splitting WITHOUT routing — each sees full context). hg keeps step/wfb conditioning.
                h_enc = tf.cast(route, cd) + hg
                if getattr(self, '_sfe', 0) > 0:
                    h_enc = h_enc + tf.cast(self.pos_proj(self._coords_ff_lat), cd)  # PE skip -> expert
                ctxu = {"h_geom": h_enc, "coeffs": cf, "metric": mt}
                # stack all experts (1 → single UnifiedExpert; ≥2 → all-experts-no-routing), summed.
                # always set contribs (the _panel return feeds ③ w_feedback; ignored if off).
                if getattr(self, '_msada', None):                       # MULTI-SCALE ADA: independent experts per scale
                    _lf0 = cfg.Nx // LF; _acc = 0.0; contribs = None
                    for _i, _s in enumerate(self._msada):
                        _r = _lf0 // int(_s)
                        if _r > 1:
                            _zc = tf.cast(tf.nn.avg_pool2d(tf.cast(zc, tf.float32), _r, _r, "VALID"), cd)
                            _hg = tf.cast(tf.nn.avg_pool2d(tf.cast(h_enc, tf.float32), _r, _r, "VALID"), cd)
                            _mt = tf.cast(tf.nn.avg_pool2d(tf.cast(mt, tf.float32), _r, _r, "VALID"), cd) * float(_r)
                        else:
                            _zc, _hg, _mt = zc, h_enc, tf.cast(mt, cd)
                        _c = tf.stack([tf.cast(e(_zc, {"h_geom": _hg, "coeffs": cf, "metric": _mt}), cd) for e in self.experts], axis=-1)
                        _ws = tf.cast(self.W_scale, cd) * tf.reduce_sum(_c, axis=-1)
                        _wsu = _ws if _r == 1 else tf.cast(tf.image.resize(tf.cast(_ws, tf.float32), [_lf0, _lf0], method="bilinear"), cd)
                        if getattr(self, 'msada_dec', None) is not None:   # scale-matched learned decode
                            _wsu = tf.cast(self.msada_dec[_i](_wsu), cd)
                        _acc = _acc + tf.cast(self._msada_w[_i], cd) * _wsu
                        if _r == 1: contribs = _c
                    w = _acc
                    if contribs is None: contribs = _c
                else:
                    contribs = tf.stack([tf.cast(e(zc, ctxu), cd) for e in self.experts], axis=-1)
                    w = tf.cast(self.W_scale, cd) * tf.reduce_sum(contribs, axis=-1)
            elif self._router_on:
                # TRUE input dispatch: the router decides how much of the encoded input h_geom flows
                # INTO each expert — expert k sees route_k(x)·h_geom as its conditioning (not the shared
                # h_geom), then contributions are summed (NO output gate). expert .out is zero-init so
                # W starts ≈0 → stable start without the neg-bias output-gate trick.
                rt = tf.cast(route, cd)                                # (B, Nx/LF, Nx/LF, K)
                # route-aware expert input: every expert also sees HOW the router allocated across ALL
                # experts → a shared expert can disambiguate the family-regime it serves and specialize.
                if self.route_film:                                    # FiLM(γ,β) modulation (preferred)
                    hgk = self.route_film_mod(rt, hg)                  # identity at init (zero-init)
                elif self.route_context:                               # additive embedding (legacy)
                    hgk = hg + tf.cast(self.route_ctx(route), cd)
                else:
                    hgk = hg
                hgk = tf.cast(hgk, cd)                                 # bf16: route_film_mod/hg may be fp32 → match rt (cd)
                contribs = tf.stack(
                    [self.experts[k](zc, {"h_geom": rt[..., k:k + 1] * hgk, "coeffs": cf, "metric": mt})
                     for k in range(len(self.experts))], axis=-1)      # (B,·,·,oc,K) at latent
                contribs = tf.cast(contribs, cd)                       # experts use fp32 Dense → match cd ops below
                if self.learned_combine:
                    w = tf.reduce_sum(contribs * tf.cast(self.omega, cd), axis=-1)  # Σ_k ω_{c,k}·B_k
                else:
                    w = tf.cast(self.W_scale, cd) * tf.reduce_sum(contribs, axis=-1)
            else:                                                      # global output gating
                ctx = {"h_geom": hg, "coeffs": cf, "metric": mt}
                contribs = tf.stack([e(zc, ctx) for e in self.experts], axis=-1)
                contribs = tf.cast(contribs, cd)                       # experts use fp32 Dense → match cd ops below
                w = tf.cast(self.W_scale, cd) * tf.reduce_sum(
                    tf.cast(alpha, cd)[:, None, None, None, :] * contribs, axis=-1)
            if LF > 1:                                                 # dynamics back to 128²
                if getattr(self, 'tcu', False):                         # windowed cross-attn upsample (kv=coarse state zc)
                    w = self.wtcu(w, zc)
                elif getattr(self, '_ms', None):                        # MULTI-SCALE pyramid upsample (energy cascade)
                    _lf0 = cfg.Nx // LF
                    _outs = []
                    for _s, _up in zip(self._ms, self.ms_up):
                        _wp = w if int(_s) >= _lf0 else tf.cast(tf.nn.avg_pool2d(tf.cast(w, tf.float32), _lf0 // int(_s), _lf0 // int(_s), "VALID"), cd)
                        _outs.append(tf.nn.depth_to_space(tf.cast(_up(_wp), cd), cfg.Nx // int(_s)))
                    w = tf.cast(self.ms_mix(tf.cast(tf.concat(_outs, -1), cd)), cd)
                elif self.pure_upsample:                                 # learned-only (no bilinear low-pass)
                    w = tf.nn.depth_to_space(tf.cast(self.w_up(tf.cast(w, cd)), cd), LF)
                else:
                    w_bil = tf.cast(tf.image.resize(w, [cfg.Nx, cfg.Nx], method="bilinear"), cd)
                    if self.learned_upsample:                          # + learned high-freq residual (zero-init)
                        w_hf = tf.nn.depth_to_space(tf.cast(self.w_up(tf.cast(w, cd)), cd), LF)
                        if self.shock_adapt:                           # adaptive resolution: amplify hf at shocks
                            sc = tf.cast(self.shock_scale(tf.cast(self._shock_sens, cd)), cd)  # zero-init → 0
                            w_hf = w_hf * (1.0 + sc)
                        w = w_bil + tf.cast(w_hf, cd)
                    else:
                        w = w_bil
            if getattr(self, '_msdec', None):                          # MULTI-SCALE correlation synthesis on 128² W
                _o = []
                for _s, _c in zip(self._msdec, self.msdec_c):
                    _p = w if int(_s) >= cfg.Nx else tf.cast(tf.nn.avg_pool2d(tf.cast(w, tf.float32), cfg.Nx // int(_s), cfg.Nx // int(_s), "VALID"), cd)
                    _f = tf.cast(_c(_p), cd)
                    _u = _f if int(_s) >= cfg.Nx else tf.cast(tf.image.resize(tf.cast(_f, tf.float32), [cfg.Nx, cfg.Nx], method="bilinear"), cd)
                    _o.append(_u)
                w = tf.cast(self.msdec_mix(tf.cast(tf.concat(_o, -1), cd)), cd)
            # return (Π·w @128², per-expert contribs @latent) — contribs feeds ③ next panel; ignored if off.
            return tf.cast(geom_mask, cd) * w, contribs                # Π: freeze solid regions (128²)
        panel_fn = tf.recompute_grad(_panel) if getattr(cfg, "grad_checkpoint", False) else _panel

        # --- tf.while_loop rollout: ONE compiled loop body instead of N_p unrolled copies →
        # O(1) graph build + XLA/jit-compatible at any N_p (Python-unroll exploded the graph at
        # 100M). Loop carry kept in float32 for stability; expert compute may be bf16 (mixed
        # precision) — cast each panel back to fp32 before accumulating. ---
        npix = cfg.Nx * cfg.Nx * Cw                                    # Cw = D (latent) or C (physical)
        N_p = int(cfg.N_p)
        # parallel_w: DROP the Euler state-feedback (z stays z0 for every panel) → panels are INDEPENDENT
        # (no recurrence, no BPTT through z) → run in parallel + no swap_memory. Per-panel variation comes
        # from step_embed; nonlinearity/history is delegated to the (full-T_in) transformer encoder. The
        # operator bank is then evaluated at the IC (z0) only, and the ADA basis integrates the panel W's.
        _pw = bool(getattr(cfg, "parallel_w", False))
        # pw_batched: when parallel_w (z fixed at z0 → panels independent) AND the unified-encoder path,
        # REPLACE the N_p-iteration while_loop with a SINGLE batched expert call (panel index folded into the
        # batch dim). The latent grid is tiny (32²); running 64 separate panel kernels under-utilizes the GPU
        # and pays 64× launch overhead. One (B·N_p, lat, lat, ·) conv fuses into a single efficient kernel.
        # Math is IDENTICAL to the parallel_w while_loop (z=z0 for every panel; the only per-panel variation
        # is step_embed). Requires unified_expert + no w_feedback (a recurrence cannot be batched).
        _pwb = _pw and self.unified_expert and (not self.w_feedback) and bool(getattr(cfg, "pw_batched", False))
        es = [None, cfg.Nx, cfg.Nx, Cw]
        Wf_ta = tf.TensorArray(tf.float32, size=N_p, element_shape=es)
        Wl_ta = tf.TensorArray(tf.float32, size=N_p, element_shape=es)
        # rollout starting state z0: latent → learned 1×1 encode of the physical IC (D ch); else the IC.
        # rollout carry stays fp32 (stability); ic_enc may output bf16 under mixed precision → cast back.
        if self.latent_decoder and self.history_augment and len(u0.shape) == 5:
            # z0 = [current-frame encode | learned K-frame history features] → bank sees recent observed trend
            cur = tf.cast(self.ic_enc(tf.cast(ic_bank, tf.float32)), tf.float32)
            K = min(self.history_k, int(u0.shape[1]))
            hist_stack = tf.reshape(tf.transpose(u0[:, -K:], [0, 2, 3, 1, 4]),
                                    [tf.shape(u0)[0], cfg.Nx, cfg.Nx, K * C])   # last K frames → channels
            hist = tf.cast(self.hist_enc(tf.cast(hist_stack, tf.float32)), tf.float32)
            z0 = tf.concat([cur, hist], axis=-1)
        elif self.latent_decoder and self.slot_structured:
            # z0 = [physical channels (clean) | learned extra] → operators read real velocity/scalar
            z0 = tf.concat([tf.cast(ic, tf.float32),                           # first C slots = physical IC
                            tf.cast(self.ic_enc(tf.cast(ic_bank, tf.float32)), tf.float32)], axis=-1)  # extra
        else:
            z0 = (tf.cast(self.ic_enc(tf.cast(ic_bank, tf.float32)), tf.float32)   # ic_bank = ic [+ ∂u/∂t if banktd]
                  if self.latent_decoder else tf.cast(ic, tf.float32))

        def _rollout_pw_batched(z_st_override=None):
            # parallel_w + unified: all N_p panels in ONE batched expert call (panel → batch dim). z fixed
            # at z0; per-panel variation = step_embed only → mirrors _panel's `if self.unified_expert` branch.
            Bd = tf.shape(z0)[0]
            zc = tf.cast(z0, cd)
            if LF > 1:                                                  # coarse state (operators), once
                zc_p = tf.nn.avg_pool2d(zc, LF, LF, "VALID")
                zc = tf.concat([zc_p, tf.cast(self.dw_down(zc), cd)], axis=-1) if self.enc_expand else zc_p
            se = tf.gather(self._step_tbl, tf.range(N_p))               # (N_p, d_step) — all panel embeds
            sp = tf.cast(self.step_proj(se), cd)                        # (N_p, d_geom), zero-init → 0 at start
            h_enc = tf.cast(route, cd) + tf.cast(h_geom_x, cd)          # (B, lat, lat, d) encoder ctx
            if getattr(self, '_sfe', 0) > 0:
                h_enc = h_enc + tf.cast(self.pos_proj(self._coords_ff_lat), cd)  # PE skip -> expert
            h_enc_all = tf.repeat(h_enc, N_p, axis=0)                   # (B·N_p, lat, lat, d), b-major blocks
            sp_til = tf.tile(sp, [Bd, 1])                               # (B·N_p, d_geom) matches b-major order
            h_enc_all = h_enc_all + sp_til[:, None, None, :]            # ① per-panel step conditioning
            zc_all = tf.repeat(zc, N_p, axis=0)
            mt_all = tf.repeat(tf.cast(metric_x, cd), N_p, axis=0)
            cf_all = tf.repeat(tf.cast(coeffs, cd), N_p, axis=0)
            ctxu = {"h_geom": h_enc_all, "coeffs": cf_all, "metric": mt_all}
            # FIXED-POINT: re-evaluate the operator bank on the W-RECONSTRUCTED evolving state, K parallel
            # iterations. z(t_i) = z0 + Σ_{j<i} W_j·dt (exclusive cumsum = the recursive Euler state, but
            # parallel over panels). Each iter recomputes the bank on the evolved state → advection −(u·∇)z
            # now uses the EVOLVED velocity (the recursion's PDEArena gain) WITHOUT the sequential loop.
            # fixedpoint_iters=0 → pure pw_batched (bank on z0 only). Fully differentiable (K small → no BPTT
            # blow-up). enc_expand OFF in this chain → zc has exactly Cw=n_latent channels (clean cumsum).
            _fp = int(getattr(cfg, "fixedpoint_iters", 0))
            latf = cfg.Nx // LF
            if getattr(self, '_sf', 1) > 1:                       # SUPER-RES: upsample expert inputs to SF*latf
                _hi = latf * self._sf
                zc_all = tf.cast(tf.image.resize(tf.cast(zc_all, tf.float32), [_hi, _hi], method='bilinear'), cd)
                _hg = tf.cast(tf.image.resize(tf.cast(ctxu['h_geom'], tf.float32), [_hi, _hi], method='bilinear'), cd)
                _mt = tf.cast(tf.image.resize(tf.cast(ctxu['metric'], tf.float32), [_hi, _hi], method='nearest'), cd) / float(self._sf)
                ctxu = {'h_geom': _hg, 'coeffs': ctxu['coeffs'], 'metric': _mt}
            if z_st_override is not None:                               # TEACHER-FORCED: bank sees GT panel states
                z_st = z_st_override; _fp = 0                           # one parallel eval, no fixedpoint reconstruction
            else:
                z_st = zc_all                                          # iter 0: every panel sees z0
            for _it in range(_fp + 1):
                contribs = tf.stack([tf.cast(e(z_st, ctxu), cd) for e in self.experts], axis=-1)
                w = tf.cast(self.W_scale, cd) * tf.reduce_sum(contribs, axis=-1)   # (B·N_p, lat, lat, oc)
                if getattr(self, '_sf', 1) > 1:                       # learned downsample SF*latf -> latf (Nx)
                    w = tf.cast(self.w_down(tf.cast(w, cd)), cd)
                if _it < _fp:                                           # reconstruct evolved states for next iter
                    w_p = tf.reshape(tf.cast(w, tf.float32), [Bd, N_p, latf, latf, Cw])
                    csum = tf.cumsum(w_p, axis=1, exclusive=True)       # Σ_{j<i} W (left-Riemann / Euler)
                    if bool(getattr(cfg, "fixedpoint_trapezoid", False)):   # 2nd-order: + ½(W_i − W_0)
                        csum = csum + 0.5 * (w_p - w_p[:, :1])          # trapezoidal rule ∫_0^{t_i} W dt
                    cum = tf.cast(self._dt, tf.float32) * csum          # ·dt
                    z_ev = tf.cast(zc, tf.float32)[:, None] + cum       # z(t_i)=z0+∫_0^{t_i}W·dt (B,N_p,lat,lat,Cw)
                    z_st = tf.cast(tf.reshape(z_ev, [Bd * N_p, latf, latf, Cw]), cd)
                    if bool(getattr(cfg, "fixedpoint_stopgrad", False)):  # detach reconstruction → cheaper
                        z_st = tf.stop_gradient(z_st)                    # backward + breaks W→state→W amplification
            if self.panel_attn:                                         # ②' couple panels (parallel, latent res)
                lat = cfg.Nx // LF; ocw = w.shape[-1]
                w = tf.reshape(w, [Bd, N_p, lat, lat, ocw])             # (B,N_p,lat,lat,oc)
                w = self.panel_attn_layer(w)                            # bidirectional panel self-attn (zero-init)
                w = tf.reshape(w, [Bd * N_p, lat, lat, ocw])
            if self.panel_advect:                                       # advective W-panel coupling (semi-Lagrangian)
                lat = cfg.Nx // LF; ocw = w.shape[-1]
                w5 = tf.reshape(w, [Bd, N_p, lat, lat, ocw])            # (B,N_p,lat,lat,oc)
                u_ic = zc[..., 0:2]                                     # IC velocity (state slots 0,1), coarse grid
                w = tf.reshape(self.apc_layer(w5, u_ic), [Bd * N_p, lat, lat, ocw])
            if LF > 1:                                                  # dynamics back to 128²
                if getattr(self, '_ms', None):                          # MULTI-SCALE pyramid upsample (energy cascade)
                    _lf0 = cfg.Nx // LF
                    _outs = []
                    for _s, _up in zip(self._ms, self.ms_up):
                        _wp = w if int(_s) >= _lf0 else tf.cast(tf.nn.avg_pool2d(tf.cast(w, tf.float32), _lf0 // int(_s), _lf0 // int(_s), "VALID"), cd)
                        _outs.append(tf.nn.depth_to_space(tf.cast(_up(_wp), cd), cfg.Nx // int(_s)))
                    w = tf.cast(self.ms_mix(tf.cast(tf.concat(_outs, -1), cd)), cd)
                elif self.pure_upsample:                                  # learned-only (no bilinear low-pass)
                    w = tf.nn.depth_to_space(tf.cast(self.w_up(tf.cast(w, cd)), cd), LF)
                else:
                    w_bil = tf.cast(tf.image.resize(w, [cfg.Nx, cfg.Nx], method="bilinear"), cd)
                    if self.learned_upsample:
                        w_hf = tf.nn.depth_to_space(tf.cast(self.w_up(tf.cast(w, cd)), cd), LF)
                        if self.shock_adapt:
                            sc = tf.cast(self.shock_scale(tf.cast(tf.repeat(self._shock_sens, N_p, axis=0), cd)), cd)
                            w_hf = w_hf * (1.0 + sc)
                        w = w_bil + tf.cast(w_hf, cd)
                    else:
                        w = w_bil
            w = tf.cast(tf.repeat(tf.cast(geom_mask, cd), N_p, axis=0), cd) * w   # Π freeze solids (128²)
            w_all = tf.reshape(tf.cast(w, tf.float32), [Bd, N_p, cfg.Nx, cfg.Nx, -1])  # (B,N_p,128,128,oc)
            if self.split_basis:
                wf_all, wl_all = w_all[..., :C], w_all[..., C:]
            else:
                wf_all = wl_all = w_all
            Wf_h = tf.transpose(wf_all, [0, 2, 3, 4, 1])                # (B,128,128,Cw,N_p)
            Wl_h = tf.transpose(wl_all, [0, 2, 3, 4, 1])
            return Wf_h, Wl_h

        def _teacher_states(tgt):
            # TEACHER FORCING: encode the GROUND-TRUTH trajectory at the N_p panel times → latent bank states
            # z_st (B·N_p, latf, latf, Cw), b-major panel-minor (matching zc_all / h_enc_all order). The bank's
            # W is then learned on the TRUE evolving state in one parallel pass. tgt is in the SAME (normalized)
            # space as ic (tgt[:,0]==ic==u0[:,-1]). GT is a constant target → stop_gradient (learn only W's map).
            Bb = tf.shape(tgt)[0]; Tf = float(cfg.T_final); Ntt = int(cfg.Nt)
            tp = tf.range(N_p, dtype=tf.float32) * (Tf / float(N_p))    # panel times i·dt over [0,T_final)
            fpos = tf.clip_by_value(tp * (float(Ntt) - 1.0) / Tf, 0.0, float(Ntt) - 1.0)  # fractional GT-frame idx
            i0 = tf.minimum(tf.cast(tf.floor(fpos), tf.int32), Ntt - 2)
            wge = (fpos - tf.cast(i0, tf.float32))[None, :, None, None, None]             # (1,N_p,1,1,1)
            g0 = tf.gather(tgt, i0, axis=1); g1 = tf.gather(tgt, i0 + 1, axis=1)          # (B,N_p,Nx,Nx,C)
            gp = (1.0 - wge) * g0 + wge * g1                            # GT interpolated to panel times
            gp = tf.stop_gradient(tf.reshape(gp, [Bb * N_p, cfg.Nx, cfg.Nx, C]))         # b-major panel-minor
            if self.slot_structured:                                   # mirror z0 = [physical | ic_enc(extra)]
                zg = tf.concat([gp, tf.cast(self.ic_enc(gp), tf.float32)], axis=-1)
            else:
                zg = tf.cast(self.ic_enc(gp), tf.float32)
            zg = tf.cast(zg, cd)
            if LF > 1:
                zg = tf.nn.avg_pool2d(zg, LF, LF, "VALID")             # coarse latent state, matches zc
            return zg

        # recursive_2nd: replace the forward-Euler z-update z+=W·dt with the 2nd-order explicit ADAMS-BASHFORTH
        # (AB2): z_{i+1}=z_i+dt·(1.5·W_i − 0.5·W_{i-1}) — the sequential-rollout analog of trapezoidal (true
        # trapezoidal needs the unavailable W_{i+1}; AB2 uses the carried previous W, 2nd-order, NO extra bank
        # eval). i=0 falls back to Euler (coef 1.0, 0.0). Carries w_prev in the loop. Only the recursive path.
        _r2 = bool(getattr(cfg, "recursive_2nd", False))

        def _cond(i, z, wp, fa, la):
            return i < N_p

        def _body(i, z, wp, fa, la):
            w_i, _ = panel_fn(z, h_geom, i)                            # (B,H,W,oc); i = panel-step index
            w_i = tf.cast(w_i, tf.float32)
            cc = tf.where(i > 0, 1.5, 1.0) if _r2 else 1.0             # AB2 coefs (i=0 → Euler)
            cp = tf.where(i > 0, -0.5, 0.0) if _r2 else 0.0
            if self.split_basis:
                wf, wl = w_i[..., :C], w_i[..., C:]                    # Fourier-W, Legendre-W
                fa = fa.write(i, wf); la = la.write(i, wl)
                w_eff = wf + wl                                        # effective dz/dt (matches z channels)
            else:
                fa = fa.write(i, w_i); la = la.write(i, w_i)          # shared-W: both bases see same W
                #   (must write la too — an unwritten TensorArray loop var breaks tf2xla conversion)
                w_eff = w_i
            if not _pw:
                if self.recursive_gru:                                # LEARNED gated update (keeps bank, learns integrator)
                    cat = tf.concat([z, w_eff], axis=-1)
                    u = tf.cast(self.gru_u(cat), tf.float32)          # update gate
                    r = tf.cast(self.gru_r(cat), tf.float32)          # reset gate
                    cand = tf.cast(self.gru_c(tf.concat([w_eff, r * z], axis=-1)), tf.float32)  # zero-init → 0
                    z = z + u * cand                                  # gated residual; cand=0 at start → z'=z
                else:
                    z = z + self._dt * (cc * w_eff + cp * wp)         # Euler (cc=1,cp=0) or AB2 (_r2)
            return [i + 1, z, w_eff, fa, la]                          # carry effective tendency as w_prev

        # swap_memory: offload the N_p panels' forward activations to HOST RAM and bring them back in
        # the backward pass — fp32-exact (no math change), trades GPU memory for CPU↔GPU transfer, so
        # the O(N_p) rollout activation stack no longer has to live on the GPU. Lets fp32 + big N_p +
        # big width fit (the path to ~160M without bf16). Requires jit_compile=False (XLA ignores it).
        _swap = bool(getattr(cfg, "swap_memory", False))
        # TEACHER-FORCED training: bank evaluated on GT panel states in ONE parallel call (BCAT-cheap training).
        # Only when training + target available + unified + no w_feedback. Eval/inference (target=None) falls
        # through to the sequential while_loop below (the recursive model).
        _tf_active = (self.teacher_forced and training and (target is not None)
                      and self.unified_expert and (not self.w_feedback))
        if _tf_active:
            Wf_hwcp, Wl_hwcp = _rollout_pw_batched(z_st_override=_teacher_states(target))
        elif _pwb:
            Wf_hwcp, Wl_hwcp = _rollout_pw_batched()                   # single batched call — no while_loop
        elif self.w_feedback:
            # ③ carry the previous panel's per-expert contribs (B,lat,lat,oc,K) so every expert sees all
            # experts' last W. cp0=0 → first panel identical to the no-feedback start (+ zero-init conv).
            lat = cfg.Nx // LF
            cp0 = tf.zeros([B, lat, lat, oc, len(self.experts)], tf.float32)

            def _cond_fb(i, z, cp, fa, la):
                return i < N_p

            def _body_fb(i, z, cp, fa, la):
                w_i, contribs = panel_fn(z, h_geom, i, cp)
                w_i = tf.cast(w_i, tf.float32); contribs = tf.cast(contribs, tf.float32)
                # parallel_w (_pw) + w_feedback = ALTERNATIVE-B "W-history recursion": keep z FIXED at z0
                # (drop the Euler state-feedback → no redundant double-integration of W, no baked-in 1st-order
                # scheme) but STILL recurse on the W history (cp = previous panels' W feeds the next operator
                # via wfb_proj). Recursion lives in W-space, the single ADA integration produces the solution.
                if self.split_basis:
                    wf, wl = w_i[..., :C], w_i[..., C:]
                    fa = fa.write(i, wf); la = la.write(i, wl)
                    if not _pw:
                        z = z + (wf + wl) * self._dt
                else:
                    fa = fa.write(i, w_i); la = la.write(i, w_i)
                    if not _pw:
                        z = z + w_i * self._dt
                return [i + 1, z, contribs, fa, la]

            _, z, _, Wf_ta, Wl_ta = tf.while_loop(
                _cond_fb, _body_fb, [tf.constant(0), z0, cp0, Wf_ta, Wl_ta],
                parallel_iterations=1, maximum_iterations=N_p, swap_memory=_swap)
        else:
            _, z, _, Wf_ta, Wl_ta = tf.while_loop(
                _cond, _body, [tf.constant(0), z0, tf.zeros_like(z0), Wf_ta, Wl_ta],  # w_prev carry (AB2)
                parallel_iterations=(N_p if _pw else 1),               # parallel_w: panels independent → run concurrently
                maximum_iterations=N_p, swap_memory=(False if _pw else _swap))
        # stack → (N_p,B,H,W,C) → (B,H,W,C,N_p) → [② spatial mix] → (B,npix,N_p)
        if not (_pwb or _tf_active):                                   # while_loop path: stack TensorArrays
            Wf_hwcp = tf.transpose(Wf_ta.stack(), [1, 2, 3, 4, 0])     # (B,H,W,C,N_p); _pwb / teacher-forced set it already
            if self.split_basis:
                Wl_hwcp = tf.transpose(Wl_ta.stack(), [1, 2, 3, 4, 0])
            if self.panel_attn:                                       # ②' couple panels post-rollout (recursive
                # path: W already at 128². (B,H,W,C,N_p)→(B,N_p,H,W,C)→attn→back. Lets the recursive/fixedpoint
                # while_loop path ALSO get the panel-axis coupling (_pwb does it at latent res inside the loop).
                Wf_hwcp = tf.transpose(self.panel_attn_layer(tf.transpose(Wf_hwcp, [0, 4, 1, 2, 3])), [0, 2, 3, 4, 1])
                if self.split_basis:
                    Wl_hwcp = tf.transpose(self.panel_attn_layer(tf.transpose(Wl_hwcp, [0, 4, 1, 2, 3])), [0, 2, 3, 4, 1])
        if self.w_spatial:
            Wf_hwcp = self._w_spatial_mix(Wf_hwcp)                     # neighbor context before basis
        Wf = tf.reshape(Wf_hwcp, [B, npix, N_p])
        if self.split_basis:
            if self.w_spatial:
                Wl_hwcp = self._w_spatial_mix(Wl_hwcp)
            Wl = tf.reshape(Wl_hwcp, [B, npix, N_p])
        else:
            Wl = Wf
        # latent_decoder: residual ansatz → basis integrates the latent W from ZERO (the physical IC is
        # added back after decoding). else: hard-IC anchored in the basis (g(0)=IC).
        if self.latent_decoder:
            ic_flat = tf.zeros([B, npix], tf.float32)
        else:
            ic_flat = tf.reshape(tf.cast(ic, tf.float32), [B, npix])
        if self.split_basis:
            mix_flat = tf.reshape(tf.cast(mix_map, tf.float32), [B, npix])
        elif self.adaptive_mix:                                       # per-pixel & per-channel Fourier↔Legendre mix
            ml = tf.cast(self.mix_head_lat(h_geom), tf.float32)       # (B,Nx,Nx,Cw) full res, zero-init
            m_px = tf.sigmoid(tf.cast(self.mix_logit, tf.float32) + ml)   # start = sigmoid(mix_logit) everywhere
            mix_flat = tf.reshape(m_px, [B, npix])                    # (B,npix) order (h,w,c) matches Wf reshape
        else:
            mix_flat = None
        return Wf, Wl, mix_flat, ic_flat

    def _w_spatial_mix(self, w):                                       # ② w:(B,H,W,C,N_p) → same shape
        B = tf.shape(w)[0]; H = self.cfg.Nx
        C = int(w.shape[3]); Np = int(w.shape[4])
        x = tf.reshape(w, [B, H, H, C * Np])
        d = self.w_dw(periodic_pad_2d(x, 1))                           # depthwise 3×3, zero-init → 0 start
        return tf.reshape(x + d, [B, H, H, C, Np])                     # residual neighbor mixing

    def evolve(self, Wf, Wl, mix_flat, ic_flat):
        ga = self.basis_adaf(Wf, ics=[ic_flat])["g1"]                  # Fourier integ of own W (B,N_pix,Nt)
        gl = self.basis_lpa(Wl, ics=[ic_flat])["g1"]                   # Legendre integ of own W
        if mix_flat is not None:                                      # split_basis OR adaptive_mix: per-pixel mix
            m = mix_flat[:, :, None]                                   # (B,N_pix,1) (already sigmoid'd)
        else:
            m = tf.sigmoid(self.mix_logit)
        g = m * ga + (1.0 - m) * gl
        return tf.transpose(g, [0, 2, 1])                              # (B,Nt,N_pix)

    def _loose_feats(self, icf):
        """loose_bank physics feature stack (B,H,W,11C+17): state powers (reaction), Δ/Δ² and c̃²-weighted
        Laplacians (diffusion/wave), hidden v0 + c̃²Δv0 (2nd-order time), |∇u|/κ|∇u|/∇c̃²·∇u (fronts,
        curvature flow, layered-c interfaces), squeezed encoder features. PIXEL-space derivatives; physical
        scales are absorbed by the learned c̃² and the downstream combiner."""
        icf = tf.cast(icf, tf.float32)
        hgd = self._h_geom
        c2 = tf.nn.softplus(tf.cast(self.lb_c2(tf.concat([tf.cast(icf, hgd.dtype), hgd], -1)), tf.float32))
        v0 = tf.cast(self.lb_v(hgd), tf.float32)
        hp = tf.cast(self.lb_hp(hgd), tf.float32)
        lap1 = laplacian(icf, 1.0, 1.0); lap2 = laplacian(lap1, 1.0, 1.0)
        lv = c2 * laplacian(v0, 1.0, 1.0)
        gx = d_dx(icf, 1.0); gy = d_dy(icf, 1.0)
        nrm = tf.sqrt(gx * gx + gy * gy + 1e-8)
        kap = (d_dx(gx / nrm, 1.0) + d_dy(gy / nrm, 1.0)) * nrm
        gc = d_dx(c2, 1.0) * gx + d_dy(c2, 1.0) * gy
        return tf.concat([icf, icf * icf, icf ** 3, lap1, lap2, c2, c2 * lap1, v0, lv, nrm, kap, gc, hp], -1)

    def _warp(self, img, disp):
        """Periodic backward bilinear warp (semi-Lagrangian transport of the IC).
        img:(B,Nx,Nx,C), disp:(B,Nt,Nx,Nx,2) [dx,dy in pixels] → (B,Nt,Nx,Nx,C): value at x,t = img(x−disp)."""
        Nx = self.cfg.Nx; Nt = self.cfg.Nt; C = int(img.shape[-1]); B = tf.shape(img)[0]
        gy, gx = tf.meshgrid(tf.range(Nx, dtype=tf.float32), tf.range(Nx, dtype=tf.float32), indexing="ij")  # (Nx,Nx)
        sx = gx[None, None] - disp[..., 0]                            # (B,Nt,Nx,Nx) departure col (x)
        sy = gy[None, None] - disp[..., 1]                            # departure row (y)
        sx = tf.math.floormod(sx, float(Nx)); sy = tf.math.floormod(sy, float(Nx))   # periodic wrap
        x0 = tf.floor(sx); y0 = tf.floor(sy); wx = sx - x0; wy = sy - y0
        x0i = tf.cast(x0, tf.int32) % Nx; x1i = (x0i + 1) % Nx
        y0i = tf.cast(y0, tf.int32) % Nx; y1i = (y0i + 1) % Nx
        imgf = tf.reshape(img, [B, Nx * Nx, C])                       # (B,Nx²,C)
        def gth(yi, xi):
            v = tf.gather(imgf, tf.reshape(yi * Nx + xi, [B, -1]), batch_dims=1)  # (B,Nt·Nx²,C)
            return tf.reshape(v, [B, Nt, Nx, Nx, C])
        v00 = gth(y0i, x0i); v01 = gth(y0i, x1i); v10 = gth(y1i, x0i); v11 = gth(y1i, x1i)
        wx = wx[..., None]; wy = wy[..., None]                        # (B,Nt,Nx,Nx,1)
        return (1 - wy) * ((1 - wx) * v00 + wx * v01) + wy * ((1 - wx) * v10 + wx * v11)

    def call(self, u0, descriptor, coeffs, metric=None, geom_mask=None, training=False, target=None):
        C = int(u0.shape[-1]) if len(u0.shape) >= 4 else 1            # rank 4|5 → last dim is C
        Wf, Wl, mix_flat, ic_flat = self.operator(u0, descriptor, coeffs, metric, geom_mask, training, target)
        traj = self.evolve(Wf, Wl, mix_flat, ic_flat)                  # (B,Nt,H*W*Cw)
        B = tf.shape(u0)[0]; Nx = self.cfg.Nx; Nt = self.cfg.Nt
        if self.latent_decoder:
            # latent trajectory (B,Nt,H,W,D) → per-pixel no-bias MLP → physical δ (B,Nt,H,W,C); + IC.
            z = tf.reshape(traj, [B, Nt, Nx, Nx, self.n_latent])
            if getattr(self, "group_decode", False):           # per-desc-group readout of the shared bank
                _dg = tf.cast(descriptor, tf.float32)
                _mc = _dg[:, 0] * _dg[:, 1] * (1.0 - _dg[:, 2]) * (1.0 - _dg[:, 3]) * _dg[:, 4] * _dg[:, 5]
                _ms = _dg[:, 0] * (1.0 - _dg[:, 1]) * (1.0 - _dg[:, 2]) * (1.0 - _dg[:, 3]) * _dg[:, 4] * (1.0 - _dg[:, 5])
                _m0 = tf.nn.relu(1.0 - _mc - _ms)              # fallback: advective trio(+cfd) & unseen descs
                _msk = [tf.stop_gradient(m)[:, None, None, None, None] for m in (_m0, _mc, _ms)]
                delta = tf.add_n([_msk[_g] * tf.cast(self.gdec_out[_g](self.gdec_h[_g](z)), tf.float32)
                                  for _g in range(3)])
            else:
                delta = tf.cast(self.decode_out(self.decode_h(z)), tf.float32)  # (B,Nt,H,W,C); decode bf16→fp32
            ic_phys = tf.reshape(self._ic_phys, [B, Nx, Nx, C])        # physical IC (B,H,W,C)
            base = ic_phys[:, None]                                    # default anchor: ic constant over t
            if self.warp_head and self.warp_steps > 1 and not self.warp_off:  # VELOCITY-COUPLED COMPOSITIONAL semi-Lagrangian
                u = tf.cast(self.warp_v(self._h_geom), tf.float32)     # (B,Nx,Nx,2) velocity field (zero-init → d=0)
                K = self.warp_steps
                tjK = (tf.cast(tf.range(Nt), tf.float32) / float(max(Nt - 1, 1)))[None, :, None, None, None] / float(K)
                disp = tf.zeros([B, Nt, Nx, Nx, 2], tf.float32)        # backward-characteristic displacement d(x,t_j)
                for _ in range(K):                                     # trace: d ← d + u(x−d)·(t_j/K); curves the path
                    u_dep = self._warp(u, disp)                        # (B,Nt,Nx,Nx,2) velocity at the moving departure
                    disp = disp + u_dep * tjK
                base = self._warp(ic_phys, disp)                       # base(x,t)=IC(x−d_K) semi-Lagrangian transport
            elif self.warp_head and not self.warp_off:                 # legacy single Taylor warp (warp_steps=1)
                vh = tf.cast(self.warp_v(self._h_geom), tf.float32)    # (B,Nx,Nx,2) displacement velocity (zero-init)
                ah = tf.cast(self.warp_a(self._h_geom), tf.float32)    # (B,Nx,Nx,2) displacement accel
                tj = tf.cast(tf.range(Nt), tf.float32) / float(max(Nt - 1, 1))     # normalized t∈[0,1]
                tj = tj[None, :, None, None, None]
                disp = vh[:, None] * tj + 0.5 * ah[:, None] * (tj * tj)            # (B,Nt,Nx,Nx,2) d(x,t)
                base = self._warp(ic_phys, disp)                       # base(x,t)=IC(x−d) Lagrangian transport
            elif self.extrap_residual and len(u0.shape) == 5 and int(u0.shape[1]) >= 3:
                dv = u0[:, -1] - u0[:, -2]                             # observed velocity (1st diff)
                da = u0[:, -1] - 2.0 * u0[:, -2] + u0[:, -3]           # observed acceleration (2nd diff = forcing)
                va = tf.cast(self.ex_proj(tf.concat([dv, da], axis=-1)), tf.float32)  # (B,H,W,2C) zero-init→0
                v0, a0 = va[..., :C], va[..., C:]
                tj = tf.cast(tf.range(Nt), tf.float32) / float(max(Nt - 1, 1))        # normalized t∈[0,1]
                tj = tj[None, :, None, None, None]
                base = base + v0[:, None] * tj + 0.5 * a0[:, None] * (tj * tj)         # ic + v0·t + ½a0·t²
            if self.wave_bank:                                         # trainable wave propagator residual (see __init__)
                hg = self._h_geom
                c2 = tf.nn.softplus(tf.cast(self.wb_c(tf.concat([tf.cast(ic_phys, hg.dtype), hg], -1)), tf.float32))
                v0w = tf.cast(self.wb_v(hg), tf.float32)               # (B,H,W,C) hidden initial velocity
                tj = tf.cast(tf.range(Nt), tf.float32) / float(max(Nt - 1, 1))
                tj = tj[None, :, None, None, None]                     # (1,Nt,1,1,1) normalized t∈[0,1]
                Lu = tf.cast(ic_phys, tf.float32); Lv = v0w            # running L^k u0 / L^k v0 (PIXEL-space Δ;
                wres = self.wb_beta[0] * tj * v0w[:, None]             #  physical dx/dt scale absorbed by c̃²)
                for k in range(1, self.wave_bank_K + 1):
                    Lu = c2 * laplacian(Lu, 1.0, 1.0)
                    Lv = c2 * laplacian(Lv, 1.0, 1.0)
                    wres = wres + self.wb_alpha[k - 1] * (tj ** (2 * k)) * Lu[:, None] \
                                + self.wb_beta[k] * (tj ** (2 * k + 1)) * Lv[:, None]
                icf = tf.cast(ic_phys, tf.float32)                     # interface/reflection term ∇c̃²·∇u0
                gci = d_dx(c2, 1.0) * d_dx(icf, 1.0) + d_dy(c2, 1.0) * d_dy(icf, 1.0)
                base = base + wres + self.wb_gamma * (tj ** 2) * gci[:, None]
            if self.ace_bank:                                          # trainable reaction-diffusion propagator (see __init__)
                icf = tf.cast(ic_phys, tf.float32)
                tja = tf.cast(tf.range(Nt), tf.float32) / float(max(Nt - 1, 1))
                tja = tja[None, :, None, None, None]                   # (1,Nt,1,1,1) normalized t∈[0,1]
                s = tf.nn.softplus(self.ab_s)                          # closed-form bistable kinetics flow
                eat = tf.exp(tf.clip_by_value(self.ab_a * tja, -20.0, 20.0))       # (1,Nt,1,1,C)
                den = tf.sqrt(tf.maximum(1.0 + s * tf.square(icf[:, None]) * (eat * eat - 1.0), 1e-6))
                ares = self.ab_gr * (icf[:, None] * eat / den - icf[:, None])
                Ld = icf                                               # diffusion semigroup Taylor Σ δ_k t^k Δ^k u0
                for k in range(1, self.ace_bank_K + 1):
                    Ld = laplacian(Ld, 1.0, 1.0)
                    ares = ares + self.ab_delta[k - 1] * (tja ** k) * Ld[:, None]
                gx = d_dx(icf, 1.0); gy = d_dy(icf, 1.0)               # curvature front speed κ|∇u0|
                nrm = tf.sqrt(gx * gx + gy * gy + 1e-8)
                kap = d_dx(gx / nrm, 1.0) + d_dy(gy / nrm, 1.0)
                base = base + ares + self.ab_gk * tja * (kap * nrm)[:, None]
            if self.loose_bank:                                        # physics vocabulary + learned combiner
                feats = self._loose_feats(tf.cast(ic_phys, tf.float32))
                Bm = tf.cast(self.lb_bm(self.lb_h1(tf.cast(feats, self._h_geom.dtype))), tf.float32)
                Bm = tf.reshape(Bm, [B, Nx, Nx, self.loose_bank_M, C])  # basis maps B_m(x)
                tl = tf.cast(tf.range(Nt), tf.float32) / float(max(Nt - 1, 1))
                temb = tf.stack([tl, tl * tl, tl ** 3, tf.sin(np.pi * tl), tf.cos(np.pi * tl),
                                 tf.sin(2.0 * np.pi * tl), tf.cos(2.0 * np.pi * tl)], -1)   # (Nt,7)
                wt = tf.cast(self.lb_tw(self.lb_t1(temb)), tf.float32)  # (Nt,M) zero-init → 0 at load
                wt = wt * tl[:, None]                                   # ramp: residual(t=0) ≡ 0 (IC exact)
                base = base + tf.einsum("tm,bxymc->btxyc", wt, Bm)
            if self.phase_bank:                                        # loose spectral-phase vocabulary (see __init__)
                icf = tf.cast(ic_phys, tf.float32)
                v0p = tf.cast(self.pb_v(self._h_geom), tf.float32)     # (B,H,W,C) hidden initial velocity
                u0h = tf.signal.rfft2d(tf.transpose(icf, [0, 3, 1, 2]))     # (B,C,H,Wf) complex64
                v0h = tf.signal.rfft2d(tf.transpose(v0p, [0, 3, 1, 2]))
                half = Nx // 2
                ky = tf.cast(tf.concat([tf.range(0, half), tf.range(-half, 0)], 0), tf.float32) / float(half)
                kx = tf.cast(tf.range(0, half + 1), tf.float32) / float(half)
                kap = tf.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)          # (H,Wf) κ=|k|/k_nyq (ky-symmetric)
                tlp = tf.cast(tf.range(Nt), tf.float32) / float(max(Nt - 1, 1))
                kt = kap[None] * tlp[:, None, None]                          # (Nt,H,Wf) κ·τ
                cs = tf.exp(self.pb_logc); gs = tf.exp(self.pb_logg)
                ph = kt[..., None] * cs                                      # (Nt,H,Wf,S) phase carriers
                dmp1 = tf.exp(-kt[..., None] * gs)                           # transport damping e^{−gκτ}
                dmp2 = tf.exp(-(kap[None] ** 2 * tlp[:, None, None])[..., None] * gs * 16.0)  # viscous e^{−gκ²τ}
                Fv = tf.concat([tf.sin(ph), tf.cos(ph), dmp1, dmp2, kt[..., None],
                                tf.broadcast_to(kap[None, ..., None], tf.shape(kt[..., None]))], -1)
                G = tf.cast(self.pb_g(self.pb_h1(tf.cast(Fv, self._h_geom.dtype))), tf.float32)  # (Nt,H,Wf,2C)
                G = G * tlp[:, None, None, None]                             # ramp: residual(τ=0) ≡ 0 (IC exact)
                G1 = tf.transpose(G[..., :C], [0, 3, 1, 2])                  # (Nt,C,H,Wf)
                G2 = tf.transpose(G[..., C:], [0, 3, 1, 2])
                dh = (tf.complex(G1, tf.zeros_like(G1))[None] * u0h[:, None]
                      + tf.complex(G2, tf.zeros_like(G2))[None] * v0h[:, None])   # (B,Nt,C,H,Wf)
                dpb = tf.signal.irfft2d(dh, fft_length=[Nx, Nx])             # (B,Nt,C,H,W) real
                base = base + tf.transpose(dpb, [0, 1, 3, 4, 2])
            if self.adv_bank and not self.adv_bank_off:                # loose transport vocabulary (see __init__)
                icf = tf.cast(ic_phys, tf.float32)
                UV = tf.cast(self.av_V(self._h_geom), tf.float32)      # (B,H,W,2(M+1)): V0 | V_1..V_M
                tla = tf.cast(tf.range(Nt), tf.float32) / float(max(Nt - 1, 1))
                temba = tf.stack([tla, tla * tla, tla ** 3, tf.sin(np.pi * tla), tf.cos(np.pi * tla),
                                  tf.sin(2.0 * np.pi * tla), tf.cos(2.0 * np.pi * tla)], -1)   # (Nt,7)
                wta = tf.cast(self.av_tw(self.av_t1(temba)), tf.float32)        # (Nt,M) temporal mixture
                Kav, Mav = self.adv_bank_K, self.adv_bank_M
                tjK = tla[None, :, None, None, None] / float(Kav)
                disp = tf.zeros([B, Nt, Nx, Nx, 2], tf.float32)
                for _ in range(Kav):                                   # backward characteristics through ṽ(x,τ)
                    S = self._warp(UV, disp)                           # (B,Nt,H,W,2(M+1)) at moving departure
                    u_dep = S[..., :2] + tf.einsum("tm,btxymc->btxyc", wta,
                                                   tf.reshape(S[..., 2:], [B, Nt, Nx, Nx, Mav, 2]))
                    disp = disp + tf.clip_by_value(u_dep, -self.adv_bank_clip, self.adv_bank_clip) * tjK
                base = base + tf.tanh(self.av_gate) * (self._warp(icf, disp) - icf[:, None])
            if self.char_bank:                                         # characteristics + interface reflection (see __init__)
                icf = tf.cast(ic_phys, tf.float32)
                hgc = self._h_geom
                ctl = tf.nn.softplus(tf.cast(self.cb_c(tf.concat([tf.cast(icf, hgc.dtype), hgc], -1)), tf.float32))
                ctl = tf.tanh(ctl) * self.char_bank_smax               # (B,H,W,1) bounded displacement px / unit τ
                mif = tf.sqrt(d_dx(ctl, 1.0) ** 2 + d_dy(ctl, 1.0) ** 2 + 1e-8)   # interface indicator |∇c̃|
                mg = tf.sigmoid(tf.cast(self.cb_m(tf.concat([tf.cast(mif, hgc.dtype), hgc], -1)), tf.float32))
                tlc = tf.cast(tf.range(Nt), tf.float32) / float(max(Nt - 1, 1))
                ttc = tlc[None, :, None, None, None]                   # (1,Nt,1,1,1) normalized τ∈[0,1]
                carr = []
                for _ex, _ey in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
                    dspc = tf.stack([_ex * ctl[..., 0], _ey * ctl[..., 0]], -1)[:, None] * ttc  # (B,Nt,H,W,2)
                    carr.append(self._warp(icf, dspc))                 # one-way characteristic transport of IC
                Uc = tf.concat(carr, -1)                               # (B,Nt,H,W,4C)
                Fc = tf.concat([Uc, Uc * mg[:, None],
                                tf.broadcast_to(ctl[:, None] / self.char_bank_smax, tf.shape(Uc[..., :1])),
                                tf.broadcast_to(mg[:, None], tf.shape(Uc[..., :1]))], -1)       # (B,Nt,H,W,8C+2)
                Ff = tf.reshape(tf.cast(Fc, hgc.dtype), [B * Nt, Nx, Nx, 8 * C + 2])
                rc = tf.cast(self.cb_g(self.cb_h1(Ff)), tf.float32)    # zero-init combiner → 0 at load
                base = base + tf.reshape(rc, [B, Nt, Nx, Nx, C]) * ttc # ramp: residual(τ=0) ≡ 0 (IC exact)
            if self.buoyancy_bank and C >= 3:                      # by2: time-consistent Boussinesq source bank
                icb = tf.cast(ic_phys, tf.float32)
                if self.adv_bank and not self.adv_bank_off:
                    cwb = self._warp(icb, disp)                    # transported state at each output time (reuse
                else:                                              #  adv_bank characteristics; no extra cost)
                    cwb = tf.tile(icb[:, None], [1, Nt, 1, 1, 1])
                csl = cwb[..., 2:3]                                # transported passive scalar (slot 2)
                dgr = (tf.roll(csl, -1, 2) - tf.roll(csl, 1, 2)) * 0.5    # d/drow (both axes: convention-proof)
                dgc = (tf.roll(csl, -1, 3) - tf.roll(csl, 1, 3)) * 0.5    # d/dcol
                fbb = tf.reshape(tf.cast(tf.concat([csl, dgr, dgc], -1), self._h_geom.dtype), [B * Nt, Nx, Nx, 3])
                hbb = tf.cast(tf.reshape(self.bb_out(self.bb_h(fbb)), [B, Nt, Nx, Nx, 2]), tf.float32)
                gpool = tf.reduce_mean(self._h_geom, axis=[1, 2])          # (B,d) global input-window readout
                beta = 1.0 + tf.cast(self.bb_coef(gpool), tf.float32)      # per-trajectory magnitude (init 1)
                tlb = (tf.cast(tf.range(Nt), tf.float32) / float(max(Nt - 1, 1)))[None, :, None, None, None]
                resb = tf.reshape(beta, [-1, 1, 1, 1, 1]) * tlb * hbb      # ramp: residual(tau=0)==0 (hard-IC)
                base = base + tf.pad(resb, [[0, 0], [0, 0], [0, 0], [0, 0], [0, C - 2]])
            if self.lag_channel and C >= 3:                        # forward-Lagrangian particle transport (v1)
                Slag = self.lag_stride; Npp = Nx // Slag; Rlag = 2; Mlag = self.lag_modes
                icf = tf.cast(ic_phys, tf.float32)
                hs = self._h_geom[:, ::Slag, ::Slag, :]            # particle features (grid-aligned gather)
                ics = icf[:, ::Slag, ::Slag, :]
                u0p = tf.reverse(ics[..., 0:2], [-1])              # data ch0=row-vel, ch1=col-vel -> (col,row)=(dx,dy)
                c0p = ics[..., 2]                                  # carried passive scalar (slot 2)
                hh = self.lag_h1(tf.concat([hs, tf.cast(ics, hs.dtype)], -1))
                Ad = tf.cast(self.lag_Ad(hh), tf.float32)          # (B,Np,Np,2M) zero-init coeffs
                Ac = tf.cast(self.lag_Ac(hh), tf.float32)          # (B,Np,Np,M)
                tlg = tf.cast(tf.range(Nt), tf.float32) / float(max(Nt - 1, 1))
                Phi = tf.stack([tlg, tlg * tlg, tf.sin(np.pi * tlg), 1.0 - tf.cos(np.pi * tlg),
                                tf.sin(2.0 * np.pi * tlg), 1.0 - tf.cos(2.0 * np.pi * tlg)], -1)[:, :Mlag]  # (Nt,M) Phi(0)=0
                Ad = tf.reshape(Ad, [B, Npp, Npp, 2, Mlag])
                disp = (self.lag_alpha * u0p[:, None] * tlg[None, :, None, None, None]   # ballistic u0*tau
                        + tf.einsum("tm,bxycm->btxyc", Phi, Ad))   # + learned basis trajectory
                cpt = c0p[:, None] + tf.einsum("tm,bxym->btxy", Phi, Ac)                 # carried value c(tau)
                gy0, gx0 = tf.meshgrid(tf.range(Npp, dtype=tf.float32) * Slag,
                                       tf.range(Npp, dtype=tf.float32) * Slag, indexing="ij")
                pxl = tf.math.floormod(gx0[None, None] + disp[..., 0], float(Nx))
                pyl = tf.math.floormod(gy0[None, None] + disp[..., 1], float(Nx))
                fxl = tf.floor(pxl); fyl = tf.floor(pyl)
                inv2s2 = 1.0 / (2.0 * self.lag_sigma * self.lag_sigma)
                BT = B * Nt
                btl = tf.tile(tf.reshape(tf.range(BT, dtype=tf.int64), [BT, 1, 1]), [1, Npp, Npp])
                pxf = tf.reshape(pxl, [BT, Npp, Npp]); pyf = tf.reshape(pyl, [BT, Npp, Npp])
                fxf = tf.reshape(fxl, [BT, Npp, Npp]); fyf = tf.reshape(fyl, [BT, Npp, Npp])
                cfl = tf.reshape(cpt, [BT, Npp, Npp])
                idxs, wvals, cvals = [], [], []
                if self.lag_native:                                # BILINEAR 2x2: exact identity at tau=0
                    frx = pxf - fxf; fry = pyf - fyf
                    taps = [(0, 0), (0, 1), (1, 0), (1, 1)]
                else:
                    taps = [(dyl, dxl) for dyl in range(-Rlag, Rlag + 1) for dxl in range(-Rlag, Rlag + 1)]
                for dyl, dxl in taps:
                    xi = tf.math.floormod(fxf + dxl, float(Nx)); yi = tf.math.floormod(fyf + dyl, float(Nx))
                    if self.lag_native:
                        wsp = ((1.0 - frx) if dxl == 0 else frx) * ((1.0 - fry) if dyl == 0 else fry)
                    else:
                        wsp = tf.exp(-((pxf - (fxf + dxl)) ** 2 + (pyf - (fyf + dyl)) ** 2) * inv2s2)
                    flat = btl * (Nx * Nx) + tf.cast(yi, tf.int64) * Nx + tf.cast(xi, tf.int64)
                    idxs.append(tf.reshape(flat, [-1])); wvals.append(tf.reshape(wsp, [-1]))
                    cvals.append(tf.reshape(wsp * cfl, [-1]))
                idxl = tf.concat(idxs, 0)[:, None]
                accv = tf.tensor_scatter_nd_add(tf.zeros([BT * Nx * Nx], tf.float32), idxl, tf.concat(cvals, 0))
                accw = tf.tensor_scatter_nd_add(tf.zeros([BT * Nx * Nx], tf.float32), idxl, tf.concat(wvals, 0))
                chat = tf.reshape(accv / (accw + 1e-6), [B, Nt, Nx, Nx])
                mnorm = tf.reshape(accw / (accw + 0.5), [B, Nt, Nx, Nx])   # coverage conf -> Eulerian fallback in holes
                if self.lag_native and getattr(self, "lag_gate_mode", "desc") == "slots":
                    # IVP/transfer gate: unused slots are constant-filled by the loader -> spatial variance
                    # 0. Fire only where BOTH a carried scalar (slot 2) and a velocity field (slots 0,1)
                    # are alive: selects the tracer/NS families, excludes reaction/dispersive/steady tasks.
                    _ic0 = tf.cast(u0[:, -1] if len(u0.shape) == 5 else u0, tf.float32)   # (B,H,W,C)
                    _vs = tf.math.reduce_variance(_ic0, axis=[1, 2])                      # (B,C) per-slot
                    _sc = tf.reduce_max(_vs, axis=1, keepdims=True) + 1e-8
                    _alive = tf.cast(_vs > 1e-4 * _sc, tf.float32)                        # (B,C)
                    lag_m = tf.stop_gradient(_alive[:, 2] * _alive[:, 0] * _alive[:, 1])[:, None, None, None]
                    self._last_lag_m = lag_m
                    lag_chat = chat
                    lag_cov = mnorm
                elif self.lag_native:                          # NATIVE decode: stash; slot-2 replaced after field
                    _dsc = tf.cast(descriptor, tf.float32)
                    _dm = (_dsc[:, 0] * _dsc[:, 1] * _dsc[:, 3]
                           * (1.0 - _dsc[:, 2]) * (1.0 - _dsc[:, 4]) * (1.0 - _dsc[:, 5]))   # advective-trio desc
                    _s2 = tf.cast(u0[..., 2], tf.float32)          # (B,T_in,H,W) input slot-2 window
                    _tv = tf.reduce_mean(tf.math.reduce_variance(_s2, axis=1), axis=[1, 2])
                    _sv = tf.math.reduce_variance(tf.reduce_mean(_s2, axis=1), axis=[1, 2])
                    _md = tf.cast(_tv > 1e-3 * (_sv + 1e-8), tf.float32)   # dynamic slot-2 (excludes cfd static mask)
                    lag_m = tf.stop_gradient(_dm * _md)[:, None, None, None]
                    self._last_lag_m = lag_m
                    lag_chat = chat
                    lag_cov = mnorm                                # coverage confidence (per-pixel fallback)
                else:
                    glag = tf.tanh(tf.cast(self.lag_gate(descriptor), tf.float32))[:, :, None, None]  # per-family gate
                    res2 = glag * mnorm * (chat - chat[:, 0:1])    # motion residual; == 0 at tau=0 (hard-IC exact)
                    base = base + tf.pad(res2[..., None], [[0, 0], [0, 0], [0, 0], [0, 0], [2, C - 3]])
                if self.lag_omega:                             # v2: omega-particles + FFT-Poisson velocity
                    vx0 = icf[..., 1]; vy0 = icf[..., 0]           # ch1=col-vel(x), ch0=row-vel(y)
                    w0 = ((tf.roll(vy0, -1, 2) - tf.roll(vy0, 1, 2))
                          - (tf.roll(vx0, -1, 1) - tf.roll(vx0, 1, 1))) * 0.5   # curl(u0), periodic, px units
                    w0p = w0[:, ::Slag, ::Slag]
                    Aw = tf.cast(self.lag_Aw(hh), tf.float32)
                    _wc = tf.stop_gradient(2.0 * tf.reduce_max(tf.abs(w0p)) + 1e-6)
                    wpt = w0p[:, None] + tf.clip_by_value(tf.einsum("tm,bxym->btxy", Phi, Aw), -_wc, _wc)
                    wfl = tf.reshape(wpt, [-1])
                    # stop_gradient on splat weights: Poisson adjoint amplifies low-k grads ~ (Nx/2pi k)^2 ->
                    # a single finite-but-huge step poisons shared Ad/alpha (diverged @15k, @4k). The omega
                    # path trains only Aw/gate_w; trajectories keep learning through the scalar path.
                    accvw = tf.tensor_scatter_nd_add(tf.zeros([BT * Nx * Nx], tf.float32), idxl,
                                                     tf.concat([tf.stop_gradient(wv) * wfl for wv in wvals], 0))
                    what = tf.reshape(accvw, [BT, Nx, Nx]) / (tf.reshape(accw, [BT, Nx, Nx]) + 0.5)   # hole->0 (1e-6 denom spiked; Poisson amplifies low-k -> NaN @15k)
                    _wb = tf.stop_gradient(4.0 * tf.reduce_max(tf.abs(w0)) + 1e-6)
                    what = tf.clip_by_value(what, -_wb, _wb)                    # bound splat vorticity
                    whh = tf.signal.rfft2d(what)                                # (BT,Nx,Nx//2+1) complex
                    k1 = tf.cast(tf.concat([tf.range(0, Nx // 2), tf.range(-Nx // 2, 0)], 0), tf.float32) * (2.0 * np.pi / Nx)
                    k2 = tf.cast(tf.range(0, Nx // 2 + 1), tf.float32) * (2.0 * np.pi / Nx)
                    ksq = k1[:, None] ** 2 + k2[None, :] ** 2
                    inv = tf.where(ksq > 1e-12, 1.0 / tf.maximum(ksq, 1e-12), tf.zeros_like(ksq))
                    psih = whh * tf.complex(inv, tf.zeros_like(inv))            # lap psi = -w  ->  psi_h = w_h/|k|^2
                    kk1 = k1[:, None] * tf.ones_like(k2[None, :]); kk2 = tf.ones_like(k1[:, None]) * k2[None, :]
                    uxs = tf.signal.irfft2d(tf.complex(tf.zeros_like(kk1), kk1) * psih, fft_length=[Nx, Nx])
                    uys = tf.signal.irfft2d(tf.complex(tf.zeros_like(kk2), -kk2) * psih, fft_length=[Nx, Nx])
                    uvw = tf.reshape(tf.stack([uys, uxs], -1), [B, Nt, Nx, Nx, 2])   # slot0=row-vel, slot1=col-vel
                    gw = tf.tanh(tf.cast(self.lag_gate_w(descriptor), tf.float32))[:, :, None, None, None]
                    _rc = tf.stop_gradient(3.0 * tf.math.reduce_std(icf[..., 0:2]) + 1e-6)
                    resw = gw * tf.stop_gradient(mnorm)[..., None] * tf.clip_by_value(uvw - uvw[:, 0:1], -_rc, _rc)
                    base = base + tf.pad(resw, [[0, 0], [0, 0], [0, 0], [0, 0], [0, C - 2]])
            field = base + delta                                       # residual; field(0)=IC exact (t=0)
            if self.lag_channel and getattr(self, "lag_native", False) and C >= 3:
                fs2 = field[..., 2]                                # Eulerian slot-2
                rep = lag_cov * lag_chat + (1.0 - lag_cov) * fs2   # splat projection w/ coverage fallback
                fs2 = lag_m * rep + (1.0 - lag_m) * fs2            # NATIVE replacement (trio only)
                field = tf.concat([field[..., :2], fs2[..., None], field[..., 3:]], -1)
                if getattr(self, "omega_decode", False):       # physics-typed velocity (solenoidal increment)
                    dwv = tf.cast(self.decode_w(self.decode_h(z)), tf.float32)[..., 0]   # (B,Nt,H,W); ==0 at tau=0
                    _wc2 = tf.stop_gradient(4.0 * tf.math.reduce_std(dwv) + 1e-6)
                    dwv = tf.clip_by_value(dwv, -_wc2, _wc2)
                    _wh = tf.signal.rfft2d(tf.reshape(dwv, [B * Nt, Nx, Nx]))
                    _k1 = tf.cast(tf.concat([tf.range(0, Nx // 2), tf.range(-Nx // 2, 0)], 0), tf.float32) * (2.0 * np.pi / Nx)
                    _k2 = tf.cast(tf.range(0, Nx // 2 + 1), tf.float32) * (2.0 * np.pi / Nx)
                    _ks = _k1[:, None] ** 2 + _k2[None, :] ** 2
                    _inv = tf.where(_ks > 1e-12, 1.0 / tf.maximum(_ks, 1e-12), tf.zeros_like(_ks))
                    _ph = _wh * tf.complex(_inv, tf.zeros_like(_inv))
                    _kk1 = _k1[:, None] * tf.ones_like(_k2[None, :]); _kk2 = tf.ones_like(_k1[:, None]) * _k2[None, :]
                    _uc = tf.signal.irfft2d(tf.complex(tf.zeros_like(_kk1), _kk1) * _ph, fft_length=[Nx, Nx])   # col-vel
                    _ur = tf.signal.irfft2d(tf.complex(tf.zeros_like(_kk2), -_kk2) * _ph, fft_length=[Nx, Nx])  # row-vel
                    duv = tf.reshape(tf.stack([_ur, _uc], -1), [B, Nt, Nx, Nx, 2])   # slot0=row, slot1=col (verified)
                    ubar = tf.reduce_mean(delta[..., :2], axis=[2, 3], keepdims=True)   # k=0 mean flow (P drops it)
                    u_new = tf.cast(ic_phys, tf.float32)[:, None, :, :, :2] + duv + ubar
                    _m5 = lag_m[..., None]                                              # (B,1,1,1,1) for 5D blend
                    fuv = _m5 * u_new + (1.0 - _m5) * field[..., :2]                    # same trio mask
                    field = tf.concat([fuv, field[..., 2:]], -1)
        else:
            field = tf.reshape(traj, [B, Nt, Nx, Nx, C])
        if self.img_refine:                                            # per-frame 2D high-freq corrector
            xr = tf.reshape(field, [B * Nt, Nx, Nx, C])
            h = self.ref_d1(periodic_pad_2d(xr, 1))
            h = self.ref_d2(periodic_pad_2d(h, 2))
            h = self.ref_d4(periodic_pad_2d(h, 4))
            dlt = tf.cast(self.ref_out(periodic_pad_2d(h, 1)), tf.float32)  # zero-init → 0 at start
            field = field + tf.reshape(dlt, [B, Nt, Nx, Nx, C])
        if C == 1:
            field = field[..., 0]                                      # (B,Nt,H,W) back-compat
        return field

    def call_with_gate(self, u0, descriptor, coeffs, metric=None, geom_mask=None, training=False, target=None):
        """Returns (field, alpha) — alpha exposed for gate-sparsity regularization."""
        field = self.call(u0, descriptor, coeffs, metric, geom_mask, training, target)
        return field, self._last_alpha

    def gate_weights(self, descriptor):
        """Interpretability read-out: α_k per expert for a given descriptor."""
        return dict(zip(self.expert_names,
                        tf.unstack(self.gate(descriptor), axis=-1)))

    @property
    def trainable_variables(self):
        out = []
        for layer in [*self.geo, self.gate, *self.experts]:
            out.extend(list(layer.trainable_variables))
        if getattr(self, "_router_on", False):
            out.extend(list(self.router.trainable_variables))
        if getattr(self, "router_mp", False):
            out.extend(list(self.grid_mp.trainable_variables))
        if getattr(self, "learned_combine", False):
            out.append(self.omega)
        if getattr(self, "route_context", False):
            out.extend(list(self.route_ctx.trainable_variables))
        if getattr(self, "route_film", False):
            out.extend(list(self.route_film_mod.trainable_variables))
        if getattr(self, "step_embed", False):
            out.extend(list(self.step_proj.trainable_variables))
        if getattr(self, "w_spatial", False):
            out.extend(list(self.w_dw.trainable_variables))
        if getattr(self, "w_feedback", False):
            out.extend(list(self.wfb_proj.trainable_variables))
        if getattr(self, "panel_attn", False):
            out.extend(list(self.panel_attn_layer.trainable_variables))
        if getattr(self, "recursive_gru", False):
            for layer in [self.gru_u, self.gru_r, self.gru_c]:
                out.extend(list(layer.trainable_variables))
        if getattr(self, "extrap_residual", False):
            out.extend(list(self.ex_proj.trainable_variables))
        if getattr(self, "learned_frontend", False):
            for layer in [*self.lfe_dil, self.lfe_pt]:
                out.extend(list(layer.trainable_variables))
        if getattr(self, "latent_decoder", False):
            for layer in [self.ic_enc, self.decode_h, self.decode_out]:
                out.extend(list(layer.trainable_variables))
            if getattr(self, "group_decode", False):
                for _gl in self.gdec_h + self.gdec_out:
                    out.extend(list(_gl.trainable_variables))
            if getattr(self, "omega_decode", False):
                out.extend(list(self.decode_w.trainable_variables))
            if getattr(self, "history_augment", False):
                out.extend(list(self.hist_enc.trainable_variables))
        if (getattr(self, "learned_upsample", False) or getattr(self, "pure_upsample", False)) and hasattr(self, "w_up"):
            out.extend(list(self.w_up.trainable_variables))
        if getattr(self, "tcu", False) and hasattr(self, "wtcu"):
            out.extend(list(self.wtcu.trainable_variables))
        if getattr(self, "img_refine", False) and hasattr(self, "ref_out"):
            for layer in [self.ref_d1, self.ref_d2, self.ref_d4, self.ref_out]:
                out.extend(list(layer.trainable_variables))
        if getattr(self, "adaptive_mix", False):
            out.extend(list(self.mix_head_lat.trainable_variables))
        if getattr(self, "warp_head", False):
            out.extend(list(self.warp_v.trainable_variables) + list(self.warp_a.trainable_variables))
        if getattr(self, "wave_bank", False):
            out.extend(list(self.wb_c.trainable_variables) + list(self.wb_v.trainable_variables))
            out.extend([self.wb_alpha, self.wb_beta, self.wb_gamma])
        if getattr(self, "ace_bank", False):
            out.extend([self.ab_a, self.ab_s, self.ab_gr, self.ab_delta, self.ab_gk])
        if getattr(self, "loose_bank", False):
            for layer in [self.lb_c2, self.lb_v, self.lb_hp, self.lb_h1, self.lb_bm, self.lb_t1, self.lb_tw]:
                out.extend(list(layer.trainable_variables))
        if getattr(self, "phase_bank", False):
            for layer in [self.pb_v, self.pb_h1, self.pb_g]:
                out.extend(list(layer.trainable_variables))
            out.extend([self.pb_logc, self.pb_logg])
        if getattr(self, "adv_bank", False):
            for layer in [self.av_V, self.av_t1, self.av_tw]:
                out.extend(list(layer.trainable_variables))
            out.append(self.av_gate)
        if getattr(self, "lag_channel", False):
            for layer in [self.lag_h1, self.lag_Ad, self.lag_Ac, self.lag_gate]:
                out.extend(list(layer.trainable_variables))
            out.append(self.lag_alpha)
            if getattr(self, "lag_omega", False):
                out.extend(list(self.lag_Aw.trainable_variables) + list(self.lag_gate_w.trainable_variables))
        if getattr(self, "buoyancy_bank", False):
            for layer in [self.bb_h, self.bb_out, self.bb_coef]:
                out.extend(list(layer.trainable_variables))
        out.extend([self.mix_logit, self.W_scale])
        seen, uniq = set(), []
        for v in out:
            if id(v) not in seen:
                seen.add(id(v)); uniq.append(v)
        return uniq
