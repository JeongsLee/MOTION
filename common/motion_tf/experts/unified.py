"""Unified physics expert — the single-expert redesign (replaces the K-expert mixture + router, which
collapsed to diffusive-only with zero per-family differentiation). Instead of ROUTING the input to
separate experts (and letting a softmax gate pick one — which collapses), we compute ALL the structured
physics operators as a FEATURE BANK and let one head combine them per pixel/channel/regime. The
transformer is now an ENCODER (feeds per-pixel features here), not a router.

Operator bank (cross-channel where physical):
  • laplacian Δz                  — diffusion (per channel)
  • ∂x z, ∂y z                    — gradients (per channel)
  • −(u·∇)z (minmod upwind)       — advection by the velocity channels (transports every channel; the NS
                                    nonlinearity + passive-tracer transport that the dead convective lacked)
  • ω = ∂y vx − ∂x vy             — vorticity (solenoidal, cross-channel from velocity)
  • δ = ∂x vx + ∂y vy             — divergence (dilatational, cross-channel)
Then [bank, z, encoder-features] → dilated-conv stack (1,2,4,8, periodic → global RF, covers the elliptic
Δ⁻¹ / helmholtz global coupling) + global-mean broadcast → zero-init out. So one expert sees every
operator at once; the head weights them (no router, no collapse). Velocity = slots vel_slots (0,1).
"""
from __future__ import annotations
import tensorflow as tf
from .base import (Branch, periodic_pad_2d, d_dx, d_dy, laplacian,
                   minmod_dx, minmod_dy, upwind_dx, upwind_dy)


class UnifiedExpert(Branch):
    name_tag = "unified"

    def __init__(self, d_geom, hidden=64, out_ch=1, vel_slots=(0, 1), dilations=(1, 2, 4, 8),
                 use_ops=True, use_shock=False, shock_hidden=48,
                 multigrid=False, mg_levels=3, mg_hidden=None,
                 xattn=False, xattn_dim=256,
                 fno=False, fno_modes=12, fno_width=64,
                 reaction=False, reaction_hidden=32,
                 wave=False, wave_hidden=32, gradx=False, gradx_hidden=32,
                 adapter=False, adapter_hidden=48, adapter_dils=(1, 2, 4),
                 hibank=False, buoyancy=False, buoyancy_hidden=32,
                 advbias=False, advbias_hidden=32, fixedops=False, heads_off=(),
                 hsplit=0, hsplit_hidden=8, hsplit_ch=16, hsplit_shared=False, learnable_deriv=False, name="unified"):
        super().__init__(name=name)
        # heads_off: names of heads to SKIP in the forward while still CREATING their vars (positional
        # ckpt-load compatible) — finetune-time selective pruning of task-irrelevant physics vocabulary
        # (e.g. ("wave","buoyancy","gradx") for a steady elliptic task). Empty = normal behavior.
        self.heads_off = frozenset(heads_off)
        # fixedops=True → ADD a branch of FIXED, EXACT analytic physics operators (Δz diffusion, −(u·∇)z
        # advection, ∇·u divergence, R(u)=u−u³ Allen-Cahn reaction, c²Δz wave restoring using the fed wave
        # speed at slot6) with a single zero-init 1×1 conv mixing them into W. The OPERATORS are fixed/exact
        # (not learned); only their per-output-channel COEFFICIENTS are learned at finetune. zero-init out →
        # no-op at load (warm-starts identically from the learned-head pretrain), then finetune turns on the
        # task-appropriate equation terms — injecting exact PDE structure on top of the learned heads.
        self.fixedops = bool(fixedops)
        if self.fixedops:
            self.fx_out = tf.keras.layers.Conv2D(out_ch, 1, kernel_initializer="zeros", name="u_fx_out")
        # buoyancy=True → ADD a cross-channel SOURCE branch (Boussinesq-like): the scalar/state field (and
        # its vertical structure ∂_y) drives a tendency, i.e. "scalar → velocity" forcing — the REVERSE of
        # advection ("velocity → scalar transport") that the bank already has. With latent_decoder this acts
        # in latent space (a learned cross-channel source); the ∂_y term gives it the vertical asymmetry of
        # gravity. zero-init out → no-op at load. Targets buoyancy-driven families (conditioned NS, vertical
        # velocity / Vy) where the regime forcing is unrepresented (descriptor is intentionally unused).
        self.buoyancy = bool(buoyancy)
        if self.buoyancy:
            self.by_l1 = tf.keras.layers.Conv2D(int(buoyancy_hidden), 1, activation="gelu", name="u_by_l1")
            self.by_out = tf.keras.layers.Conv2D(out_ch, 1, kernel_initializer="zeros", name="u_by_out")
        # gradx=True → HORIZONTAL counterpart to buoyancy: a learned head on the ∂_x gradient (state + horizontal
        # gradient + geom). buoyancy(∂_y) specializes in the vertical buoyant force; gradx(∂_x) specializes in
        # horizontal gradient/shear/transport (Kelvin–Helmholtz shear ∂_x u_y, horizontal advection). Same
        # seed-into-learned-head pattern as buoyancy; zero-init out → no-op at load.
        self.gradx = bool(gradx)
        if self.gradx:
            self.gx_l1 = tf.keras.layers.Conv2D(int(gradx_hidden), 1, activation="gelu", name="u_gx_l1")
            self.gx_out = tf.keras.layers.Conv2D(out_ch, 1, kernel_initializer="zeros", name="u_gx_out")
        # advbias=True → ADD a learned UPWIND/directional ADVECTION-bias head (additive, works without use_ops):
        # sign-aware upwind advection feature −(u·∇^up)z (the paper's upwind-conv architectural bias) → learned
        # head → W. The upwind STRUCTURE is the bias; the contribution is learned. zero-init out → no-op start.
        self.advbias = bool(advbias)
        if self.advbias:
            self.av_h = tf.keras.layers.Conv2D(int(advbias_hidden), 1, activation="gelu", name="u_av_h")
            self.av_out = tf.keras.layers.Conv2D(out_ch, 1, kernel_initializer="zeros", name="u_av_out")
        # hibank=True → extend the analytic FD operator bank with HIGHER-ORDER / general-PDE features
        # (param-free, just more concat channels the conv head learns to weight): Δ² biharmonic
        # (Cahn-Hilliard/KS), ∂²xy mixed (anisotropic), ∂³x/∂³y (dispersive/KdV), |∇| (Hamilton-Jacobi/
        # level-set). Pretraining the head WITH this vocabulary lets OOD finetune re-weight to the operator
        # the new physics needs (vs lacking it entirely). No weights here.
        self.hibank = bool(hibank)
        self.vel_slots = tuple(vel_slots)
        self.dilations = tuple(dilations)
        # reaction=True → ADD a pointwise nonlinear-source branch R(u)+f (1×1 convs, zero receptive field,
        # GELU) summed into the output W. This is the reaction-diffusion inductive bias the operator bank
        # lacks (bank has Δ diffusion + advection but no pointwise R(u) like Allen-Cahn's u−u³). zero-init
        # out → no-op at load (cont base loads identically); finetune learns R. The expert output IS the
        # RHS contribution W (∂u/∂t = Σ W_i), and R(u) is literally a term of ∂u/∂t → physically the right spot.
        self.reaction = bool(reaction)
        if self.reaction:
            self.rx_l1 = tf.keras.layers.Conv2D(int(reaction_hidden), 1, activation="gelu", name="u_rx_l1")
            self.rx_l2 = tf.keras.layers.Conv2D(int(reaction_hidden), 1, activation="gelu", name="u_rx_l2")
            self.rx_out = tf.keras.layers.Conv2D(out_ch, 1, kernel_initializer="zeros", name="u_rx_out")
        # wave=True → ADD an explicit cross-channel GRADIENT-coupling branch (the WaveExpert operator the
        # bank's deep conv only implicitly approximates): [∂x all-ch, ∂y all-ch, z, geom] -> 1×1 mix forms
        # ANY linear cross-channel ∇ coupling (the wave restoring force: ∇ of one field drives another).
        # zero-init out → no-op at load. Targets the long-horizon wave-propagation weakness (late frames).
        self.wave = bool(wave)
        if self.wave:
            self.wv_mix = tf.keras.layers.Conv2D(int(wave_hidden), 1, activation="gelu", name="u_wv_mix")
            self.wv_out = tf.keras.layers.Conv2D(out_ch, 1, kernel_initializer="zeros", name="u_wv_out")
        # adapter=True → ADD a GENERIC learnable local-operator branch (NOT a hardcoded physics operator): a
        # small fresh dilated-conv stack on [z, geom] -> zero-init out. Δ/∇/advection are all special cases of
        # local convs, so this can represent ANY local operator the OOD physics needs — and being FRESH
        # (zero-init, not fluid-pretrained) it adapts without fighting the fluid-tuned main conv head. The
        # transfer-favorable lever: few new params + decoupled from the fluid prior. (option A)
        self.adapter = bool(adapter)
        if self.adapter:
            self.ad_dils = tuple(int(d) for d in adapter_dils)
            self.ad_convs = [tf.keras.layers.Conv2D(int(adapter_hidden), 3, padding="valid",
                                                    dilation_rate=d, activation="gelu", name=f"u_ad{d}")
                             for d in self.ad_dils]
            self.ad_out = tf.keras.layers.Conv2D(out_ch, 1, kernel_initializer="zeros", name="u_ad_out")
        # DUAL-BRANCH head: the LOCAL branch (dilated-conv below) handles the local differential operators
        # (Δ, ∇, −u·∇, shock flux); a NONLOCAL branch (multigrid V-cycle) handles the elliptic/pressure
        # GLOBAL coupling that the single global-mean scalar only weakly proxied. Both are PRESERVED and
        # summed → out. Non-spectral (real-space hierarchy → orthogonal to ADA time-Fourier), general
        # (multigrid for elliptic/parabolic; local branch keeps hyperbolic/shock & compressible), per-panel
        # on the EVOLVING state (the transformer encoder is 1-time/input-only → can't do this).
        self.multigrid = bool(multigrid)
        if self.multigrid:
            self.mg_levels = int(mg_levels)
            mgh = int(mg_hidden) if mg_hidden else min(int(hidden), 384)
            self.mg_pre = tf.keras.layers.Conv2D(mgh, 3, padding="valid", activation="gelu", name="u_mg_pre")
            self.mg_sd = [tf.keras.layers.Conv2D(mgh, 3, padding="valid", activation="gelu", name=f"u_mg_sd{l}")
                          for l in range(self.mg_levels)]
            self.mg_su = [tf.keras.layers.Conv2D(mgh, 3, padding="valid", activation="gelu", name=f"u_mg_su{l}")
                          for l in range(self.mg_levels)]
            self.mg_proj = tf.keras.layers.Conv2D(hidden, 1, kernel_initializer="zeros", name="u_mg_proj")
        # use_shock=True → ADD a shock-capturing feature to the bank: ∇·F̂ with a state-gated upwind blend
        # of left/right-biased learned flux kernels (Rankine–Hugoniot-consistent, for strong compressible
        # shocks — targets com_ns, the lean unified's one weak family). Modest shock_hidden so the bank
        # barely grows; stays a SINGLE expert (vs the bloated 8-expert sum that underperformed).
        self.use_shock = bool(use_shock)
        if self.use_shock:
            self.s_gate = tf.keras.layers.Conv2D(2, 1, kernel_initializer="zeros",
                                                 bias_initializer="zeros", name="u_s_gate")
            self.s_fxL = tf.keras.layers.Conv2D(shock_hidden, 3, padding="valid", name="u_s_fxL")
            self.s_fxR = tf.keras.layers.Conv2D(shock_hidden, 3, padding="valid", name="u_s_fxR")
            self.s_fyL = tf.keras.layers.Conv2D(shock_hidden, 3, padding="valid", name="u_s_fyL")
            self.s_fyR = tf.keras.layers.Conv2D(shock_hidden, 3, padding="valid", name="u_s_fyR")
        # use_ops=False → ABLATION: drop the hardcoded physics-operator bank (Δ,∇,−u·∇,ω,δ); the expert
        # sees only [state z, encoder features] → same dilated-conv head + ADA basis. Tests whether the
        # physics decomposition actually contributes, or the transformer-encoder + ADA basis alone suffice.
        # xattn=True → COMBINE the physics ops with the transformer features by CROSS-ATTENTION instead of
        # concat: each operator (Δ,∇x,∇y,−u·∇,ω,δ,z) is a per-pixel TOKEN, the transformer feature is the
        # QUERY, and a per-pixel softmax over the 7 operator tokens picks how much of each to apply WHERE
        # (soft, data-driven, per-pixel operator selection — unlike the collapsed hard router; non-spectral).
        self.xattn = bool(xattn)
        if self.xattn:
            self.x_dim = int(xattn_dim)
            self.n_optok = 7                                     # lap, gx, gy, adv, omega, div, z
            self.x_tok = [tf.keras.layers.Conv2D(self.x_dim, 1, name=f"u_x_tok{i}") for i in range(self.n_optok)]
            self.x_q = tf.keras.layers.Conv2D(self.x_dim, 1, name="u_x_q")
        # fno=True → add an FNO-style SPECTRAL global branch (2D FFT → keep low modes → learned complex
        # per-mode channel mixing → iFFT). Global, parameter-efficient. NOTE: ADA basis is TEMPORAL Fourier,
        # so this is "double-sin" spatial+temporal Fourier — empirical test of whether that clashes.
        self.fno = bool(fno)
        if self.fno:
            self.fno_modes = int(fno_modes); self.fno_w = int(fno_width)
            self.fno_lift = tf.keras.layers.Conv2D(self.fno_w, 1, name="u_fno_lift")
            _sc = 1.0 / self.fno_w
            self.fno_wr = tf.Variable(_sc * tf.random.normal((2, self.fno_modes, self.fno_modes, self.fno_w, self.fno_w)), name="u_fno_wr")
            self.fno_wi = tf.Variable(_sc * tf.random.normal((2, self.fno_modes, self.fno_modes, self.fno_w, self.fno_w)), name="u_fno_wi")
            self.fno_proj = tf.keras.layers.Conv2D(hidden, 1, kernel_initializer="zeros", name="u_fno_proj")
        self.use_ops = bool(use_ops)
        # HSPLIT — per-group latent differentiation. Instead of every operator reading the SAME latent z,
        # give operator GROUPS their own bottleneck-adapted latent  z_k = z + Up_k(gelu(Down_k(z)))  (Up is
        # zero-init → z_k == z at step 0, so the model is IDENTICAL to the shared-latent baseline at load and
        # the groups differentiate through training). 3 groups: transport {adv,omega,div + velocity},
        # diffusive {lap,gx,gy}, reaction/pointwise {raw z}. The differentiation pays off on the NONLINEAR
        # transport operators (advection/vorticity/divergence) which the downstream head can't linearly
        # absorb — targets the pdearena turbulence laggards. Down-project (hidden<ch) = a real bottleneck
        # forcing each group to select its own subspace. hsplit_shared=True → ONE shared adapter with 3x the
        # middle width (param-matched control): same parameter budget but a single shared subspace, so a win
        # for the split arm is attributable to per-group differentiation, not raw capacity.
        self.hsplit = int(hsplit)
        self.hsplit_shared = bool(hsplit_shared)
        if self.hsplit:
            _ch = int(hsplit_ch)

            def _mk(tag, hid):
                return (tf.keras.layers.Conv2D(int(hid), 1, activation="gelu", name=f"u_hs_{tag}1"),
                        tf.keras.layers.Conv2D(_ch, 1, kernel_initializer="zeros", name=f"u_hs_{tag}2"))
            if self.hsplit_shared:
                self.hs_S = _mk("S", int(hsplit_hidden) * 3)     # param-matched: one adapter, 3x middle
            else:
                self.hs_T = _mk("T", hsplit_hidden)              # transport latent (adv/omega/div + vel)
                self.hs_D = _mk("D", hsplit_hidden)              # diffusive latent (lap/gx/gy)
                self.hs_R = _mk("R", hsplit_hidden)              # reaction/pointwise latent (raw z)
        self.convs = [tf.keras.layers.Conv2D(hidden, 3, padding="valid", dilation_rate=d,
                                             activation="gelu", name=f"u_d{d}")
                      for d in self.dilations]
        self.gdense = tf.keras.layers.Dense(hidden, name="u_glob")
        # learnable_deriv: make the FD derivative operators LEARNABLE while PRESERVING their meaning.
        # 5-point roll stencils with learnable coeffs: d/dx = (a1*(z+1 - z-1) + a2*(z+2 - z-2))/dx  (antisym+
        # zero-sum by construction -> still a valid 1st derivative); Laplacian = (b1*(z+1+z-1-2z) + b2*(z+2+
        # z-2-2z))/dx^2 (symmetric+zero-sum -> valid 2nd derivative). FD-init (a1=.5,a2=0 ; b1=1,b2=0) => at
        # start identical to the fixed operators; training adapts the stencil (higher-order / anti-aliased /
        # learned eddy-viscosity closure for under-resolved turbulence). omega/div inherit via _ddx/_ddy.
        self.learnable_deriv = bool(learnable_deriv)
        if self.learnable_deriv:
            _C = int(hsplit_ch)                                       # per-channel (per latent field) 7-point stencils
            self.ld_dx  = tf.Variable(tf.tile(tf.constant([[0.5, 0.0, 0.0]]), [_C, 1]), trainable=True, name="u_ld_dx")
            self.ld_dy  = tf.Variable(tf.tile(tf.constant([[0.5, 0.0, 0.0]]), [_C, 1]), trainable=True, name="u_ld_dy")
            self.ld_lap = tf.Variable(tf.tile(tf.constant([[1.0, 0.0, 0.0]]), [_C, 1]), trainable=True, name="u_ld_lap")  # b1 per-ch = per-field eddy-viscosity
        self.out = tf.keras.layers.Conv2D(out_ch, 1, kernel_initializer="zeros", name="u_out")

    def _ddx(self, z, dx, slot=None):
        co = self.ld_dx if slot is None else self.ld_dx[slot:slot + 1]
        a1 = tf.cast(co[:, 0], z.dtype); a2 = tf.cast(co[:, 1], z.dtype); a3 = tf.cast(co[:, 2], z.dtype)
        d = (a1 * (tf.roll(z, -1, axis=1) - tf.roll(z, 1, axis=1))
             + a2 * (tf.roll(z, -2, axis=1) - tf.roll(z, 2, axis=1))
             + a3 * (tf.roll(z, -3, axis=1) - tf.roll(z, 3, axis=1)))
        return d / tf.cast(dx, z.dtype)

    def _ddy(self, z, dy, slot=None):
        co = self.ld_dy if slot is None else self.ld_dy[slot:slot + 1]
        a1 = tf.cast(co[:, 0], z.dtype); a2 = tf.cast(co[:, 1], z.dtype); a3 = tf.cast(co[:, 2], z.dtype)
        d = (a1 * (tf.roll(z, -1, axis=2) - tf.roll(z, 1, axis=2))
             + a2 * (tf.roll(z, -2, axis=2) - tf.roll(z, 2, axis=2))
             + a3 * (tf.roll(z, -3, axis=2) - tf.roll(z, 3, axis=2)))
        return d / tf.cast(dy, z.dtype)

    def _lap(self, z, dx, dy, slot=None):
        co = self.ld_lap if slot is None else self.ld_lap[slot:slot + 1]
        b1 = tf.cast(co[:, 0], z.dtype); b2 = tf.cast(co[:, 1], z.dtype); b3 = tf.cast(co[:, 2], z.dtype)
        lx = (b1 * (tf.roll(z, -1, axis=1) + tf.roll(z, 1, axis=1) - 2.0 * z)
              + b2 * (tf.roll(z, -2, axis=1) + tf.roll(z, 2, axis=1) - 2.0 * z)
              + b3 * (tf.roll(z, -3, axis=1) + tf.roll(z, 3, axis=1) - 2.0 * z))
        ly = (b1 * (tf.roll(z, -1, axis=2) + tf.roll(z, 1, axis=2) - 2.0 * z)
              + b2 * (tf.roll(z, -2, axis=2) + tf.roll(z, 2, axis=2) - 2.0 * z)
              + b3 * (tf.roll(z, -3, axis=2) + tf.roll(z, 3, axis=2) - 2.0 * z))
        return lx / tf.cast(dx, z.dtype) ** 2 + ly / tf.cast(dy, z.dtype) ** 2

    def contribution(self, z, ctx):
        if not self.use_ops:                                         # ABLATION: no physics ops
            bank = tf.concat([z, ctx["h_geom"]], axis=-1)
        else:
            dx = ctx["metric"][..., 0:1]
            dy = ctx["metric"][..., 1:2]
            # HSPLIT: differentiate the latent per operator group (z_k = z at init via zero-init Up).
            if self.hsplit and self.hsplit_shared:
                zc = z + self.hs_S[1](self.hs_S[0](z)); zT = zD = zR = zc
            elif self.hsplit:
                zT = z + self.hs_T[1](self.hs_T[0](z))              # transport
                zD = z + self.hs_D[1](self.hs_D[0](z))              # diffusive
                zR = z + self.hs_R[1](self.hs_R[0](z))              # reaction/pointwise
            else:
                zT = zD = zR = z
            sx, sy = self.vel_slots
            ux = zT[..., sx:sx + 1]                                  # transport velocity from zT
            uy = zT[..., sy:sy + 1]
            if self.learnable_deriv:                                 # learnable-FD (meaning-preserving); omega/div inherit
                lap = self._lap(zD, dx, dy); gx = self._ddx(zD, dx); gy = self._ddy(zD, dy)
                adv = -(ux * minmod_dx(zT, dx) + uy * minmod_dy(zT, dy))
                omega = self._ddy(ux, dy, slot=sx) - self._ddx(uy, dx, slot=sy); div = self._ddx(ux, dx, slot=sx) + self._ddy(uy, dy, slot=sy)
            else:
                lap = laplacian(zD, dx, dy)                              # diffusion (per channel), diffusive latent
                gx = d_dx(zD, dx)
                gy = d_dy(zD, dy)
                adv = -(ux * minmod_dx(zT, dx) + uy * minmod_dy(zT, dy))  # −(u·∇)z, stable 2nd-order TVD (transport)
                omega = d_dy(ux, dy) - d_dx(uy, dx)                      # vorticity (1ch, cross-channel)
                div = d_dx(ux, dx) + d_dy(uy, dy)                        # divergence (1ch)
            feats = [lap, gx, gy, adv, omega, div, zR, ctx["h_geom"]]
            if self.hibank:                                          # higher-order / general-PDE FD ops (param-free)
                # high-order stencils (Δ², ∂³) amplify grid-scale noise by ~1/dx^k → huge magnitude vs the rest
                # of the bank → training feedback diverged (NaN @ ~600 steps). RMS-NORMALIZE each per-channel
                # (fp32, spatial) so the conv head sees O(1) inputs (pattern preserved, magnitude bounded).
                def _rms(t):
                    t32 = tf.cast(t, tf.float32)
                    r = tf.sqrt(tf.reduce_mean(t32 * t32, axis=[1, 2], keepdims=True) + 1e-12)
                    return tf.cast(t32 / r, t.dtype)
                lap2 = laplacian(lap, dx, dy)                        # Δ² biharmonic (phase-field/KS)
                dxy = d_dx(gy, dx)                                   # ∂²/∂x∂y mixed (anisotropic)
                dxxx = d_dx(d_dx(gx, dx), dx)                        # ∂³/∂x³ (dispersive)
                dyyy = d_dy(d_dy(gy, dy), dy)                        # ∂³/∂y³
                gmag = tf.sqrt(gx * gx + gy * gy + 1e-12)            # |∇| (Hamilton-Jacobi/level-set)
                feats += [_rms(lap2), _rms(dxy), _rms(dxxx), _rms(dyyy), _rms(gmag)]
            if self.use_shock:                                       # conservation-form ∇·F̂ (shock-capturing)
                grad = tf.abs(gx) + tf.abs(gy)                       # shock indicator
                g = tf.sigmoid(self.s_gate(tf.concat([z, grad], axis=-1)))
                gxs, gys = g[..., 0:1], g[..., 1:2]
                xp = periodic_pad_2d(tf.concat([z, ctx["h_geom"]], axis=-1), 1)
                Fx = gxs * self.s_fxR(xp) + (1.0 - gxs) * self.s_fxL(xp)   # state-gated upwind flux
                Fy = gys * self.s_fyR(xp) + (1.0 - gys) * self.s_fyL(xp)
                feats.append(-(d_dx(Fx, dx) + d_dy(Fy, dy)))         # −∇·F̂ feature
            if self.xattn:
                # CROSS-ATTENTION combine: operators = tokens, transformer feature = query, per-pixel
                # softmax over the 7 operator tokens → attended physics (replaces the concat bank).
                optoks = [lap, gx, gy, adv, omega, div, z]           # 7 operator token sources
                toks = tf.stack([self.x_tok[i](optoks[i]) for i in range(self.n_optok)], axis=3)  # (B,H,W,7,d)
                q = self.x_q(ctx["h_geom"])                          # (B,H,W,d) ← transformer query
                scores = tf.einsum("bhwd,bhwnd->bhwn", q, toks) * (self.x_dim ** -0.5)  # (B,H,W,7)
                attn = tf.cast(tf.nn.softmax(tf.cast(scores, tf.float32), axis=-1), toks.dtype)
                bank = tf.einsum("bhwn,bhwnd->bhwd", attn, toks)     # (B,H,W,d) attended physics
            else:
                bank = tf.concat(feats, axis=-1)
        h = bank
        for conv, d in zip(self.convs, self.dilations):              # LOCAL branch: dilated conv stack
            h = conv(periodic_pad_2d(h, d))
        if self.fno:                                                 # NONLOCAL branch: FNO spectral
            h = h + self._fno(bank)
        elif self.multigrid:                                         # NONLOCAL branch: multigrid V-cycle
            h = h + self._vcycle(bank)                               # LOCAL + NONLOCAL preserved, summed
        else:
            gmean = tf.reduce_mean(h, axis=[1, 2], keepdims=True)
            h = h + self.gdense(gmean)                               # (legacy) weak global-mean scalar
        W = self.out(h)
        if self.reaction:                                            # +pointwise R(u)+f (zero-init → no-op at load)
            rx = tf.concat([z, ctx["h_geom"]], axis=-1)
            _hr = self.rx_out(self.rx_l2(self.rx_l1(rx)))
            W = W + (0.0 * _hr if "reaction" in self.heads_off else _hr)  # off: built but inert (positional order kept)
        if self.wave:                                                # +cross-channel ∇ coupling (zero-init → no-op)
            dx = ctx["metric"][..., 0:1]; dy = ctx["metric"][..., 1:2]
            wv = tf.concat([d_dx(z, dx), d_dy(z, dy), z, ctx["h_geom"]], axis=-1)
            _hw = self.wv_out(self.wv_mix(wv))
            W = W + (0.0 * _hw if "wave" in self.heads_off else _hw)     # off: built but inert
        if self.adapter:                                             # +generic fresh local operator (zero-init → no-op)
            a = tf.concat([z, ctx["h_geom"]], axis=-1)
            for conv, d in zip(self.ad_convs, self.ad_dils):
                a = conv(periodic_pad_2d(a, d))
            _ha = self.ad_out(a)
            W = W + (0.0 * _ha if "adapter" in self.heads_off else _ha)  # off: built but inert
        if self.buoyancy:                                            # +scalar→velocity source (Boussinesq, zero-init)
            dy = ctx["metric"][..., 1:2]
            by = tf.concat([z, d_dy(z, dy), ctx["h_geom"]], axis=-1)  # state + vertical gradient + geom
            _hb = self.by_out(self.by_l1(by))
            W = W + (0.0 * _hb if "buoyancy" in self.heads_off else _hb)  # off: built but inert
        if self.gradx:                                               # +horizontal gradient/shear head (∂_x, zero-init)
            dx = ctx["metric"][..., 0:1]
            gx = tf.concat([z, d_dx(z, dx), ctx["h_geom"]], axis=-1)  # state + horizontal gradient + geom
            _hg = self.gx_out(self.gx_l1(gx))
            W = W + (0.0 * _hg if "gradx" in self.heads_off else _hg)    # off: built but inert
        if self.advbias:                                             # +learned upwind advection-bias head (zero-init)
            dxa = ctx["metric"][..., 0:1]; dya = ctx["metric"][..., 1:2]
            sx, sy = self.vel_slots
            ux = z[..., sx:sx + 1]; uy = z[..., sy:sy + 1]
            adv_up = -(ux * upwind_dx(z, ux, dxa) + uy * upwind_dy(z, uy, dya))  # sign-aware upwind advection
            W = W + self.av_out(self.av_h(tf.concat([adv_up, z, ctx["h_geom"]], axis=-1)))
        if self.fixedops:                                            # +FIXED exact PDE operators, learned coeff (zero-init)
            dxf = ctx["metric"][..., 0:1]; dyf = ctx["metric"][..., 1:2]
            sxf, syf = self.vel_slots
            uxf = z[..., sxf:sxf + 1]; uyf = z[..., syf:syf + 1]
            lap = laplacian(z, dxf, dyf)                             # Δz  (diffusion / Laplacian)          [C]
            adv = -(uxf * minmod_dx(z, dxf) + uyf * minmod_dy(z, dyf))  # −(u·∇)z (advection, TVD)           [C]
            div = d_dx(uxf, dxf) + d_dy(uyf, dyf)                    # ∇·u (divergence / incompressibility) [1]
            react = z - z * z * z                                    # R(u)=u−u³ (Allen-Cahn reaction)      [C]
            cwave = z[..., 6:7]                                      # wave speed field (slot6); ~0 off-wave
            wave_op = (cwave * cwave) * lap                          # c²Δz (wave restoring)                [C]
            feats = tf.concat([lap, adv, react, wave_op, div], axis=-1)
            W = W + self.fx_out(feats)                               # zero-init mix → picks task's eq. terms
        return W

    def _fno(self, bank):
        """FNO SpectralConv2d: lift→2D rFFT→keep low modes (2 H-corners × low W)→learned complex per-mode
        channel mix→iFFT→proj (zero-init → no-op start). Global, non-local, parameter-efficient. FFT needs
        fp32/complex64 → cast around the bf16 compute path."""
        cd = tf.keras.mixed_precision.global_policy().compute_dtype
        x = tf.cast(self.fno_lift(bank), tf.float32)                 # (B,H,W,fw)
        Hh = int(x.shape[1]); Ww = int(x.shape[2]); m = self.fno_modes
        xt = tf.transpose(x, [0, 3, 1, 2])                           # (B,fw,H,W)
        ft = tf.signal.rfft2d(xt)                                    # (B,fw,H,W//2+1) complex64
        Wk = Ww // 2 + 1
        w = tf.complex(self.fno_wr, self.fno_wi)                     # (2,m,m,fw,fw) = (corner,x,y,in,out)
        c1 = tf.einsum("bixy,xyio->boxy", ft[:, :, :m, :m], w[0])    # low H modes
        c2 = tf.einsum("bixy,xyio->boxy", ft[:, :, Hh - m:, :m], w[1])  # high (negative) H modes
        c1p = tf.pad(c1, [[0, 0], [0, 0], [0, 0], [0, Wk - m]])      # pad W: m→Wk
        c2p = tf.pad(c2, [[0, 0], [0, 0], [0, 0], [0, Wk - m]])
        mid = tf.zeros([tf.shape(ft)[0], self.fno_w, Hh - 2 * m, Wk], tf.complex64)
        out_ft = tf.concat([c1p, mid, c2p], axis=2)                  # (B,fw,H,Wk)
        out = tf.signal.irfft2d(out_ft, fft_length=[Hh, Ww])         # (B,fw,H,W)
        out = tf.transpose(out, [0, 2, 3, 1])                        # (B,H,W,fw)
        return tf.cast(self.fno_proj(tf.cast(out, cd)), cd)

    def _vcycle(self, bank):
        """Non-local branch: multigrid V-cycle (the real elliptic/Poisson global coupling). Restrict
        avg-pool L→L/2…coarsest (near-global RF), prolong bilinear + fine-skip, periodic smoothers; zero-init
        proj so it starts as a no-op (hard-IC). Operates per-panel on the EVOLVING state's feature bank."""
        x = self.mg_pre(periodic_pad_2d(bank, 1))
        skips = []
        for l in range(self.mg_levels):
            skips.append(x)
            x = tf.nn.avg_pool2d(x, 2, 2, "VALID")                   # restrict
            x = self.mg_sd[l](periodic_pad_2d(x, 1))
        for l in reversed(range(self.mg_levels)):
            x = tf.cast(tf.image.resize(x, tf.shape(skips[l])[1:3], method="bilinear"), skips[l].dtype)
            x = x + skips[l]                                         # prolong + fine-level correction
            x = self.mg_su[l](periodic_pad_2d(x, 1))
        return self.mg_proj(x)                                       # → hidden ch (zero-init → no-op at start)
