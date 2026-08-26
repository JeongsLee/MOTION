"""Physics tendency banks (combo) + steady field head — the mechanism vocabulary.

Design (user directive, 2026-07-14):
  * combo, NOT adv-specialized: general zero-init gated heads that each supply one
    mechanism's contribution to the tendency W. Present for every problem; the
    per-family metadata prior + a learned gate turn irrelevant banks off (pruning).
  * steady vs unsteady bifurcation: steady problems solve BETTER without the time
    trajectory — they route through a direct elliptic field head (no time integration,
    no tendency banks), consistent with the old Poisson-Gauss result.
  * recursion (added separately) re-applies the unsteady banks on the evolved state.

Banks act on the latent box h (B, *dims, d). Each returns a tendency contribution
of width d_w*n_p (reshaped to (B,*dims,d_w,n_p) by the caller), zero-init so it is
inert at load and can be attached to a pretrained core without perturbation.
Derivatives use the axis-factorized 1D stencils (role-aware: gravity axis = buoyancy).
"""
from __future__ import annotations

import tensorflow as tf

from .axops import _to_axis_major

GRAVITY_ROLE = 2
# op-vocab indices (must match data.symbolic.OPERATOR_VOCAB) for the bank mechanisms
OP_INDEX = {"reaction": 3, "diffusion": 2, "buoyancy": 7, "shear": 8, "wave": 9,
            "advection": 1, "shock": 1, "elliptic": 11, "bernoulli": 6,  # bernoulli~pressure_projection
            "dilatation": 4, "reaction_nl": 3, "phase_interface": 13}  # dilatation~divergence; spectral/gradvec unmasked.
# 'geometry' is intentionally ABSENT -> never op-masked (curvature applies to any geometry, all families).
# STEADY problems solve a relaxation-to-equilibrium trajectory (fictional time), so the purely
# TRANSIENT/hyperbolic mechanisms are noise: wave (2nd time-deriv -> 0 at steady), shock (transient
# shock-capturing; it also rides advection's shared OP_INDEX bit so it leaks onto steady NS), buoyancy
# (drives a transient convective overturn). Dropped architecturally when steady=True; the steady
# balance (elliptic/diffusion/advection/bernoulli/BL/geometry/kinematic) is kept.
STEADY_DROP = ("wave", "shock", "buoyancy")


# RESOLUTION-INVARIANT STENCILS. Every operator below is a raw CELL difference — no 1/dx — so a
# tendency computed on a finer box comes out systematically smaller: a physical gradient scales
# like 1/n and a Laplacian like 1/n^2. The bank heads downstream absorbed whatever magnitude they
# were trained on, so simply raising the latent resolution silently rescales every mechanism and
# destroys a warm start (measured: shallow_water train loss 0.002 -> 0.063 going 64^2 -> 128^2).
# Differences are therefore expressed RELATIVE to the resolution the current weights were trained
# at: the factor is exactly 1.0 at REF_DIMS, so this is a no-op for existing checkpoints, and at
# 2x resolution it restores the magnitude the heads expect. Update REF_DIMS only together with a
# from-scratch run.
# MEASURED WRONG, kept as a record: rescaling by resolution made the warm start WORSE, not better
# (128^2 gate, cfdbench 0.51 -> 1.30, pdearena_ns 0.49 -> 1.74). The bank heads absorbed the CELL
# difference, and the latent's per-cell smoothness changes with resolution too, so multiplying by
# n/64 overshoots. REF_DIMS = dims turns _dscale into the identity, i.e. the original operators.
REF_DIMS = {}


def _dscale(x, axis):
    """Cell-difference -> physical-derivative factor along `axis`, relative to REF_DIMS."""
    K = len(x.shape) - 2
    n = x.shape[axis + 1]
    if n is None:
        return 1.0
    return float(n) / float(REF_DIMS.get(K, n))      # 1.0 unless REF_DIMS is populated


def _axis_grad(x, axis):
    """Central difference along spatial `axis` (periodic), rank-agnostic via axis-major fold."""
    flat, restore = _to_axis_major(x, axis)                    # (B*, n, d)
    up = tf.concat([flat[:, 1:], flat[:, :1]], 1)
    dn = tf.concat([flat[:, -1:], flat[:, :-1]], 1)
    return restore(0.5 * _dscale(x, axis) * (up - dn))


def _axis_diff_fb(x, axis):
    """Forward and backward first differences along `axis` (periodic) -> (fwd, bwd), each same
    shape as x. Used for sign(u)-biased UPWIND advection (shock-capturing)."""
    flat, restore = _to_axis_major(x, axis)                    # (B*, n, d)
    up = tf.concat([flat[:, 1:], flat[:, :1]], 1)
    dn = tf.concat([flat[:, -1:], flat[:, :-1]], 1)
    s = _dscale(x, axis)
    return restore(s * (up - flat)), restore(s * (flat - dn))  # fwd = h[i+1]-h[i], bwd = h[i]-h[i-1]


def _laplacian(x, K):
    acc = 0.0
    for a in range(K):
        flat, restore = _to_axis_major(x, a)
        up = tf.concat([flat[:, 1:], flat[:, :1]], 1)
        dn = tf.concat([flat[:, -1:], flat[:, :-1]], 1)
        acc = acc + restore(_dscale(x, a) ** 2 * (up - 2.0 * flat + dn))
    return acc


class TendencyBanks(tf.keras.layers.Layer):
    """W = base(h) + Σ_k σ_k ⊙ bank_k(h). base is small-init (the always-on head); each
    bank is a zero-init gated mechanism. Gates σ_k are per-channel learnable; a family
    metadata mask (which operators are present) multiplies them so absent mechanisms stay
    pruned. Output width = out (= d_w*n_p)."""

    def __init__(self, out, d, name="banks", active=None, dual=False,
                 gate_cond=0, gate_experts=0, **kw):
        super().__init__(name=name, **kw)
        # active: set of mechanism names to KEEP (others hard-off). None = all active. For steady
        # finetune, restrict to the family-relevant physics (e.g. {"diffusion","shear"}) so the
        # irrelevant mechanisms cannot inject noise / destabilise a specialist fit.
        self.active = set(active) if active else None
        self.base = tf.keras.layers.Dense(
            out, name=f"{name}_base",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3))
        # mechanism heads (combo): reaction, diffusion, buoyancy, shear, wave, advection.
        # heads are SMALL-random (not zeros): gate x head with BOTH zero-init is a gradient-dead
        # saddle (dL/dgate = head = 0, dL/dhead = gate = 0 forever). Zero gates alone keep the
        # output exactly unchanged at load while dL/dgate = head(x) != 0 lets gates open.
        # 'advection' is NONLINEAR self-advection -u.grad(h) with a LEARNED velocity u=vel(h)
        # (restores advb2's convective expert; the other banks are all LINEAR in h). NS/turbulent/
        # shock families are advection-dominated -> this is the missing nonlinear structure.
        # FULL physics set (advb2 parity): + 'shock' = sign(u)-biased UPWIND advection (CE/shock
        # capturing), + 'elliptic' = instantaneous global coupling (spatial-mean broadcast; the
        # long-range/pressure term local stencils miss).
        # + 'geometry' = mean curvature div(grad h / |grad h|) of the latent iso-surfaces (follows
        # small surface features; for geometry-dominated steady mesh like drivaernet). NEVER op-masked.
        # + 'bernoulli' = dynamic pressure -1/2 |u|^2 (u=vel(h)); surface pressure ~ Bernoulli.
        # + 'boundary_layer' = viscous near-wall physics: near-wall-weighted WALL-NORMAL shear &
        # diffusion (n=grad(SDF)); the pressure-LOSS / drag mechanism that Bernoulli/potential flow
        # miss (d'Alembert: inviscid recovers pressure -> zero drag). NEVER op-masked; self-gates via
        # the SDF (=0 where no wall). Needs geom_box (wall geometry) threaded from the encoder.
        # PDE-audit additions (2026-07-31): 'dilatation' = conservation-form flux divergence
        # [div(u), h*div(u)] — the other half of the product rule -div(h u) that 'advection'
        # (u.grad h) alone misses (SWE mass eq, compressible density/energy eqs); 'gradvec' =
        # the FULL per-axis first-derivative stack (the old feature set only had d/dx and
        # d/dgravity — K=3 lost d/dy entirely).
        self.names = ["reaction", "diffusion", "buoyancy", "shear", "wave",
                      "advection", "shock", "elliptic", "geometry", "boundary_layer",
                      "dilatation", "gradvec",
                      # Fig2c NEW-MECHANISM slot (PoolBoil arm B): interface-localized drive
                      # |grad h| * lap(h). Zero-init gate + op-mask ("phase_change") keep it an
                      # exact no-op for every existing family and at warm-start load.
                      "phase_interface"]
        self.heads = {m: tf.keras.layers.Dense(
            out, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3),
            name=f"{name}_{m}") for m in self.names}
        # NONLINEAR reaction: single-Dense heads are linear in their feature, so u-u^3 /
        # FitzHugh-Nagumo / EOS-type algebraic couplings were unrepresentable inside the bank.
        # A small gelu hidden layer restores them (advb2's reaction expert was an MLP).
        self.reaction_hidden = tf.keras.layers.Dense(
            2 * out if out < 128 else out, activation="gelu", name=f"{name}_reaction_h")
        # SPECTRAL / GREEN'S-FUNCTION bank: learnable radial transfer function g(|k|) applied
        # in Fourier space on the latent box = a learnable translation-invariant Green's
        # function (Delta^{-1}-class NONLOCAL operator). Serves the incompressible pressure
        # projection (grad Delta^{-1} div), the Poisson BVP, and potential-flow far fields —
        # the true elliptic coupling the global-mean 'elliptic' head cannot express.
        # K-agnostic weights: the transfer MLP maps scalar |k| -> per-feature multiplier.
        self.spec_kmlp = tf.keras.layers.Dense(16, activation="gelu", name=f"{name}_spec_k")
        self.spec_kout = tf.keras.layers.Dense(8, name=f"{name}_spec_ko")   # 8 spectral features
        self.spec_in = tf.keras.layers.Dense(8, name=f"{name}_spec_in")     # h -> 8 fields to filter
        # HEAD INIT (2026-08-01 gate probe): the 1e-3 head init that suits raw-derivative features
        # is too small when the feature path is ITSELF small/learned — dL/dgate = head(feat) becomes
        # ~1e-6 and the zero-init gate never opens (measured: spectral 1.7e-4, coefficient 3.6e-4,
        # reaction_nl 1.9e-4 after 22k steps, vs 0.06-0.29 for the raw-feature banks). These three
        # heads therefore use GLOROT init (same reasoning as `vel` below), keeping the zero gate as
        # the sole warm-safety device.
        self.spec_head = tf.keras.layers.Dense(out, name=f"{name}_spectral")
        self._spec = "spectral"
        # COEFFICIENT-FIELD bank: estimate a (quasi-static) spatially-varying material
        # coefficient field c(x) from the latent (the observed window's temporal signature
        # lives there) and combine it MULTIPLICATIVELY with the diffusion operator —
        # c(x)^2*lap(u)-type structure (Wave-Layer layered wave speed, spatially varying
        # viscosity) that purely additive heads cannot express. NOTE v1 re-estimates c per
        # AR step from the evolving latent (not frozen from the first window).
        self.coef_mlp = tf.keras.layers.Dense(8, activation="gelu", name=f"{name}_coef_c")
        self.coef_lproj = tf.keras.layers.Dense(8, name=f"{name}_coef_l")
        self.coef_head = tf.keras.layers.Dense(out, name=f"{name}_coefop")   # glorot: see spec_head
        self._coef = "coefficient"
        # COMPRESSIBLE bank: the momentum equation's dominant velocity source is the
        # RECIPROCAL-DENSITY pressure acceleration  -(1/rho) grad p  (+ (1/rho) div tau).
        # Every other head is a Dense ON a feature, i.e. LINEAR in it, so it can only produce
        # a*grad p — fine when rho is constant (incompressible) but structurally wrong when rho
        # varies by orders of magnitude (Mach ~1, shocks). Measured consequence: in the bylfa
        # convergence extension every family kept improving while com_ns sat at 1.56 for 32k
        # steps, and its velocity channel is the known laggard (66% vs PROSE-FD's 19%).
        # Here w = softplus(...) ~ 1/rho is inferred from the state and MULTIPLIES the gradient
        # stack, making the quotient representable. Also carries u*div(u), the compressible
        # self-transport term that vanishes when div u = 0.
        self.cmp_w = tf.keras.layers.Dense(1, name=f"{name}_cmp_w")          # ~ 1/rho (positive)
        self.cmp_head = tf.keras.layers.Dense(out, name=f"{name}_compress")  # glorot: see spec_head
        self._cmp = "compressible"
        # bernoulli has a DEDICATED head: input is the 3 squared velocity comps (dynamic pressure),
        # not the d-dim latent -> Dense(3->out). Its gate is added to self.gates below.
        self.bern_head = tf.keras.layers.Dense(
            out, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3),
            name=f"{name}_bernoulli")
        self._bern = "bernoulli"
        # KINEMATIC bank: velocity-gradient-tensor invariants [|S|^2, |Omega|^2, Q] from u=vel(h)
        # -> dedicated Dense(3->out). Vorticity/strain/Q ARE the turbulence & vortex descriptors and
        # set near-surface pressure kinematically (shared language for turbulence AND geometry).
        # Complements the 1st-order advection with 2nd-order gradient-tensor structure. Unmasked.
        self.kin_head = tf.keras.layers.Dense(
            out, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3),
            name=f"{name}_kinematic")
        self._kin = "kinematic"
        # velocity uses NORMAL (glorot) init, NOT tiny: u must be O(1) so the advection feat u.grad(h)
        # is O(grad h) and dL/dgate = head(adv) is healthy (tiny vel -> ~1e-9 gate grad, near-dead).
        # Warm-safety is unaffected (the zero gate, not vel, holds the branch off at load).
        self.vel = tf.keras.layers.Dense(3, name=f"{name}_vel")   # per-cell latent velocity (<=3 axes)
        # nonlinear reaction is a SEPARATE head (reaction_nl) so the r2-trained linear
        # 'reaction' head warm-loads untouched.
        self.reaction_nl_head = tf.keras.layers.Dense(out, name=f"{name}_reaction_nl")  # glorot
        self._rnl = "reaction_nl"
        # VORTEX STRETCHING bank (MOTION-NCS item A, 2026-08-06): (omega . grad) u from the
        # learned per-cell velocity — the 3D enstrophy-production mechanism the 2D-lineage
        # vocabulary above never needed. The STRUCTURE is prescribed (the exact tensor form of
        # omega.grad u, built from the same axis stencils as every other bank) and the CONTENT is
        # learned (u = vel(h) is a learned field; the head and gate are trained) — the paper-1
        # design rule. The feature is IDENTICALLY ZERO for K=2 by construction (see _vortex_feat):
        # no dimension flag, no op-mask needed — the architecture itself knows planar flow cannot
        # stretch vorticity, which is the "structural zero" row of the mechanism-ablation table.
        # GLOROT head (not 1e-3): the feature is a product of first derivatives of a learned field
        # AND non-zero in only 3 of 19 families, so the gate gradient dL/dgate = head(feat) is both
        # small and RARE — exactly the regime where the tiny-init heads measurably never opened
        # (08-01 gate probe; the boundary-layer-gate failure mode). Devices (i) gate_cond and
        # (ii) the support-normalized gate-gradient boost in v3/train.py complete item B.
        self.vs_head = tf.keras.layers.Dense(out, name=f"{name}_vortex")
        self._vs = "vortex"
        self._gnames = (self.names + [self._bern, self._kin, self._spec, self._rnl,
                                      self._coef, self._cmp, self._vs])
        self.gates = {m: self.add_weight(name=f"{name}_gate_{m}", shape=(out,),
                                         initializer="zeros", trainable=True)
                      for m in self._gnames}
        # WHERE THE SPECIALISATION BELONGS. The mechanism ANALYSIS above (gradients, laplacian,
        # curvature, wall-normal profile, spectral transfer) is the same physics for every family,
        # and the measurements agree: the core's 16 MoE experts sit 2.2% from their own mean, blend
        # at a top-1 share of 0.52, and using all 16 instead of 2 moves the class average by
        # 0.16pp — the core does not want to specialise. What differs between families is WHICH
        # mechanisms drive the next trajectory, and that lived in `gates[m]`: ONE (out,) vector
        # shared by all 26 families, modulated only by a BINARY op_multihot presence mask. So the
        # mixture was in the universal part and the part that should differentiate was a global
        # constant. Two modulations are added here, both zero-init so W is bit-identical at load:
        #   cond  — the symbolic equation embedding (which never reached the banks at all) scales
        #           each mechanism per EXAMPLE: continuous emphasis instead of a 0/1 mask.
        #   route — a per-CELL router mixes E gate profiles, so a cell near a wall can lean on the
        #           boundary-layer mechanism while a cell in the free stream leans on advection.
        self.gate_cond = int(gate_cond)
        self.gate_experts = int(gate_experts)
        if self.gate_cond:
            self.gcond = {m: tf.keras.layers.Dense(out, use_bias=False,
                                                   kernel_initializer="zeros",
                                                   name=f"{name}_gcond_{m}")
                          for m in self._gnames}
        if self.gate_experts > 1:
            self.grouter = tf.keras.layers.Dense(
                self.gate_experts, use_bias=False, name=f"{name}_grouter",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-2))
            self.gmix = {m: self.add_weight(name=f"{name}_gmix_{m}",
                                            shape=(self.gate_experts, out),
                                            initializer="zeros", trainable=True)
                         for m in self._gnames}
        # DECOMPOSED SECOND SOURCE (dual_core): a parallel hidden box h2 (from an orthogonal core)
        # feeds a SECOND set of mechanism heads reading the SAME operators computed on h2. These are
        # ZERO-init (contribution 0 at load -> warm-safe) but UNGATED, so dL/dW2 = upstream * feat2(h2)
        # != 0 (live gradient, no dead saddle). A decorrelation aux pushes h1 _|_ h2, so each source
        # spans a distinct feature subspace -> ~2x the banks' independent operators, no redundancy.
        self.dual = bool(dual)
        if self.dual:
            zero = tf.keras.initializers.Zeros()
            self.base2 = tf.keras.layers.Dense(out, kernel_initializer=zero, name=f"{name}_base2")
            self.heads2 = {m: tf.keras.layers.Dense(out, kernel_initializer=zero, name=f"{name}2_{m}")
                           for m in self.names}
            self.bern_head2 = tf.keras.layers.Dense(out, kernel_initializer=zero, name=f"{name}2_bernoulli")
            self.kin_head2 = tf.keras.layers.Dense(out, kernel_initializer=zero, name=f"{name}2_kinematic")
            self.vel2 = tf.keras.layers.Dense(3, name=f"{name}_vel2")   # h2's own per-cell velocity

    def _ops(self, h, K, u, geom_box):
        """Operator features for hidden h with per-cell velocity u: the mechanism vocabulary
        (reaction/diffusion/buoyancy/shear/wave/advection/shock/elliptic/geometry/boundary_layer)."""
        gx = _axis_grad(h, 0)
        gz = _axis_grad(h, min(GRAVITY_ROLE, K - 1)) if K >= 2 else gx
        lap = _laplacian(h, K)
        adv = 0.0                                                   # central: -u.grad(h)
        shock = 0.0                                                 # upwind: -u.grad_upwind(h)
        for a in range(K):
            ua = u[..., a:a + 1]
            adv = adv - ua * _axis_grad(h, a)
            fwd, bwd = _axis_diff_fb(h, a)
            shock = shock - ua * tf.where(ua > 0.0, bwd, fwd)       # sign(u)-biased upwind
        rank = h.shape.rank
        ell = tf.reduce_mean(h, axis=list(range(1, rank - 1)), keepdims=True) + 0.0 * h  # global mean
        # GEOMETRY: mean curvature of the latent iso-surfaces = div(grad h / |grad h|)
        gmag = tf.sqrt(sum(_axis_grad(h, a) ** 2 for a in range(K)) + 1e-6)
        kappa = 0.0
        for a in range(K):
            kappa = kappa + _axis_grad(_axis_grad(h, a) / gmag, a)
        # BOUNDARY LAYER: near-wall-weighted wall-normal shear + diffusion. n_hat = grad(SDF)/|.|,
        # w_wall = exp(-|SDF|/tau) localizes to the thin BL. Zero where no wall (SDF=0 -> n=0).
        if geom_box is not None:
            sdf = geom_box[..., 0:1]
            ng = [_axis_grad(sdf, a) for a in range(K)]
            nmag = tf.sqrt(sum(g * g for g in ng) + 1e-6)
            dn_h = sum((ng[a] / nmag) * _axis_grad(h, a) for a in range(K))        # (n_hat.grad) h
            dn2 = sum((ng[a] / nmag) * _axis_grad(dn_h, a) for a in range(K))      # wall-normal 2nd deriv
            bl = tf.exp(-tf.abs(sdf) / 0.15) * (dn_h + dn2)                        # near-wall weighted
        else:
            bl = tf.zeros_like(h)
        # DILATATION (conservation form): div(u) and h*div(u) — the -div(h u) product-rule
        # half that u.grad(h) misses (SWE mass eq, compressible density/energy equations).
        divu = 0.0
        for a in range(K):
            divu = divu + _axis_grad(u[..., a:a + 1], a)
        dil = tf.concat([divu, h * divu], -1)
        # GRADVEC: the FULL per-axis first-derivative stack in a FIXED role-indexed layout
        # (3 slots; absent axes zero) so one head weight serves K=2 and K=3. The old feature
        # set exposed only d/dx and d/dgravity — K=3 lost d/dy entirely.
        roles3 = [0, 1, 2]
        gv = [tf.zeros_like(h) for _ in range(3)]
        for a in range(K):
            gv[roles3[a] if a < 3 else 2] = _axis_grad(h, a)
        gradvec = tf.concat(gv, -1)
        gmag = tf.sqrt(gx * gx + gz * gz + 1e-12)                  # latent interface indicator
        return {"phase_interface": gmag * lap,
                "reaction": h, "diffusion": lap, "buoyancy": gz, "shear": gx, "wave": lap,
                "advection": adv, "shock": shock, "elliptic": ell, "geometry": kappa,
                "boundary_layer": bl, "dilatation": dil, "gradvec": gradvec}

    def _spectral_feat(self, h, K):
        """Learnable Green's-function bank: filter 8 projected fields by a learned radial
        transfer g(|k|) in Fourier space = translation-invariant NONLOCAL operator
        (Delta^{-1} class: incompressible pressure projection, Poisson BVP, potential flow).
        Weights are K-agnostic (transfer MLP reads scalar |k|); the FFT is per-K."""
        import numpy as np
        f = self.spec_in(h)                                     # (B,*dims,8)
        rank = h.shape.rank
        dims = [int(h.shape[i]) for i in range(1, rank - 1)]
        perm = [0, rank - 1] + list(range(1, rank - 1))         # (B,8,*dims)
        # tf.signal only implements FFTs for float32/64, and a spectral Green's function is
        # exactly where reduced mantissa hurts most, so this bank stays f32 under a mixed policy.
        cdt = f.dtype
        x = tf.cast(tf.transpose(f, perm), tf.float32)
        ax = [np.fft.fftfreq(n) * n for n in dims[:-1]] + [np.arange(dims[-1] // 2 + 1)]
        mesh = np.meshgrid(*ax, indexing="ij")
        kmag = np.sqrt(sum((m / max(n, 1)) ** 2 for m, n in zip(mesh, dims))).astype("float32")
        g = tf.cast(self.spec_kout(self.spec_kmlp(tf.constant(kmag[..., None]))), tf.float32)
        g = tf.complex(g, tf.zeros_like(g))                                # (*kdims,8)
        gt = tf.transpose(g, [len(dims)] + list(range(len(dims))))         # (8,*kdims)
        if K == 2:
            X = tf.signal.rfft2d(x)
            y = tf.signal.irfft2d(X * gt[None], fft_length=dims)
        else:
            # TF registers gradients for 1D/2D FFTs only (no IRFFT3D grad) -> compose:
            # rfft over the last spatial dim, then fft2d over (d0,d1) moved innermost.
            X = tf.signal.rfft(x)                               # (B,8,d0,d1,d2r) complex
            X = tf.transpose(X, [0, 1, 4, 2, 3])                # (B,8,d2r,d0,d1)
            X = tf.signal.fft2d(X)
            Gt = tf.transpose(gt, [0, 3, 1, 2])                 # (8,*kdims)->(8,d2r,d0,d1)
            X = X * Gt[None]
            X = tf.signal.ifft2d(X)
            X = tf.transpose(X, [0, 1, 3, 4, 2])                # back to (B,8,d0,d1,d2r)
            y = tf.signal.irfft(X, fft_length=[dims[-1]])
        inv = [0] + list(range(2, rank)) + [1]                  # back to (B,*dims,8)
        return tf.cast(tf.transpose(y, inv), cdt)

    def _gate_fn(self, h, e):
        """m -> the effective gate for mechanism m: the shared vector plus a per-example (cond)
        and a per-cell (router) modulation, both zero at load."""
        rank = h.shape.rank
        bshape = [-1] + [1] * (rank - 2)          # (B,1,..,1) -> broadcasts over the box only
        ce = None
        if self.gate_cond and e is not None:
            ce = tf.cast(e, h.dtype)
        w = None
        if self.gate_experts > 1:
            w = tf.nn.softmax(self.grouter(h), -1)                 # (B,*dims,E) per-cell mixture

        def gate(m):
            g = tf.cast(self.gates[m], h.dtype)                    # (out,)
            if ce is not None:
                g = g + tf.reshape(self.gcond[m](ce),
                                   bshape + [int(self.gates[m].shape[0])])   # per example
            if w is not None:
                g = g + tf.einsum("...e,eo->...o", w, tf.cast(self.gmix[m], h.dtype))
            return g

        return gate

    def _vortex_feat(self, u, K):
        """(omega . grad) u in the fixed 3-slot role layout, from the FIRST K components of the
        learned velocity only — vel(h) always emits 3 channels, and in K=2 the out-of-plane one
        is latent scratch, not physical velocity; including it would fabricate a 2D
        vortex-stretching signal. With i,j < K enforced, K=2 leaves omega = (0, 0, w_z) while
        the stretching sum only reaches d/dx and d/dy, so the feature is IDENTICALLY zero —
        the mathematical fact (planar flow cannot tilt or stretch vorticity), not a mask."""
        z = tf.zeros_like(u[..., :1])

        def g(i, j):                                            # du_i/dx_j, 0 off the K axes
            return _axis_grad(u[..., i:i + 1], j) if (i < K and j < K) else z

        om = [g(2, 1) - g(1, 2), g(0, 2) - g(2, 0), g(1, 0) - g(0, 1)]   # omega = curl(u)
        return tf.concat([sum(om[a] * g(i, a) for a in range(3))         # (omega . grad) u_i
                          for i in range(3)], -1)

    def _kin_feat(self, u, K):
        """Velocity-gradient-tensor invariants [|S|^2, |Omega|^2, Q] from per-cell velocity u."""
        G = [[_axis_grad(u[..., i:i + 1], j) for j in range(K)] for i in range(K)]  # du_i/dx_j
        S2 = sum((0.5 * (G[i][j] + G[j][i])) ** 2 for i in range(K) for j in range(K))  # |strain|^2
        Om2 = sum((0.5 * (G[i][j] - G[j][i])) ** 2 for i in range(K) for j in range(K))  # |rotation|^2
        return tf.concat([S2, Om2, 0.5 * (Om2 - S2)], -1)                # [|S|^2, |Om|^2, Q]

    def call(self, h, K=2, roles=None, op_multihot=None, geom_box=None, h2=None, steady=False,
             e=None):
        """h (B,*dims,d) -> W_flat (B,*dims,out). op_multihot (B,|vocab|): per-example metadata
        prior that PRUNES mechanisms absent from a family's governing equation (gate*mask).
        geom_box (B,*dims,>=1): wall geometry (ch0=SDF) for the boundary_layer bank; None -> BL off.
        h2 (B,*dims,d): decomposed second source (dual_core); None -> single-source (unchanged).
        steady: drop the transient/hyperbolic mechanisms (STEADY_DROP) — they are noise on the
        relaxation-to-equilibrium trajectory (no physical time)."""
        roles = roles or list(range(K))
        drop = set(STEADY_DROP) if steady else set()
        # Normalise the auxiliary inputs to h's dtype AT THE BOUNDARY rather than at each of the
        # ~15 mechanisms below. Under a mixed policy Keras 2 and Keras 3 disagree about which
        # kwargs get auto-cast, so relying on the caller is how a version-specific dtype error
        # ends up only appearing on the cluster.
        if geom_box is not None:
            geom_box = tf.cast(geom_box, h.dtype)
        if h2 is not None:
            h2 = tf.cast(h2, h.dtype)
        if op_multihot is not None:
            op_multihot = tf.cast(op_multihot, h.dtype)
        u = self.vel(h)                                            # (B,*dims,3) learned latent velocity
        feats = self._ops(h, K, u, geom_box)
        gate = self._gate_fn(h, e)                                 # m -> gate, broadcastable to W
        # EVAL-ONLY PROBE (MOTION-NCS figures): when `self.probe` is a dict (eager calls only),
        # each mechanism's gated contribution is recorded — mean |contrib| under key m, and the
        # full spatial field under key m+"_field" when self.probe_field == m. No effect when None.
        probe = getattr(self, "probe", None)

        def _rec(m, contrib):
            if probe is None:
                return
            probe[m] = float(tf.reduce_mean(tf.abs(contrib)))
            if getattr(self, "probe_field", None) == m:
                probe[m + "_field"] = contrib.numpy()
        bshape = [-1] + [1] * (h.shape.rank - 1)             # (B,1,..,1) broadcast over box + out
        W = self.base(h)
        for m in self.names:
            if self.active is not None and m not in self.active:
                continue                                    # mechanism hard-off (steady finetune)
            if m in drop:
                continue                                    # transient mechanism off for steady
            contrib = gate(m) * self.heads[m](feats[m])                  # (B,*dims,out)
            if op_multihot is not None and m in OP_INDEX:               # 'geometry' is never masked
                contrib = contrib * tf.reshape(op_multihot[:, OP_INDEX[m]], bshape)
            _rec(m, contrib)
            W = W + contrib
        # BERNOULLI: dynamic pressure -1/2 |u|^2 (per-axis squared velocity -> dedicated head)
        if self.active is None or self._bern in self.active:
            bern = gate(self._bern) * self.bern_head(-0.5 * u * u)     # u*u = (B,*dims,3)
            if op_multihot is not None:
                bern = bern * tf.reshape(op_multihot[:, OP_INDEX[self._bern]], bshape)
            _rec(self._bern, bern)
            W = W + bern
        # KINEMATIC: velocity-gradient-tensor invariants from u=vel(h). UNMASKED.
        if self.active is None or self._kin in self.active:
            kin = gate(self._kin) * self.kin_head(self._kin_feat(u, K))
            _rec(self._kin, kin)
            W = W + kin
        # VORTEX STRETCHING: (omega.grad)u from u=vel(h). UNMASKED — the feature itself is
        # identically zero for K=2 (see _vortex_feat), so the architecture, not metadata,
        # switches it off on planar families.
        if self.active is None or self._vs in self.active:
            vs = gate(self._vs) * self.vs_head(self._vortex_feat(u, K))
            _rec(self._vs, vs)
            W = W + vs
        # NONLINEAR REACTION: gelu-hidden pointwise map (u-u^3 / FitzHugh-Nagumo / EOS-type
        # algebraic couplings are unrepresentable by the linear single-Dense heads).
        if self.active is None or self._rnl in self.active:
            rnl = gate(self._rnl) * self.reaction_nl_head(self.reaction_hidden(h))
            if op_multihot is not None:
                rnl = rnl * tf.reshape(op_multihot[:, OP_INDEX[self._rnl]], bshape)
            _rec(self._rnl, rnl)
            W = W + rnl
        # SPECTRAL Green's-function bank (nonlocal elliptic coupling). UNMASKED: pressure
        # projection / Poisson / potential-flow far fields span many families; gate learns.
        if self.active is None or self._spec in self.active:
            sp = gate(self._spec) * self.spec_head(self._spectral_feat(h, K))
            _rec(self._spec, sp)
            W = W + sp
        # COEFFICIENT-FIELD bank: c(x) (x) diffusion — multiplicative material structure.
        # UNMASKED (layered media, variable viscosity, per-sample nu all qualify).
        if self.active is None or self._coef in self.active:
            cf = self.coef_mlp(h)                                       # (B,*dims,8) ~ c(x)
            if probe is not None and getattr(self, "probe_field", None) == "coefficient":
                probe["coefficient_field"] = cf.numpy()
            feat = tf.concat([cf * self.coef_lproj(feats["diffusion"]), cf], -1)
            co = gate(self._coef) * self.coef_head(feat)
            _rec(self._coef, co)
            W = W + co
        # COMPRESSIBLE bank: (1/rho)-weighted pressure/stress acceleration + u*div(u).
        # UNMASKED — the gate learns where compressibility matters (it is ~0 for div-free flow).
        if self.active is None or self._cmp in self.active:
            w = tf.nn.softplus(self.cmp_w(h))                           # (B,*dims,1) ~ 1/rho > 0
            divu = 0.0
            for a in range(K):
                divu = divu + _axis_grad(u[..., a:a + 1], a)
            cfeat = tf.concat([w * feats["gradvec"],                    # (1/rho) grad(state)
                               w * feats["diffusion"],                  # (1/rho) div(tau)
                               u * divu, w], -1)                        # compressible self-transport
            cp = gate(self._cmp) * self.cmp_head(cfeat)
            _rec(self._cmp, cp)
            W = W + cp
        # DECOMPOSED h2 SOURCE: same operator vocabulary on the orthogonal box h2, through UNGATED
        # zero-init heads (warm-safe: 0 at load; live gradient). Same op-mask so absent mechanisms
        # stay pruned. Its own velocity (self.vel2) so h2's advection/bernoulli/kinematic are h2-native.
        if self.dual and h2 is not None:
            u2 = self.vel2(h2)
            f2 = self._ops(h2, K, u2, geom_box)
            W = W + self.base2(h2)
            for m in self.names:
                if self.active is not None and m not in self.active:
                    continue
                c2 = self.heads2[m](f2[m])
                if op_multihot is not None and m in OP_INDEX:
                    c2 = c2 * tf.reshape(op_multihot[:, OP_INDEX[m]], bshape)
                W = W + c2
            if self.active is None or self._bern in self.active:
                b2 = self.bern_head2(-0.5 * u2 * u2)
                if op_multihot is not None:
                    b2 = b2 * tf.reshape(op_multihot[:, OP_INDEX[self._bern]], bshape)
                W = W + b2
            if self.active is None or self._kin in self.active:
                W = W + self.kin_head2(self._kin_feat(u2, K))
        return W


class SteadyField(tf.keras.layers.Layer):
    """Direct elliptic field head for STEADY problems: box features -> field coefficients,
    NO time integration, NO tendency banks. u(query) = spectral synthesis of this field.
    Solves airfrans/elasticity/Poisson which the trajectory path leaves stuck at ~1.0."""

    def __init__(self, c_out, d, hidden=None, name="steady", **kw):
        super().__init__(name=name, **kw)
        h = hidden or 2 * d
        self.h = tf.keras.layers.Dense(h, activation="gelu", name=f"{name}_h")
        self.o = tf.keras.layers.Dense(c_out, name=f"{name}_o")   # normal init: field, not a residual

    def call(self, hbox):
        """hbox (B,*dims,d) -> field (B,*dims,c_out) on the latent box."""
        return self.o(self.h(hbox))
