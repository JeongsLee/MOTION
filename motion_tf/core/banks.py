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

import os

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

# REVISION-R1 ablation V1a ("typed heads -> generic learned heads", R1 #1). MOTION_BANK_GENERIC=1
# replaces every PRESCRIBED operator feature with a learned one of the same spatial support and
# nonlinearity class; head count, Dense heads, gates, metadata masks and the summed structure are
# unchanged. Heads without a prescribed operator (reaction, reaction_nl, elliptic mean, base H0)
# are untouched. Random init (no operator values leak in).
BANK_GENERIC = os.environ.get("MOTION_BANK_GENERIC", "0") == "1"
# V2 ("all gates open, no metadata gating", R1 #1/#2): no op_multihot mask, no steady drop, and the
# per-example gate modulation reads the DATA (mean-pooled latent) instead of the equation embedding.
BANK_NOMETA = os.environ.get("MOTION_BANK_NOMETA", "0") == "1"
# V5 (R3 #1 "generic head"): no always-on state head H0; W is the sum of the gated heads only.
BANK_NOH0 = os.environ.get("MOTION_BANK_NOH0", "0") == "1"
# HYBRID (revision R1, R2 head-count / vocabulary-gap): MOTION_BANK_NGENERIC=G adds G GENERIC heads
# next to the typed vocabulary. Each reads [learned axial stencil (.) learned low-rank map of h,
# anisotropic learned spectral transfer g(k) of 8 projected fields] -> Dense(out). No operator is
# prescribed, no metadata mask, never steady-dropped; zero-init gate (output unchanged at init).
# With MOTION_BANK_NOMETA=1 the typed AND generic gates are data-driven (consistent hybrid); without
# it typed heads keep the metadata mask while generic heads are always open (mixed hybrid).
BANK_NGENERIC = int(os.environ.get("MOTION_BANK_NGENERIC", "0"))
# MERGED HYBRID (revision R1, head-count control): MOTION_BANK_GENERIC_MERGE=1 keeps the G generic BRANCHES
# (stencil (.) low-rank, spectral) but concatenates them into ONE head: one Dense((G*(d+8)) -> out) projection
# (= the G per-head projections stacked, same parameter count) behind ONE gate. Only the per-head gates
# (G-1 gate vectors + their cond modulators) are lost; the generic capacity is otherwise unchanged.
BANK_GENERIC_MERGE = os.environ.get("MOTION_BANK_GENERIC_MERGE", "0") == "1"
# PHYSICAL GATING of two always-on heads (revision R1, MOTION-NS-S com_ns, 10-19): every velocity operator reads one
# latent velocity u = vel(h); the always-open compressible head (u div u, (1/rho) grad p) lets the div-free families
# (pressure_projection token) drive div u -> 0, while com_ns, which needs div u != 0, leans on the always-open
# spectral (projection-type, non-local) head. MOTION_BANK_PHYSGATE=1 opens 'compressible' only with the divergence
# token and 'spectral' only with the pressure_projection token (same op-mask product as the typed heads). Default
# off = unchanged. Ignored without metadata (nometa: every head open).
BANK_PHYSGATE = os.environ.get("MOTION_BANK_PHYSGATE", "0") == "1"
PHYSGATE_TOK = {"compressible": 4, "spectral": 6}   # data/symbolic.py OPERATOR_VOCAB: divergence, pressure_projection
# SINGLE-HEAD BANK (revision R1/R2, "is head separation needed?"): MOTION_BANK_SINGLE=lin|mlp.
# The typed heads' per-head Dense maps A_k are kept as the blocks of ONE linear map over the
# concatenated (metadata-masked) operator features, L = sum_k m_k A_k F_k, behind ONE shared gate
# ("_bank": vector + equation-cond modulation) instead of one gate per mechanism.
#   lin: W = H0 + sigma (.) L           -- same function class as the per-head bank (Eq. 1 unchanged)
#   mlp: W = H0 + sigma (.) (L + MLP(L)) -- adds pointwise nonlinear mechanism coupling; MLP out is
#        zero-init, so it starts as `lin`. MOTION_BANK_SINGLE_HIDDEN sets the MLP width (default 512).
BANK_SINGLE = os.environ.get("MOTION_BANK_SINGLE", "")
BANK_SINGLE_HIDDEN = int(os.environ.get("MOTION_BANK_SINGLE_HIDDEN", "512"))
assert BANK_SINGLE in ("", "lin", "mlp"), BANK_SINGLE
_MAX_ROLES = 4


class _AxStencil(tf.keras.layers.Layer):
    """Learned per-channel axial stencil of width w: one kernel (w, d) per axis ROLE, applied along
    every spatial axis with periodic shifts (the typed stencils are periodic cell differences).
    K-agnostic like the typed bank. Returns the per-axis list or its sum."""

    def __init__(self, d, w=3, name="axs", **kw):
        super().__init__(name=name, **kw)
        self.w = int(w)
        self.k = self.add_weight(name=f"{name}_k", shape=(_MAX_ROLES, self.w, d),
                                 initializer=tf.keras.initializers.RandomNormal(stddev=self.w ** -0.5),
                                 trainable=True)

    def call(self, h, K, roles, per_axis=False):
        c = self.w // 2
        outs = []
        for a in range(K):
            ker = tf.cast(self.k[roles[a] % _MAX_ROLES], h.dtype)          # (w, d)
            outs.append(tf.add_n([ker[j] * tf.roll(h, shift=c - j, axis=1 + a)
                                  for j in range(self.w)]))
        return outs if per_axis else tf.add_n(outs)


class _LowRank(tf.keras.layers.Layer):
    """Learned low-rank linear map h -> d (rank r): the generic partner of a bilinear product."""

    def __init__(self, d, r=8, name="lr", **kw):
        super().__init__(name=name, **kw)
        self.a = tf.keras.layers.Dense(r, use_bias=False, name=f"{name}_a")
        self.b = tf.keras.layers.Dense(d, name=f"{name}_b")

    def call(self, h):
        return self.b(self.a(h))


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
        # VOCABULARY-SUBSET ABLATION (revision R2): MOTION_BANK_ACTIVE="a,b,c" keeps only these gated heads
        # (H0 always on). The other heads are skipped in the forward (their variables exist but stay unused).
        if os.environ.get("MOTION_BANK_ACTIVE"):
            self.active = {m.strip() for m in os.environ["MOTION_BANK_ACTIVE"].split(",") if m.strip()}
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
                      # interface-sensitive head kept from development, not part of the vocabulary
                      # analysed in the paper: |grad h| * lap(h). Zero-init gate + op-mask
                      # ("phase_change", a token that no family of the paper carries) keep it an
                      # exact no-op with metadata and at warm-start load; models trained without
                      # metadata apply no op-mask, so every head including this one is open.
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
        self.generic = BANK_GENERIC
        if self.generic:
            # same support as the typed operator: 3-point stencils, 5 for the nested-gradient
            # curvature; bilinear partners for every head whose typed feature is a product
            self.gS = {m: _AxStencil(d, 5 if m == "geometry" else 3, name=f"{name}_gS_{m}")
                       for m in ("diffusion", "wave", "shear", "buoyancy", "gradvec", "geometry",
                                 "phase_interface", "advection", "shock", "boundary_layer",
                                 "dilatation", "kinematic", "vortex", "coefficient",
                                 "cmp_gv", "cmp_lap", "cmp_uu")}
            self.gL = {m: _LowRank(d, 8, name=f"{name}_gL_{m}")
                       for m in ("geometry", "phase_interface", "advection", "shock",
                                 "boundary_layer", "dilatation", "kinematic", "vortex", "cmp_uu")}
            self.g_dil1 = tf.keras.layers.Dense(1, name=f"{name}_g_dil1")      # -> the scalar div slot
            self.g_kin3 = tf.keras.layers.Dense(3, name=f"{name}_g_kin3")      # -> 3 invariant slots
            self.g_vs3 = tf.keras.layers.Dense(3, name=f"{name}_g_vs3")        # -> 3 stretching slots
            self.g_uu3 = tf.keras.layers.Dense(3, name=f"{name}_g_uu3")        # -> u*div(u) slots
            self.g_bA = tf.keras.layers.Dense(3, name=f"{name}_g_bernA")       # bernoulli: quadratic
            self.g_bB = tf.keras.layers.Dense(3, name=f"{name}_g_bernB")       #   form (A h)(B h)
            # spectral: ANISOTROPIC learned transfer g(k) from the role-indexed k-vector (3 slots)
            self.g_spec_k = tf.keras.layers.Dense(16, activation="gelu", name=f"{name}_g_spec_k")
        self.n_generic = BANK_NGENERIC
        self.gen_merge = BANK_GENERIC_MERGE and self.n_generic > 0
        self._gen = (["generic"] if self.gen_merge else [f"generic{i}" for i in range(self.n_generic)])
        if self.n_generic:
            self.gen_S = [_AxStencil(d, 3, name=f"{name}_genS{i}") for i in range(self.n_generic)]
            self.gen_L = [_LowRank(d, 8, name=f"{name}_genL{i}") for i in range(self.n_generic)]
            self.gen_si = [tf.keras.layers.Dense(8, name=f"{name}_genSI{i}") for i in range(self.n_generic)]
            self.gen_sk = [tf.keras.layers.Dense(16, activation="gelu", name=f"{name}_genSK{i}")
                           for i in range(self.n_generic)]
            self.gen_so = [tf.keras.layers.Dense(8, name=f"{name}_genSO{i}") for i in range(self.n_generic)]
            self.gen_head = [tf.keras.layers.Dense(
                out, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3),
                name=f"{name}_generic{i}") for i in range(1 if self.gen_merge else self.n_generic)]
        self._gnames = (self.names + [self._bern, self._kin, self._spec, self._rnl,
                                      self._coef, self._cmp, self._vs] + self._gen)
        self.single = BANK_SINGLE
        self._gatenames = (["_bank"] + self._gen) if self.single else self._gnames
        if self.single == "mlp":
            self.single_h = tf.keras.layers.Dense(BANK_SINGLE_HIDDEN, activation="gelu", name=f"{name}_single_h")
            self.single_o = tf.keras.layers.Dense(out, kernel_initializer="zeros", name=f"{name}_single_o")
        self.gates = {m: self.add_weight(name=f"{name}_gate_{m}", shape=(out,),
                                         initializer="zeros", trainable=True)
                      for m in self._gatenames}
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
        self.nometa = BANK_NOMETA
        self.noh0 = BANK_NOH0
        if self.gate_cond and self.nometa:      # data-driven per-example gate input (d -> d_cond=128)
            self.g_data = tf.keras.layers.Dense(128, activation="gelu", name=f"{name}_g_data")
        if self.gate_cond:
            self.gcond = {m: tf.keras.layers.Dense(out, use_bias=False,
                                                   kernel_initializer="zeros",
                                                   name=f"{name}_gcond_{m}")
                          for m in self._gatenames}
        if self.gate_experts > 1:
            self.grouter = tf.keras.layers.Dense(
                self.gate_experts, use_bias=False, name=f"{name}_grouter",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-2))
            self.gmix = {m: self.add_weight(name=f"{name}_gmix_{m}",
                                            shape=(self.gate_experts, out),
                                            initializer="zeros", trainable=True)
                         for m in self._gatenames}
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

    def _ops_generic(self, h, K, roles, geom_box):
        """V1a: learned features with the typed features' shapes (so every head keeps its size)."""
        S, L = self.gS, self.gL
        lin = lambda m: S[m](h, K, roles)
        bil = lambda m: S[m](h, K, roles) * L[m](h)
        rank = h.shape.rank
        ell = tf.reduce_mean(h, axis=list(range(1, rank - 1)), keepdims=True) + 0.0 * h
        gv = [tf.zeros_like(h) for _ in range(3)]
        for a, y in enumerate(S["gradvec"](h, K, roles, per_axis=True)):
            gv[a if a < 3 else 2] = y
        if geom_box is not None:                     # wall localisation is DATA, kept as in typed
            sdf = tf.cast(geom_box[..., 0:1], h.dtype)
            n2 = sum(_axis_grad(sdf, a) ** 2 for a in range(K))
            wall = tf.exp(-tf.abs(sdf) / 0.15) * tf.cast(n2 > 1e-8, h.dtype)
            bl = bil("boundary_layer") * wall
        else:
            bl = tf.zeros_like(h)
        dl = bil("dilatation")
        return {"phase_interface": bil("phase_interface"),
                "reaction": h, "diffusion": lin("diffusion"), "buoyancy": lin("buoyancy"),
                "shear": lin("shear"), "wave": lin("wave"),
                "advection": bil("advection"), "shock": bil("shock"), "elliptic": ell,
                "geometry": bil("geometry"), "boundary_layer": bl,
                "dilatation": tf.concat([self.g_dil1(lin("dilatation")), dl], -1),
                "gradvec": tf.concat(gv, -1)}

    def _spectral_feat(self, h, K, kvec=False):
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
        if kvec:                                     # V1a: anisotropic g(k), role-indexed 3 slots
            comps = [(m / max(n, 1)).astype("float32") for m, n in zip(mesh, dims)]
            comps = comps + [np.zeros_like(kmag)] * (3 - len(comps))
            g = tf.cast(self.spec_kout(self.g_spec_k(tf.constant(np.stack(comps, -1)))), tf.float32)
        else:
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

    def _generic_spec(self, h, K, i):
        """Anisotropic learned spectral transfer g_i(k) on 8 projected fields (generic head i)."""
        import numpy as np
        f = self.gen_si[i](h)
        rank = h.shape.rank
        dims = [int(h.shape[j]) for j in range(1, rank - 1)]
        perm = [0, rank - 1] + list(range(1, rank - 1))
        cdt = f.dtype
        x = tf.cast(tf.transpose(f, perm), tf.float32)
        ax = [np.fft.fftfreq(n) * n for n in dims[:-1]] + [np.arange(dims[-1] // 2 + 1)]
        mesh = np.meshgrid(*ax, indexing="ij")
        comps = [(m / max(n, 1)).astype("float32") for m, n in zip(mesh, dims)]
        comps = comps + [np.zeros_like(comps[0])] * (3 - len(comps))
        g = tf.cast(self.gen_so[i](self.gen_sk[i](tf.constant(np.stack(comps, -1)))), tf.float32)
        gt = tf.transpose(tf.complex(g, tf.zeros_like(g)), [len(dims)] + list(range(len(dims))))
        if K == 2:
            y = tf.signal.irfft2d(tf.signal.rfft2d(x) * gt[None], fft_length=dims)
        else:
            X = tf.transpose(tf.signal.rfft(x), [0, 1, 4, 2, 3])
            X = tf.signal.fft2d(X) * tf.transpose(gt, [0, 3, 1, 2])[None]
            X = tf.transpose(tf.signal.ifft2d(X), [0, 1, 3, 4, 2])
            y = tf.signal.irfft(X, fft_length=[dims[-1]])
        return tf.cast(tf.transpose(y, [0] + list(range(2, rank)) + [1]), cdt)

    def _gate_fn(self, h, e):
        """m -> the effective gate for mechanism m: the shared vector plus a per-example (cond)
        and a per-cell (router) modulation, both zero at load."""
        rank = h.shape.rank
        bshape = [-1] + [1] * (rank - 2)          # (B,1,..,1) -> broadcasts over the box only
        ce = None
        if self.gate_cond and self.nometa:      # V2: gate modulation from the data, not metadata
            ce = self.g_data(tf.reduce_mean(h, axis=list(range(1, rank - 1))))
        elif self.gate_cond and e is not None:
            ce = tf.cast(e, h.dtype)
        w = None
        if self.gate_experts > 1:
            w = tf.nn.softmax(self.grouter(h), -1)                 # (B,*dims,E) per-cell mixture

        def gate(m):
            if self.single and m not in self.gates:                # single-head bank: typed heads
                return 1.0                                         # are blocks of one map (shared gate)
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
        drop = set(STEADY_DROP) if (steady and not self.nometa) else set()
        if self.nometa:
            op_multihot = None                  # V2: every head open for every family
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
        feats = (self._ops_generic(h, K, roles, geom_box) if self.generic
                 else self._ops(h, K, u, geom_box))
        gate = self._gate_fn(h, e)                                 # m -> gate, broadcastable to W
        # EVAL-ONLY PRESCRIBED DIFFUSION COMMAND (REVISION-R1 10-20): when `self.lap_steer` = c is set, the exact
        # latent Laplacian c*Lap(h) is added to the Laplacian feature of the heads in `self.lap_steer_heads`
        # (default: diffusion only) -- the operator component alone, without the head's bias (cf. the advection
        # command, which adds du to u(h) inside -u.grad(h)). No effect when None.
        _ls = getattr(self, "lap_steer", None)
        _lsh = getattr(self, "lap_steer_heads", ("diffusion",)) if _ls is not None else ()
        feats_s = feats
        if _ls is not None:
            feats_s = dict(feats)
            feats_s["diffusion"] = feats["diffusion"] + tf.cast(_ls, h.dtype) * _laplacian(h, K)
        # EVAL-ONLY PROBE (MOTION-NCS figures): when `self.probe` is a dict (eager calls only),
        # each mechanism's gated contribution is recorded — mean |contrib| under key m, and the
        # full spatial field under key m+"_field" when self.probe_field == m. No effect when None.
        probe = getattr(self, "probe", None)

        # EVAL-ONLY RUNTIME KNOCKOUT (revision R1): when `self.ko_mask` (a non-trainable tf.Variable,
        # one entry per head, order `self.ko_index`) is set, each head's contribution is multiplied
        # by its entry. Assigning the variable switches knockouts WITHOUT retracing / recompiling
        # (the python-level `active` set needs a fresh trace per configuration). x*0 == 0 exactly,
        # so a 0 entry reproduces `active`-based removal. No effect when ko_mask is None.
        ko_mask = getattr(self, "ko_mask", None)
        ko_index = getattr(self, "ko_index", None)

        def _rec(m, contrib):
            if ko_mask is not None and m in ko_index:
                contrib = contrib * tf.cast(ko_mask[ko_index[m]], contrib.dtype)
            if probe is None:
                return contrib
            if tf.executing_eagerly():
                probe[m] = float(tf.reduce_mean(tf.abs(contrib)))
                if getattr(self, "probe_field", None) == m:
                    probe[m + "_field"] = contrib.numpy()
            else:                                           # graph mode: return tensors via the dict
                probe[m] = tf.reduce_mean(tf.abs(tf.cast(contrib, tf.float32)))
                if getattr(self, "probe_all_fields", False):    # redundancy analysis: full fields
                    probe[m + "_field"] = tf.cast(contrib, tf.float32)
            return contrib
        bshape = [-1] + [1] * (h.shape.rank - 1)             # (B,1,..,1) broadcast over box + out
        W = tf.zeros_like(self.base(h)) if self.noh0 else self.base(h)   # V5: no H0
        if not self.noh0:
            W = _rec("H0", W)                               # eval probe; runtime knob/KO when "H0" is in ko_index
                                                            # (REVISION-R1 10-20; identity otherwise)
        W0 = W
        for m in self.names:
            if self.active is not None and m not in self.active:
                continue                                    # mechanism hard-off (steady finetune)
            if m in drop:
                continue                                    # transient mechanism off for steady
            if (op_multihot is not None and m in OP_INDEX
                    and OP_INDEX[m] >= int(op_multihot.shape[-1])):
                continue                                    # token absent from this vocabulary
                #   (MOTION_LEGACY_VOCAB: the 13-token checkpoints have no 'phase_change') -> closed
            contrib = gate(m) * self.heads[m]((feats_s if m in _lsh else feats)[m])   # (B,*dims,out)
            if op_multihot is not None and m in OP_INDEX:               # 'geometry' is never masked
                contrib = contrib * tf.reshape(op_multihot[:, OP_INDEX[m]], bshape)
            contrib = _rec(m, contrib)
            W = W + contrib
        # BERNOULLI: dynamic pressure -1/2 |u|^2 (per-axis squared velocity -> dedicated head)
        if self.active is None or self._bern in self.active:
            bfeat = (self.g_bA(h) * self.g_bB(h) if self.generic else -0.5 * u * u)
            bern = gate(self._bern) * self.bern_head(bfeat)            # u*u = (B,*dims,3)
            if op_multihot is not None:
                bern = bern * tf.reshape(op_multihot[:, OP_INDEX[self._bern]], bshape)
            bern = _rec(self._bern, bern)
            W = W + bern
        # KINEMATIC: velocity-gradient-tensor invariants from u=vel(h). UNMASKED.
        if self.active is None or self._kin in self.active:
            kfeat = (self.g_kin3(self.gS["kinematic"](h, K, roles) * self.gL["kinematic"](h))
                     if self.generic else self._kin_feat(u, K))
            kin = gate(self._kin) * self.kin_head(kfeat)
            kin = _rec(self._kin, kin)
            W = W + kin
        # VORTEX STRETCHING: (omega.grad)u from u=vel(h). UNMASKED — the feature itself is
        # identically zero for K=2 (see _vortex_feat), so the architecture, not metadata,
        # switches it off on planar families.
        # REVISION-R1 fix (matches the paper: 'identically zero in 2D'): the feature is zero for
        # K=2 but vs_head carries a bias, so the legacy path added gate*bias on planar families.
        # The head is now evaluated only for K=3 (static). MOTION_VORTEX2D_LEGACY=1 reproduces
        # checkpoints trained with the legacy path.
        _vs_on = K == 3 or os.environ.get("MOTION_VORTEX2D_LEGACY", "0") == "1"
        if (self.active is None or self._vs in self.active) and _vs_on:
            vfeat = (self.g_vs3(self.gS["vortex"](h, K, roles) * self.gL["vortex"](h))
                     if self.generic else self._vortex_feat(u, K))
            vs = gate(self._vs) * self.vs_head(vfeat)
            vs = _rec(self._vs, vs)
            W = W + vs
        # NONLINEAR REACTION: gelu-hidden pointwise map (u-u^3 / FitzHugh-Nagumo / EOS-type
        # algebraic couplings are unrepresentable by the linear single-Dense heads).
        if self.active is None or self._rnl in self.active:
            rnl = gate(self._rnl) * self.reaction_nl_head(self.reaction_hidden(h))
            if op_multihot is not None:
                rnl = rnl * tf.reshape(op_multihot[:, OP_INDEX[self._rnl]], bshape)
            rnl = _rec(self._rnl, rnl)
            W = W + rnl
        # SPECTRAL Green's-function bank (nonlocal elliptic coupling). UNMASKED: pressure
        # projection / Poisson / potential-flow far fields span many families; gate learns.
        if self.active is None or self._spec in self.active:
            sp = gate(self._spec) * self.spec_head(self._spectral_feat(h, K, kvec=self.generic))
            if BANK_PHYSGATE and op_multihot is not None:
                sp = sp * tf.reshape(op_multihot[:, PHYSGATE_TOK["spectral"]], bshape)
            sp = _rec(self._spec, sp)
            W = W + sp
        # COEFFICIENT-FIELD bank: c(x) (x) diffusion — multiplicative material structure.
        # UNMASKED (layered media, variable viscosity, per-sample nu all qualify).
        if self.active is None or self._coef in self.active:
            cf = self.coef_mlp(h)                                       # (B,*dims,8) ~ c(x)
            if probe is not None and getattr(self, "probe_field", None) == "coefficient":
                probe["coefficient_field"] = cf.numpy()
            cl = self.gS["coefficient"](h, K, roles) if self.generic else (feats_s if "coefficient" in _lsh else feats)["diffusion"]
            feat = tf.concat([cf * self.coef_lproj(cl), cf], -1)
            co = gate(self._coef) * self.coef_head(feat)
            co = _rec(self._coef, co)
            W = W + co
        # COMPRESSIBLE bank: (1/rho)-weighted pressure/stress acceleration + u*div(u).
        # UNMASKED — the gate learns where compressibility matters (it is ~0 for div-free flow).
        if self.active is None or self._cmp in self.active:
            w = tf.nn.softplus(self.cmp_w(h))                           # (B,*dims,1) ~ 1/rho > 0
            if self.generic:
                gvl = [tf.zeros_like(h) for _ in range(3)]
                for a, y in enumerate(self.gS["cmp_gv"](h, K, roles, per_axis=True)):
                    gvl[a if a < 3 else 2] = y
                uu = self.g_uu3(self.gS["cmp_uu"](h, K, roles) * self.gL["cmp_uu"](h))
                cfeat = tf.concat([w * tf.concat(gvl, -1), w * self.gS["cmp_lap"](h, K, roles),
                                   uu, w], -1)
            else:
                divu = 0.0
                for a in range(K):
                    divu = divu + _axis_grad(u[..., a:a + 1], a)
                cfeat = tf.concat([w * feats["gradvec"],                # (1/rho) grad(state)
                                   w * (feats_s if "compressible" in _lsh else feats)["diffusion"],   # (1/rho) div(tau)
                                   u * divu, w], -1)                    # compressible self-transport
            cp = gate(self._cmp) * self.cmp_head(cfeat)
            if BANK_PHYSGATE and op_multihot is not None:
                cp = cp * tf.reshape(op_multihot[:, PHYSGATE_TOK["compressible"]], bshape)
            cp = _rec(self._cmp, cp)
            W = W + cp
        if self.single:                                     # single-head bank: one map, one gate
            L = W - W0                                      # sum_k m_k A_k F_k (KO/probe act per block)
            if self.single == "mlp":
                L = L + self.single_o(self.single_h(L))     # pointwise nonlinear coupling
            W = W0 + gate("_bank") * L
        # HYBRID GENERIC HEADS (no prescribed operator, no metadata mask, never steady-dropped)
        if self.gen_merge and not (self.active is not None and "generic" not in self.active):
            gfeat = tf.concat([tf.concat([self.gen_S[i](h, K=K, roles=roles) * self.gen_L[i](h),
                                          self._generic_spec(h, K, i)], -1)
                               for i in range(self.n_generic)], -1)        # (.., G*(d+8))
            gc = gate("generic") * self.gen_head[0](gfeat)
            gc = _rec("generic", gc)
            W = W + gc
        for i, m in enumerate([] if self.gen_merge else self._gen):
            if self.active is not None and m not in self.active:
                continue
            gfeat = tf.concat([self.gen_S[i](h, K=K, roles=roles) * self.gen_L[i](h),
                               self._generic_spec(h, K, i)], -1)
            gc = gate(m) * self.gen_head[i](gfeat)
            gc = _rec(m, gc)
            W = W + gc
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
