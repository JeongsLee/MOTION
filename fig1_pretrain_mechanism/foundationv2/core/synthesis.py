"""K-axis TP-ADA synthesis in TF: fixed closed-form tables, dimension-agnostic weights.

The tendency panel tensor W (B, n_1..n_K, D, N_p) lives on the latent box. Per spatial
axis a fixed Fourier projection C_a maps panels -> modes; the time axis maps panels ->
an analytic trajectory through the Legendre anti-derivative (hard IC: g(0) = 0). All
tables come from foundationv2.tpada (validated to machine precision) and enter TF as
constants — the only trainable objects here are optional separable mode gains (init 1).

    z(x, t) = sum_modes [gains * A](modes, p) * phi_modes(x) * G(t, p),
    A = W x_1 C_1 ... x_K C_K,   G = I1_eval(t) @ C_t   (G(0, :) = 0 structurally)

Continuous queries: phi evaluated at arbitrary coordinates; G at arbitrary times.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from numpy.polynomial import legendre as _LEG

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tpada.basis import Axis, AxisSpec  # noqa: E402


class KAxisTPADA(tf.Module):
    """Tables + synthesis for one latent-box geometry (one K). Weight-free except
    optional separable gains; instantiate one per K, SHARE the surrounding model."""

    def __init__(self, dims, n_time_panels, t_final=1.0, harmonics=None, time_order=24,
                 gains=True, name="tpada", spatial_ada=True, time_basis="legendre",
                 spatial_basis="global", n_patches=4, patch_order=3):
        """spatial_ada: True -> V-int (k_x=1 anti-derivative Fourier modes, DESIGN.md 2.3);
        False -> V-spec (k=0, plain Fourier). Both variants coexist as ensemble towers.
        time_basis: 'legendre' (polynomial anti-derivative, smooth growth/decay) | 'fourier'
        (anti-derivative Fourier, oscillatory-time dictionary). BOTH give hard IC G(0)=0.
        spatial_basis: 'global' (Fourier primary path, above) | 'local' (patch-Legendre
        spectral element: each axis split into n_patches patches, each carrying a COMPACT-
        SUPPORT local Legendre expansion of order patch_order -> a patch's coefficients only
        affect that patch's region = 'global IC in, local out'. Attacks local phenomena
        (shocks/walls/turbulent vortices) that global bases smear). Time-panel count / core /
        banks / decoder shapes are unchanged -> warm-safe (only the spatial gains resize)."""
        super().__init__(name=name)
        self.K = len(dims)
        self.dims = tuple(dims)
        self.Np = n_time_panels
        self.spatial_ada = bool(spatial_ada)
        self.time_basis = time_basis
        self.spatial_basis = spatial_basis
        self.n_patches = int(n_patches)
        self.patch_order = int(patch_order)
        self.axes = []
        self._C = []
        self._Ec = []
        self._cells01 = []
        for a, n in enumerate(dims):
            if spatial_basis == "local":
                # patch-Legendre spectral element: block-diagonal projection + compact-support eval
                Cl = _local_C_np(n, self.n_patches, self.patch_order)          # (P*(M+1), n)
                El = _local_modes_np(_01centers(n), self.n_patches, self.patch_order)  # (n, P*(M+1))
                self.axes.append(None)
                self._C.append(tf.constant(Cl, tf.float32))
                self._Ec.append(tf.constant(El, tf.float32))
                self._cells01.append(None)
                continue
            if spatial_basis == "localdec":
                # LOCAL DECODER (v56aq-style): the box CELLS are the coefficients (identity
                # projection); the spatial readout is a separable, per-axis NORMALISED Gaussian
                # kernel from cells to the query (Nadaraya-Watson). A learnable per-axis sigma
                # sets the write locality ('global IC read, local Gaussian write'). Dynamic (sigma
                # is trainable) -> _Ec computed on the fly in grid_traj.
                self.axes.append(None)
                self._C.append(tf.constant(np.eye(n), tf.float32))             # cells = coeffs
                self._Ec.append(None)
                self._cells01.append(tf.constant(_01centers(n), tf.float32))   # (n,) cell centers
                continue
            self._cells01.append(None)
            H = harmonics[a] if harmonics else (n // 3)          # 2/3-dealias default
            ax = Axis(AxisSpec("fourier", n, H, panel="constant"))
            # V-int (k_x=k_y=1, DESIGN.md 2.3): the TENDENCY path synthesizes from the
            # anti-derivative Fourier modes (spatial ADA, validated closed forms). The k=0
            # axis keeps the projection C (k-independent) and the field_at INTERPOLATION.
            ax_syn = Axis(AxisSpec("fourier", n, H, panel="constant",
                                   k=1 if self.spatial_ada else 0))
            self.axes.append(ax)
            self._C.append(tf.constant(ax.C, tf.float32))                     # (nc, n)
            self._Ec.append(tf.constant(ax_syn.eval(_centers(n)), tf.float32))  # (n, nc)
        # PARALLEL Legendre-ADA spatial basis (dual-basis): a polynomial anti-derivative space
        # (hard BC at xi=-1) alongside Fourier. Different function space -> non-degenerate. The model
        # gates this path (encoder-selected) so it is OFF at load (warm-safe) and ON for bounded/wall
        # families. (Trainable Jacobi alpha,beta is a later enhancement via weighted projection.)
        self._Cj, self._Icoefj, self._Nleg = [], [], []
        for a, n in enumerate(dims):
            Nl = max(2, n // 3)
            axl = Axis(AxisSpec("legendre", n, Nl, panel="constant", k=0))
            self._Cj.append(tf.constant(axl.C, tf.float32))                   # (Nl+1, n) panels->Leg
            Ic = np.stack([_LEG.legint(np.eye(Nl + 1)[m], m=1, lbnd=-1.0)
                           for m in range(Nl + 1)], -1)                       # (Nl+2, Nl+1) anti-deriv
            self._Icoefj.append(tf.constant(Ic, tf.float32))
            self._Nleg.append(Nl)
        if time_basis == "fourier":
            # oscillatory-time dictionary: Fourier harmonics over [0, t_final], k=1 anti-derivative
            # (DC -> linear ramp absorbs non-periodic drift; harmonics -> native oscillations, all
            # vanishing at t=0 -> hard IC preserved). H chosen to ~match the Legendre mode count.
            H_t = max(1, time_order // 2)
            self.taxis = Axis(AxisSpec("fourier", n_time_panels, H_t, panel="constant",
                                       k=1, domain=(0.0, float(t_final))))
        else:
            self.taxis = Axis(AxisSpec("legendre", n_time_panels, time_order, panel="constant",
                                       k=1, domain=(0.0, float(t_final))))
        self._Ct = tf.constant(self.taxis.C, tf.float32)                      # (no+1, Np)
        with self.name_scope:
            self.gains = ([tf.Variable(tf.ones([c.shape[0]]), name=f"gain_ax{a}")
                           for a, c in enumerate(self._C)] +
                          [tf.Variable(tf.ones([n_time_panels]), name="gain_t")]) if gains else None
            # Legendre-ADA path amplitude per spatial axis: init SMALL (0.1) = "open" (active &
            # learnable, NOT a zero gate), so both bases contribute from the start and the banks +
            # these gains DIFFERENTIATE per family through training (no designed selector).
            self.gains_leg = ([tf.Variable(0.1 * tf.ones([Nl + 1]), name=f"gain_leg{a}")
                               for a, Nl in enumerate(self._Nleg)] if gains else None)
            # localdec: trainable per-axis Gaussian log-sigma (init ~2 cells -> local write)
            self.kern_vars = ([tf.Variable(float(np.log(2.0 / n)), name=f"logsig{a}")
                               for a, n in enumerate(dims)] if spatial_basis == "localdec" else [])

    # -- panels -> spectral coefficients (per spatial axis) --------------------
    def coef(self, W):
        """W (B, n_1..n_K, D, Np) -> A (B, c_1..c_K, D, Np)."""
        A = W
        for a in range(self.K):
            A = tf.tensordot(A, self._C[a], axes=[[1 + a], [1]])   # coef axis lands LAST
            perm = list(range(A.shape.rank))
            perm.insert(1 + a, perm.pop())
            A = tf.transpose(A, perm)
        if self.gains is not None:
            for a in range(self.K):
                shape = [1] * A.shape.rank
                shape[1 + a] = -1
                A = A * tf.reshape(self.gains[a], shape)
            A = A * tf.reshape(self.gains[-1], [1] * (A.shape.rank - 1) + [-1])
        return A

    # -- PARALLEL Legendre-ADA path (dual basis) -------------------------------
    def coef_leg(self, W):
        """W -> A_L (Legendre-ADA coefficients), same contraction as coef but with the Legendre
        projection _Cj and the small-init 'open' Legendre gains (per spatial axis; the time gain is
        SHARED with the Fourier path). Differentiation per family emerges via W + these gains."""
        A = W
        for a in range(self.K):
            A = tf.tensordot(A, self._Cj[a], axes=[[1 + a], [1]])
            perm = list(range(A.shape.rank)); perm.insert(1 + a, perm.pop())
            A = tf.transpose(A, perm)
        if self.gains_leg is not None:
            for a in range(self.K):
                shape = [1] * A.shape.rank; shape[1 + a] = -1
                A = A * tf.reshape(self.gains_leg[a], shape)
            A = A * tf.reshape(self.gains[-1], [1] * (A.shape.rank - 1) + [-1])   # shared time gain
        return A

    def _leg_eval_at(self, a, x01):
        """Anti-derivative Legendre modes (vanishing at xi=-1) at coords x01 in [0,1] -> (..., Nl+1).
        TF Legendre Vandermonde by recurrence, then the constant anti-derivative coeffs."""
        Nl = self._Nleg[a]
        xi = tf.cast(x01, tf.float32) * 2.0 - 1.0
        V = [tf.ones_like(xi), xi]
        for n in range(1, Nl + 1):
            V.append(((2.0 * n + 1.0) * xi * V[n] - n * V[n - 1]) / (n + 1.0))
        Vmat = tf.stack(V, -1)                                       # (..., Nl+2)
        return tf.tensordot(Vmat, self._Icoefj[a], axes=[[-1], [0]])  # (..., Nl+1)

    def point_modes_leg(self, A, pts):
        evs = [self._leg_eval_at(a, pts[..., a]) for a in range(self.K)]
        return _seq_traj(A, evs, self.K)

    def point_traj_perquery_leg(self, A, pts, Gq):
        return tf.einsum("bqdp,bqp->bqd", self.point_modes_leg(A, pts), Gq)

    def field_at_leg(self, f, pts):
        """STEADY dual path: box field -> Legendre-ADA coefficients -> value at pts."""
        A = f
        for a in range(self.K):
            A = tf.tensordot(A, self._Cj[a], axes=[[1 + a], [1]])
            perm = list(range(A.shape.rank)); perm.insert(1 + a, perm.pop())
            A = tf.transpose(A, perm)
        if self.gains_leg is not None:
            for a in range(self.K):
                shape = [1] * A.shape.rank; shape[1 + a] = -1
                A = A * tf.reshape(self.gains_leg[a], shape)
        evs = [self._leg_eval_at(a, pts[..., a]) for a in range(self.K)]
        return _seq_field(A, evs, self.K)

    # -- time map: panels -> trajectory values (hard IC: row t=0 is zero) ------
    def time_map(self, times):
        """(T,) physical times -> G (T, Np) with G @ w = g1(times), g1(0)=0."""
        E = self.taxis.eval(np.asarray(times, np.float64))         # (T, no+1)
        return tf.constant(E, tf.float32) @ self._Ct               # (T, Np)

    # -- synthesis at the box centers -------------------------------------------
    def grid_traj(self, A, times):
        """A -> z (B, T, n_1..n_K, D)."""
        z = A
        for a in range(self.K):
            # localdec sigma is trainable -> build the cell->center Gaussian table on the fly
            Ec = self._kern(a, self._cells01[a]) if self.spatial_basis == "localdec" else self._Ec[a]
            z = tf.tensordot(z, Ec, axes=[[1 + a], [1]])
            perm = list(range(z.shape.rank))
            perm.insert(1 + a, perm.pop())
            z = tf.transpose(z, perm)
        G = self.time_map(times)                                   # (T, Np)
        z = tf.tensordot(z, G, axes=[[z.shape.rank - 1], [1]])     # (B, n.., D, T)
        perm = [0, z.shape.rank - 1] + list(range(1, z.shape.rank - 1))
        return tf.transpose(z, perm)

    def _kern(self, a, x01):
        """localdec: per-axis NORMALISED Gaussian from query x01 in [0,1] to the n cell centers,
        sigma = exp(logsig[a]) (trainable). Returns (..., n) partition-of-unity weights (local)."""
        c = self._cells01[a]                                       # (n,)
        sig = tf.exp(self.kern_vars[a])
        d2 = (tf.cast(x01, tf.float32)[..., None] - c) ** 2        # (..., n)
        g = tf.exp(-d2 / (2.0 * sig * sig + 1e-12))
        return g / (tf.reduce_sum(g, -1, keepdims=True) + 1e-12)   # Nadaraya-Watson normalise

    def _ev_primary(self, a, x01):
        """Tendency-path spatial modes at coords x01 in [0,1]: local -> compact-support patch
        Legendre; localdec -> Gaussian kernel over cells; global -> V-int/V-spec Fourier."""
        if self.spatial_basis == "local":
            return _eval_local_tf(x01, self.n_patches, self.patch_order)
        if self.spatial_basis == "localdec":
            return self._kern(a, x01)
        return (_eval_at_ada if self.spatial_ada else _eval_at)(self.axes[a], x01)

    def _ev_interp(self, a, x01):
        """Interpolation-path modes: local -> patch Legendre; localdec -> Gaussian kernel; global -> k=0 Fourier."""
        if self.spatial_basis == "local":
            return _eval_local_tf(x01, self.n_patches, self.patch_order)
        if self.spatial_basis == "localdec":
            return self._kern(a, x01)
        return _eval_at(self.axes[a], x01)

    # -- synthesis at arbitrary points -------------------------------------------
    def point_modes(self, A, pts):
        """Spatial synthesis at points -> zq (B, Q, D, Np) (time panels not yet contracted).
        local towers evaluate compact-support patch modes; global towers the Fourier modes."""
        evs = [self._ev_primary(a, pts[..., a]) for a in range(self.K)]
        return _seq_traj(A, evs, self.K)

    def point_traj(self, A, pts, times):
        """A, pts (B, Q, K) in [0,1]^K, times (T,) -> z (B, T, Q, D)."""
        zq = self.point_modes(A, pts)
        G = self.time_map(times)
        return tf.transpose(tf.tensordot(zq, G, axes=[[3], [1]]), [0, 3, 1, 2])

    def query_time_map(self, t_q):
        """Data-derived per-query time map: t_q (B,Q) numpy -> Gq tf constant (B,Q,Np)."""
        t_q = np.asarray(t_q, np.float64)
        B, Q = t_q.shape
        G = self.time_map(t_q.reshape(-1))                         # (B*Q, Np) tf constant
        return tf.reshape(G, [B, Q, -1])

    def point_traj_perquery(self, A, pts, Gq):
        """Per-query time: pts (B,Q,K), Gq (B,Q,Np) constant -> z (B, Q, D)."""
        zq = self.point_modes(A, pts)                              # (B,Q,D,Np)
        return tf.einsum("bqdp,bqp->bqd", zq, Gq)

    # -- band-limited field interpolation (IC anchor at query points) -----------
    def field_at(self, f, pts):
        """f (B, n_1..n_K, C) box field -> values at pts (B, Q, K) via the SAME
        spectral representation (k=0 projection + point synthesis). INTERPOLATION
        semantics — stays k=0 even under V-int (an IC anchor must reproduce f)."""
        A = self._project(f)
        evs = [self._ev_interp(a, pts[..., a]) for a in range(self.K)]
        return self._syn(A, evs)

    def field_at_ada(self, f, pts):
        """STEADY primary path: f is a LEARNED panel field (not data to interpolate) ->
        synthesize with this tower's primary spatial basis (local patch modes, or V-int/V-spec
        Fourier), mirroring field_at_leg."""
        A = self._project(f)
        evs = [self._ev_primary(a, pts[..., a]) for a in range(self.K)]
        return self._syn(A, evs)

    def _project(self, f):
        A = f
        for a in range(self.K):
            A = tf.tensordot(A, self._C[a], axes=[[1 + a], [1]])
            perm = list(range(A.shape.rank))
            perm.insert(1 + a, perm.pop())
            A = tf.transpose(A, perm)
        return A

    def _syn(self, A, evs):
        return _seq_field(A, evs, self.K)


def _seq_traj(A, evs, K):
    """Spatial synthesis (trajectory path) by SEQUENTIAL per-axis contraction, introducing the
    query axis q first. Bounds the intermediate to O(B·Q·n^{K-1}·d·p) instead of the single
    multi-operand einsum's O(B·Q·n^K·d·p) — essential for localdec (cells-as-coeffs, n large)."""
    if K == 2:
        z = tf.einsum("bnmdp,bqn->bqmdp", A, evs[0])
        return tf.einsum("bqmdp,bqm->bqdp", z, evs[1])
    z = tf.einsum("bnmldp,bqn->bqmldp", A, evs[0])
    z = tf.einsum("bqmldp,bqm->bqldp", z, evs[1])
    return tf.einsum("bqldp,bql->bqdp", z, evs[2])


def _seq_field(A, evs, K):
    """Spatial synthesis (field path) by sequential per-axis contraction (see _seq_traj)."""
    if K == 2:
        z = tf.einsum("bnmc,bqn->bqmc", A, evs[0])
        return tf.einsum("bqmc,bqm->bqc", z, evs[1])
    z = tf.einsum("bnmlc,bqn->bqmlc", A, evs[0])
    z = tf.einsum("bqmlc,bqm->bqlc", z, evs[1])
    return tf.einsum("bqlc,bql->bqc", z, evs[2])


def _centers(n):
    return -1.0 + (np.arange(n) + 0.5) * (2.0 / n)


def _01centers(n):
    return (np.arange(n) + 0.5) / n                               # panel/cell centers in [0,1]


# --------------------------------------------------------------------------- #
# LOCAL patch-Legendre spectral element (compact support: 'global IC in, local out')
# --------------------------------------------------------------------------- #
def _local_C_np(n, P, M):
    """Block-diagonal panel -> local-Legendre projection. The n panels are split into P
    contiguous patches (n/P panels each); each patch is projected onto its OWN Legendre
    basis of order M. Returns (P*(M+1), n): a patch's coefficients read only its own panels."""
    if n % P != 0:
        raise ValueError(f"n_panels {n} not divisible by n_patches {P}")
    npp = n // P
    Cp = np.asarray(Axis(AxisSpec("legendre", npp, M, panel="constant", k=0)).C, np.float64)  # (M+1, npp)
    C = np.zeros((P * (M + 1), n))
    for p in range(P):
        C[p * (M + 1):(p + 1) * (M + 1), p * npp:(p + 1) * npp] = Cp
    return C


def _local_modes_np(x01, P, M):
    """numpy: local patch-Legendre modes at coords x01 in [0,1] -> (len(x01), P*(M+1)).
    Compact support: only the block of the containing patch is nonzero."""
    x = np.clip(np.asarray(x01, np.float64), 0.0, 1.0 - 1e-9)
    pidx = np.minimum((x * P).astype(int), P - 1)
    xi = 2.0 * (x * P - pidx) - 1.0                               # local reference in [-1,1]
    V = np.stack([_LEG.legval(xi, np.eye(M + 1)[m]) for m in range(M + 1)], -1)  # (Q, M+1)
    out = np.zeros((x.shape[0], P * (M + 1)))
    for q in range(x.shape[0]):
        p = pidx[q]
        out[q, p * (M + 1):(p + 1) * (M + 1)] = V[q]
    return out


def _eval_local_tf(x01, P, M):
    """TF (differentiable, graph-friendly): local patch-Legendre modes at x01 in [0,1] ->
    (..., P*(M+1)). Legendre Vandermonde by recurrence on the per-patch local coordinate,
    scattered into the containing patch's block via a one-hot (compact support)."""
    x = tf.clip_by_value(tf.cast(x01, tf.float32), 0.0, 1.0 - 1e-6)
    scaled = x * float(P)
    pidx = tf.minimum(tf.cast(tf.floor(scaled), tf.int32), P - 1)     # (...)
    xi = 2.0 * (scaled - tf.floor(scaled)) - 1.0                      # local [-1,1]
    Vl = [tf.ones_like(xi), xi]
    for m in range(1, M):                                            # up to L_M
        Vl.append(((2.0 * m + 1.0) * xi * Vl[m] - m * Vl[m - 1]) / (m + 1.0))
    Vmat = tf.stack(Vl[:M + 1], -1)                                  # (..., M+1)
    onehot = tf.one_hot(pidx, P)                                     # (..., P)
    modes = onehot[..., None] * Vmat[..., None, :]                   # (..., P, M+1)
    return tf.reshape(modes, tf.concat([tf.shape(xi), [P * (M + 1)]], 0))


def _eval_at(axis, x01):
    """Evaluate axis modes at coordinates in [0,1] (TF tensor) -> (B, Q, nc).
    Fourier modes are elementary, so this is built directly in TF (differentiable)."""
    H = axis.spec.n_modes
    tau = tf.cast(x01, tf.float32) * 2.0                       # reference tau in [0, 2]
    n = tf.range(1, H + 1, dtype=tf.float32) * np.pi           # (H,)
    ph = tau[..., None] * n                                    # (B, Q, H)
    return tf.concat([tf.ones_like(tau)[..., None], tf.cos(ph), tf.sin(ph)], axis=-1)


def _eval_at_ada(axis, x01):
    """V-int (k=1) mode evaluation: the closed-form anti-derivatives from tau=0 of the
    Fourier modes (tpada Axis._fourier_eval order=1, TF-differentiable):
      DC -> tau,  cos(k_n tau) -> sin(k_n tau)/k_n,  sin(k_n tau) -> (1-cos(k_n tau))/k_n.
    All vanish at tau=0 (axis base point) — spatial ADA analogue of the hard-IC time map."""
    H = axis.spec.n_modes
    tau = tf.cast(x01, tf.float32) * 2.0
    n = tf.range(1, H + 1, dtype=tf.float32) * np.pi           # (H,)
    ph = tau[..., None] * n                                    # (B, Q, H)
    return tf.concat([tau[..., None], tf.sin(ph) / n, (1.0 - tf.cos(ph)) / n], axis=-1)
