"""2D Allen-Cahn on the periodic unit square (revision R1-3; mechanism-knob reference for ACE).
  u_t = sd * Lap u - sr * eps^2 * u (u^2 - 1)        (Poseidon ACE: eps = 220, sd = sr = 1, T = 2e-4, 128^2)
Strang splitting: half-step exact pointwise reaction, full-step exact spectral diffusion, half-step reaction.
  from ac_spectral import AC2D; traj = AC2D(128, sd=1, sr=1).run(u0, n_frames, dt_frame)
  python ac_spectral.py UNITS.pkl   -> validation against the data (scan of the frame interval)
"""
import sys
import numpy as np


class AC2D:
    def __init__(self, n, eps=220.0, sd=1.0, sr=1.0, sub=40):
        self.n, self.r, self.sd, self.sub = n, sr * eps * eps, sd, sub
        k = np.fft.fftfreq(n, 1.0 / n) * 2 * np.pi
        kx, ky = np.meshgrid(k, k, indexing="ij"); self.k2 = kx ** 2 + ky ** 2

    def _react(self, u, t):                     # exact solution of u_t = -r (u^3 - u) over time t
        if self.r == 0: return u
        e = np.exp(-2 * self.r * t)
        return u / np.sqrt(u * u + (1 - u * u) * e)

    def run(self, u0, n_frames, dt_frame):
        u = np.asarray(u0, np.float64).copy(); out = [u.copy()]; h = dt_frame / self.sub
        dif = np.exp(-self.sd * self.k2 * h)
        for _ in range(n_frames - 1):
            for _ in range(self.sub):
                u = self._react(u, h / 2)
                u = np.real(np.fft.ifft2(np.fft.fft2(u) * dif))
                u = self._react(u, h / 2)
            out.append(u.copy())
        return np.stack(out)


if __name__ == "__main__":
    import pickle
    U = pickle.load(open(sys.argv[1], "rb")); fam = sys.argv[2] if len(sys.argv) > 2 else "ACE"
    rl = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    for i, un in enumerate(U[fam][:3]):
        f = np.asarray(un["fields"], np.float64); T = f.shape[0]; F = f.reshape(T, 128, 128, -1)[..., 0]
        for dtf in (2e-4 / 20, 2e-4 / 19):
            tr = AC2D(128).run(F[9], 11, dtf)
            pers = [rl(F[9], F[9 + k]) for k in (1, 5, min(10, T - 10))]
            e = [rl(tr[k], F[9 + k]) for k in (1, 5, min(10, T - 10))]
            print(f"traj {i} T={T} dt {dtf:.3e}  relL2 +1/+5/+10 {e[0]:.4f} {e[1]:.4f} {e[2]:.4f}  (persistence {pers[0]:.3f} {pers[1]:.3f} {pers[2]:.3f})", flush=True)


class FHN2D:
    """FitzHugh-Nagumo reaction-diffusion (PDEBench / PROSE diff_react) on a periodic square of side L:
      u_t = sd Du Lap u + sr (u - u^3 - k - v),   v_t = sd Dv Lap v + sr (u - v)
    Strang splitting: exact spectral diffusion, RK4 pointwise reaction."""
    def __init__(self, n, L=2.0, Du=1e-3, Dv=5e-3, k=5e-3, sd=1.0, sr=1.0, sub=20, bc="neumann"):
        # PDEBench diff_react (identified from the data by regression, R^2 0.999): Neumann BC, L = 2, dt_frame = 0.05
        self.Du, self.Dv, self.k, self.sr, self.sub, self.bc = sd * Du, sd * Dv, k, sr, sub, bc
        if bc == "neumann":                         # cell-centred Neumann: DCT-II eigenvalues (2/dx sin(pi m / 2n))^2
            dx = L / n; m = np.arange(n); lam = (2 / dx * np.sin(np.pi * m / (2 * n))) ** 2
            self.k2 = lam[:, None] + lam[None, :]
        else:
            kk = np.fft.fftfreq(n, L / n) * 2 * np.pi
            kx, ky = np.meshgrid(kk, kk, indexing="ij"); self.k2 = kx ** 2 + ky ** 2

    def _diffuse(self, q, f):
        if self.bc == "neumann":
            from scipy.fft import dctn, idctn
            return idctn(dctn(q, type=2, norm="ortho") * f, type=2, norm="ortho")
        return np.real(np.fft.ifft2(np.fft.fft2(q) * f))

    def _R(self, u, v):
        return self.sr * (u - u ** 3 - self.k - v), self.sr * (u - v)

    def _react(self, u, v, h):
        a1 = self._R(u, v); a2 = self._R(u + h / 2 * a1[0], v + h / 2 * a1[1])
        a3 = self._R(u + h / 2 * a2[0], v + h / 2 * a2[1]); a4 = self._R(u + h * a3[0], v + h * a3[1])
        return (u + h / 6 * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0]), v + h / 6 * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1]))

    def run(self, uv0, n_frames, dt_frame):
        u, v = (np.asarray(uv0[..., 0], np.float64).copy(), np.asarray(uv0[..., 1], np.float64).copy())
        out = [np.stack([u, v], -1)]; h = dt_frame / self.sub
        du, dv = np.exp(-self.Du * self.k2 * h), np.exp(-self.Dv * self.k2 * h)
        for _ in range(n_frames - 1):
            for _ in range(self.sub):
                u, v = self._react(u, v, h / 2)
                u, v = self._diffuse(u, du), self._diffuse(v, dv)
                u, v = self._react(u, v, h / 2)
            out.append(np.stack([u, v], -1))
        return np.stack(out)
