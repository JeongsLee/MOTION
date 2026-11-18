"""2D incompressible Navier-Stokes on the periodic unit square, pseudo-spectral vorticity form (revision R1-3; A19).
  omega_t + u . grad omega = nu Lap omega,   u = (d_y psi, -d_x psi),   Lap psi = -omega
Grid axis 0 = x (velocity component u, data slot 0), axis 1 = y (v, slot 1); NS-Sines (Poseidon) convention:
domain [0,1]^2, 128^2, 21 frames, frame interval dt_frame (fitted, ~0.05). 2/3 dealiasing, RK4, CFL-adaptive substeps.
  from ns_spectral import NS2D; s = NS2D(128, nu); traj = s.run(u0, v0, n_frames, dt_frame)
  python ns_spectral.py UNITS.pkl [FAM]   -> validation: rel L2 of the solver against the data, scan of nu and dt_frame
"""
import sys
import numpy as np


class NS2D:
    def __init__(self, n, nu, cfl=0.4):
        self.n, self.nu, self.cfl = n, nu, cfl
        k = np.fft.fftfreq(n, 1.0 / n) * 2 * np.pi                       # angular wavenumbers on [0,1)
        self.kx, self.ky = np.meshgrid(k, k, indexing="ij")
        self.k2 = self.kx ** 2 + self.ky ** 2; self.k2i = np.where(self.k2 > 0, 1.0 / np.where(self.k2 > 0, self.k2, 1), 0)
        kmax = np.abs(np.fft.fftfreq(n, 1.0 / n)).max()
        self.dealias = (np.abs(np.fft.fftfreq(n, 1.0 / n))[:, None] < 2 / 3 * kmax) & \
                       (np.abs(np.fft.fftfreq(n, 1.0 / n))[None, :] < 2 / 3 * kmax)

    def vort(self, u, v):                                                 # omega_hat = i kx v_hat - i ky u_hat
        return 1j * self.kx * np.fft.fft2(v) - 1j * self.ky * np.fft.fft2(u)

    def vel(self, wh, mean=(0.0, 0.0)):
        ph = wh * self.k2i                                                # psi_hat = omega_hat / k^2
        u = np.real(np.fft.ifft2(1j * self.ky * ph)) + mean[0]; v = np.real(np.fft.ifft2(-1j * self.kx * ph)) + mean[1]
        return u, v

    def rhs(self, wh, mean):
        u, v = self.vel(wh, mean)
        wx = np.real(np.fft.ifft2(1j * self.kx * wh)); wy = np.real(np.fft.ifft2(1j * self.ky * wh))
        return -np.fft.fft2(u * wx + v * wy) * self.dealias - self.nu * self.k2 * wh

    def run(self, u0, v0, n_frames, dt_frame):
        """Frames 0..n_frames-1 of (u, v) starting from (u0, v0); the mean velocity is conserved."""
        mean = (float(u0.mean()), float(v0.mean())); wh = self.vort(u0, v0) * self.dealias
        out = [np.stack([u0, v0], -1)]
        for _ in range(n_frames - 1):
            t = 0.0
            while t < dt_frame - 1e-12:
                u, v = self.vel(wh, mean); umax = max(np.abs(u).max(), np.abs(v).max(), 1e-6)
                h = min(self.cfl / self.n / umax, dt_frame - t)
                k1 = self.rhs(wh, mean); k2 = self.rhs(wh + 0.5 * h * k1, mean)
                k3 = self.rhs(wh + 0.5 * h * k2, mean); k4 = self.rhs(wh + h * k3, mean)
                wh = wh + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); t += h
            out.append(np.stack(self.vel(wh, mean), -1))
        return np.stack(out)


if __name__ == "__main__":
    import pickle
    U = pickle.load(open(sys.argv[1], "rb")); fam = sys.argv[2] if len(sys.argv) > 2 else "NS-Sines"
    rl = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    for i, un in enumerate(U[fam][:3]):
        F = np.asarray(un["fields"], np.float64).reshape(21, 128, 128, -1)[..., :2]
        best = None
        for nu in (0.0, 1e-4, 4e-4, 1e-3):
            for dtf in (0.045, 0.05, 0.055):
                tr = NS2D(128, nu).run(F[0, ..., 0], F[0, ..., 1], 11, dtf)
                e = [rl(tr[k], F[k]) for k in (1, 5, 10)]
                print(f"traj {i} nu {nu:.0e} dt {dtf:.3f}  relL2 frame1/5/10 {e[0]:.4f} {e[1]:.4f} {e[2]:.4f}", flush=True)
                if best is None or e[2] < best[0]: best = (e[2], nu, dtf)
        print(f"traj {i} BEST nu {best[1]:.0e} dt {best[2]}  relL2@10 {best[0]:.4f}", flush=True)
