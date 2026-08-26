import numpy as np
z = np.load("/eu/_bylfa_cfd_condsym.npz")
P0 = z["pid"].astype(np.float64)[..., :2]; Pf = z["pfr"].astype(np.float64)[..., :2]
den = np.sqrt((P0 ** 2).sum(axis=(2, 3, 4))) + 1e-12
print("[MOTION cfdbench scalar row-mirror] EPS(mean conv) = %.3f%%"
      % ((np.sqrt(((Pf - P0) ** 2).sum(axis=(2, 3, 4))) / den).mean() * 100), flush=True)
print("DONE", flush=True)
