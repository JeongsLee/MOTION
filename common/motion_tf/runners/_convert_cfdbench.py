"""Convert CFDBench (HF chen-yingfa/CFDBench interpolated) into our cache format, matching BCAT/PROSE-FD:
per-case u/v/mask .npy (T,64,64) → [Vx,Vy,mask] → segment into 20-step chunks → bilinear 64→128 →
(N, 20, 128, 128, 3). Channels [Vx, Vy, boundary-mask]; the mask is geometry (excluded from loss). Uses
cavity + cylinder(bc,geo,prop) + tube (dam excluded, per BCAT)."""
import os, glob
import numpy as np
import torch
import torch.nn.functional as F

BASE = "/code-vol/data/prose/cfdbench_hf"
OUT = os.environ.get("PREBUILT_DIR", "/code-vol/data/prose/prebuilt")
T_NUM = 20

# every case dir = a directory containing u.npy (cavity/cylinder{bc,geo,prop}/tube; dam excluded)
cases = sorted(set(os.path.dirname(p) for p in glob.glob(BASE + "/**/u.npy", recursive=True)
                   if "/dam" not in p))
print(f"cfdbench cases (dam excluded): {len(cases)}", flush=True)

out = []
nseg_tot = 0
for c in cases:
    try:
        u = np.load(os.path.join(c, "u.npy"))               # (T,64,64)
        v = np.load(os.path.join(c, "v.npy"))
        mp = os.path.join(c, "mask.npy")
        mask = np.load(mp) if os.path.exists(mp) else np.ones_like(u[0])   # (64,64)
    except Exception:
        continue
    T = u.shape[0]
    nseg = T // T_NUM
    if nseg == 0:
        continue
    u = u[:nseg * T_NUM]; v = v[:nseg * T_NUM]
    m = np.broadcast_to(mask[None], u.shape)                # (T,64,64) static geometry
    arr = np.stack([u, v, m], -1).reshape(nseg, T_NUM, 64, 64, 3).astype(np.float32)
    t = torch.from_numpy(arr).permute(0, 1, 4, 2, 3).reshape(-1, 3, 64, 64)   # (nseg*T,3,64,64)
    t = F.interpolate(t, size=(128, 128), mode="bilinear", align_corners=False)
    arr = t.reshape(nseg, T_NUM, 3, 128, 128).permute(0, 1, 3, 4, 2).contiguous().numpy()  # (nseg,20,128,128,3)
    out.append(arr.astype(np.float32))
    nseg_tot += nseg

data = np.concatenate(out, 0)
p = os.path.join(OUT, f"cfdbench_n100000_t20.npy")
np.save(p, data)
print(f"cfdbench: {data.shape} ({nseg_tot} segments from {len(cases)} cases) -> {p}", flush=True)
print("CFDCONVERT DONE", flush=True)
