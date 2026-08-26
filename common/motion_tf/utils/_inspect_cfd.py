import numpy as np, glob, os
d = "/eu/data/prose/prebuilt"
cands = [p for p in glob.glob(d + "/*") if "cfdbench" in os.path.basename(p).lower()]
print("CFDBENCH cache candidates:", cands, flush=True)
for p in cands[:2]:
    f = p
    if os.path.isdir(p):
        sh = sorted(glob.glob(p + "/*.npy"))
        if not sh:
            print("  empty dir", p, flush=True)
            continue
        f = sh[0]
    a = np.load(f, mmap_mode="r")
    print("FILE", f, "shape", a.shape, flush=True)
    s = np.asarray(a[0])  # (t,128,128,C)
    print("  native C =", s.shape[-1], flush=True)
    for c in range(s.shape[-1]):
        ch = s[0, :, :, c]
        u = np.unique(np.round(ch, 3))
        print("  ch%d: min=%.3f max=%.3f mean=%.3f nuniq=%d frac_gt0.99=%.3f frac_lt0.01=%.3f"
              % (c, ch.min(), ch.max(), ch.mean(), u.size, float(np.mean(ch > 0.99)), float(np.mean(ch < 0.01))), flush=True)
    # static over time? compare ch2 at t0 vs t-mid
    if s.shape[-1] >= 3:
        d2 = np.abs(s[0, :, :, 2] - s[s.shape[0] // 2, :, :, 2]).max()
        print("  ch2 |t0 - tmid| max =", float(d2), "(0 → static geometry)", flush=True)
