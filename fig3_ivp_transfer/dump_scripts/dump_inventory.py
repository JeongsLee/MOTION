"""What per-trajectory prediction dumps survive on /eu?  Each holds pred and gt for one
(model, transform, family), so per-trajectory error and defect follow without re-running."""
import glob, os, numpy as np
pats = ["/eu/_bcatsymE_*.npz", "/eu/_prosesymE_*.npz", "/eu/_bfa_*.npz", "/eu/*sym*.npz"]
seen = {}
for p in pats:
    for f in sorted(glob.glob(p)):
        b = os.path.basename(f)
        if b in seen: continue
        try:
            z = np.load(f, mmap_mode="r")
            seen[b] = {k: tuple(z[k].shape) for k in z.files}
        except Exception as e:
            seen[b] = f"unreadable: {type(e).__name__}"
for b, v in seen.items(): print(f"{b:46s} {v}")
print(f"--- {len(seen)} dumps")
print("INVENTORY_DONE", flush=True)
