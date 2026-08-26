import glob, os, collections, h5py
root = os.environ.get("PDEARENA_ROOT", "/code-vol/data/prose/pdearena")
fs = sorted(glob.glob(root + "/**/*.h5", recursive=True))
c = collections.Counter()
for p in fs:
    try:
        with h5py.File(p, "r") as f:
            g = f[list(f.keys())[0]]
            c[tuple(g["vx"].shape)] += 1
    except Exception as e:
        c[("ERR", type(e).__name__)] += 1
print("n_files", len(fs))
for k, v in c.most_common():
    print(f"  {v} files  vx.shape={k}")
