"""Inspect h5 structure of downloaded datasets. Usage: python -m motion_tf.data._inspect <glob-root>"""
import glob
import os
import sys

import h5py


def walk(g, pre=""):
    for k in g:
        it = g[k]
        if isinstance(it, h5py.Group):
            print("  GRP", pre + k)
            walk(it, pre + k + "/")
        else:
            print("  DSET", pre + k, it.shape, it.dtype)


def main(root):
    fs = []
    for ext in ("*.h5", "*.hdf5", "*.nc"):
        fs += glob.glob(os.path.join(root, "**", ext), recursive=True)
    fs = sorted(fs)
    print(f"root={root}  n_files={len(fs)}", flush=True)
    for p in fs[:6]:
        print("FILE", p.replace(root, ""), round(os.path.getsize(p) / 1e6, 1), "MB")
    if not fs:
        return
    with h5py.File(fs[0], "r") as f:
        print("--- structure of", fs[0].replace(root, ""), "---")
        walk(f)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/code-vol/data/prose/pdearena")
