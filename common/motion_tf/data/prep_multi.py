"""Pre-build per-family native .npy caches ONCE (run as a cheap CPU job), so GPU training jobs
load them in seconds instead of re-reading 68GB of raw multi-file data every launch.

    PROSE_DATA_DIR=/code-vol/data/prose PDEARENA_ROOT=/code-vol/data/prose/pdearena \
      python -m motion_tf.data.prep_multi --datasets shallow_water,pdearena_ns,diff_react --n_per 900

RAM-safe: processes one family at a time (≤~3.5GB), never the full 6-slot concat. Cache lands in
PROSE_DATA_DIR/prebuilt/<family>_n<N>_t<T>.npy and is reused by build_multi(cache_dir=...).
"""
from __future__ import annotations
import argparse
import os

from . import prose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="shallow_water,pdearena_ns,diff_react")
    ap.add_argument("--n_per", type=int, default=900)
    ap.add_argument("--t_num", type=int, default=20)
    ap.add_argument("--cache_dir", default=None)
    a = ap.parse_args()
    cache_dir = a.cache_dir or os.path.join(prose.PROSE_ROOT, "prebuilt")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"prep: datasets={a.datasets} n_per={a.n_per} t_num={a.t_num} -> {cache_dir}", flush=True)
    import glob as _glob
    for ds in a.datasets.split(","):
        fp = prose._ensure_cache(ds, a.n_per, a.t_num, cache_dir)     # writes if absent, NO full load
        if os.path.isdir(fp):                                        # pdearena = shard dir
            n_sh = len(_glob.glob(os.path.join(fp, "shard_*.npy")))
            print(f"  {ds}: cached {n_sh} shards -> {fp}", flush=True)
        else:
            print(f"  {ds}: cached {prose._npy_shape(fp)}", flush=True)
    print("PREP DONE", flush=True)


if __name__ == "__main__":
    main()
