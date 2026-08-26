"""Download + preprocess PROSE-FD datasets directly onto the (mounted) VESSL volume.

Run as a CPU-only VESSL job (no GPU). Sources:
  - PDEBench (DaRUS): SWE, comp NS (Rand+Turb), incom NS — via the official URL CSV.
  - PDEArena (HF): NavierStokes-2D (+conditioned).
  - CFDBench: DPOT source (deferred — fiddly Google-Drive; add later).
Preprocess: comp NS 512→128 (mean-pool by 4), matching PROSE-FD's convert_com_ns.

Env: PROSE_DATA_DIR = volume mount target (e.g. /code-vol/data/prose).
Usage: python -m motion_tf.data.download_prose --datasets pdearena_ns,com_ns --n_per 8
"""
from __future__ import annotations
import argparse
import csv
import io
import os
import shutil
import time
import urllib.request

import numpy as np

ROOT = os.environ.get("PROSE_DATA_DIR", "/code-vol/data/prose")
PDEBENCH_CSV = "https://raw.githubusercontent.com/pdebench/PDEBench/main/pdebench/data_download/pdebench_data_urls.csv"


def _csv_rows():
    with urllib.request.urlopen(PDEBENCH_CSV, timeout=60) as r:
        text = r.read().decode()
    return list(csv.reader(io.StringIO(text)))


def _dl(url, dst, retries=12):
    """Robust download for the ~10GB PDEBench files: RESUME a partial .part via HTTP Range + RETRY on
    drop (urllib.urlretrieve had no retry/resume → large incom files failed mid-stream). If the server
    ignores Range (returns 200 not 206) we restart that file from 0."""
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"  skip (exists): {dst}", flush=True); return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    part = dst + ".part"
    for attempt in range(retries):
        pos = os.path.getsize(part) if os.path.exists(part) else 0
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={pos}-"} if pos else {})
            with urllib.request.urlopen(req, timeout=120) as r:
                mode = "ab" if (pos and r.status == 206) else "wb"   # 206 → append; else restart
                if mode == "wb":
                    pos = 0
                with open(part, mode) as f:
                    shutil.copyfileobj(r, f, 1024 * 1024)
            os.rename(part, dst)
            print(f"  done {os.path.getsize(dst)/1e9:.2f} GB  {os.path.basename(dst)}", flush=True)
            return
        except Exception as e:                                       # drop/timeout → retry, keep .part
            print(f"  retry {attempt+1}/{retries} (@{pos/1e9:.1f}GB): {e}", flush=True)
            time.sleep(min(5 * (attempt + 1), 30))
    print(f"  FAILED after {retries} retries: {url}", flush=True)


def pdebench(prefix_match, subdir, n_per=None, skip_512=False):
    """Download PDEBench files whose target subdir matches; optional cap n_per.
    skip_512: skip 512-resolution files — they need 512->128 downsampling which loads the whole
    (~88GB) file into RAM and OOMs a 13GB CPU node. The native-128 files are used directly."""
    rows = _csv_rows()
    sel = [r for r in rows if len(r) >= 4 and prefix_match in r[3]]
    if skip_512:
        sel = [r for r in sel if "512" not in r[1]]
    if n_per:
        sel = sel[:n_per]
    for _, fname, url, sub, *_ in sel:
        _dl(url, os.path.join(ROOT, "pdebench", sub.strip("/"), fname))


def downsample_512_to_128(folder):
    """PROSE convert_com_ns: mean-pool 512→128 for files with '512' in the name."""
    import h5py
    for f in os.listdir(folder):
        if "512" not in f or "128" in f:
            continue
        src = os.path.join(folder, f); tgt = os.path.join(folder, f.replace("512", "128"))
        if os.path.exists(tgt):
            continue
        print(f"  downsample {f} -> 128", flush=True)
        with h5py.File(src, "r") as a, h5py.File(tgt, "w") as b:
            for k in a.keys():
                if k == "t-coordinate":
                    b.create_dataset(k, data=np.array(a[k]))
                elif k.endswith("coordinate"):
                    b.create_dataset(k, data=np.mean(np.array(a[k]).reshape(-1, 4), axis=1))
                else:
                    d = np.array(a[k])
                    d = d.reshape(d.shape[0], d.shape[1], d.shape[2]//4, 4, d.shape[3]//4, 4).mean((-3, -1))
                    b.create_dataset(k, data=d)


def pdearena(n_files=None):
    from huggingface_hub import hf_hub_download, HfApi
    api = HfApi()
    for repo in ["pdearena/NavierStokes-2D", "pdearena/NavierStokes-2D-conditoned"]:
        sub = repo.split("/")[-1]
        files = [f.rfilename for f in api.repo_info(repo, repo_type="dataset", files_metadata=True).siblings
                 if f.rfilename.endswith((".h5", ".hdf5", ".nc"))]
        if n_files:
            files = files[:n_files]
        for fn in files:
            hf_hub_download(repo, fn, repo_type="dataset",
                            local_dir=os.path.join(ROOT, "pdearena", sub))
            print(f"  [pdearena] {sub}/{fn}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="com_ns")
    ap.add_argument("--n_per", type=int, default=0)   # 0 = all
    args = ap.parse_args()
    np_cap = args.n_per or None
    ds = args.datasets.split(",")
    print(f"ROOT={ROOT}  datasets={ds}  n_per={np_cap}", flush=True)
    if "com_ns" in ds:
        # 128-native only (skip 512 — downsampling 88GB files OOMs a 13GB node; native-128 suffices)
        pdebench("2D/CFD/2D_Train_Rand", "", np_cap, skip_512=True)
        pdebench("2D/CFD/2D_Train_Turb", "", np_cap, skip_512=True)
    if "incom_ns" in ds:
        pdebench("2D/NS_incom", "", np_cap)
    if "diff_react" in ds:
        pdebench("2D/diffusion-reaction", "", np_cap)   # 2D_diff-react_NA_NA.h5 (single file)
    if "pdearena_ns" in ds:
        pdearena(np_cap)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
