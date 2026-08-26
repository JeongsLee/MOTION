"""Download the remaining PDEBench 2D incompressible-NS files (ns_incom_inhom_2d_512-89..274) from DaRUS to
the volume, resuming the interrupted download (88/274 present). Reads the resume TSV (filename<TAB>url<TAB>md5),
curl -L -C - each (follow DaRUS 303→S3 redirect, byte-resume). Skips files already >9GB."""
import os, subprocess
TSV = os.environ.get("TSV", "/code-vol/data/prose/ns_incom_resume.tsv")
DST = os.environ.get("DST", "/code-vol/data/prose/pdebench/2D/NS_incom")
SKIP = set(x for x in os.environ.get("SKIP", "").split(",") if x)   # filenames to skip (DaRUS-bad files)
os.makedirs(DST, exist_ok=True)
rows = []
with open(TSV) as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2 and p[1].startswith("http"):
            rows.append((p[0], p[1]))
print(f"resume set: {len(rows)} files -> {DST}", flush=True)
done = 0
for i, (fn, url) in enumerate(rows, 1):
    dst = os.path.join(DST, fn)
    if os.path.exists(dst) and os.path.getsize(dst) > 9_000_000_000:
        continue
    if fn in SKIP:
        print(f"[{i}/{len(rows)}] {fn} — SKIPPED (DaRUS-bad)", flush=True); continue
    print(f"[{i}/{len(rows)}] {fn}", flush=True)
    sz = 0
    for attempt in range(30):                                  # auto-recover DaRUS stalls/drops (server-side)
        subprocess.run(["curl", "-L", "-C", "-", "-sS",
                        "--connect-timeout", "60",
                        "--speed-limit", "100000", "--speed-time", "120",  # abort if <100KB/s for 120s (hung)
                        "--retry", "5", "--retry-delay", "10", "--retry-all-errors",
                        "-o", dst, url])
        sz = os.path.getsize(dst) / 1e9 if os.path.exists(dst) else 0
        if sz > 9:
            break
        print(f"  attempt {attempt+1}: {sz:.2f}GB (stalled/incomplete) — resuming", flush=True)
    print(f"  -> {sz:.2f}GB", flush=True)
    if sz > 9:
        done += 1
tot = len([x for x in os.listdir(DST) if x.startswith("ns_incom_inhom_2d_512-") and x.endswith(".h5")])
print(f"=== downloaded {done} this run; NS_incom total = {tot} files ===", flush=True)
print("INCOMDL_DONE", flush=True)
