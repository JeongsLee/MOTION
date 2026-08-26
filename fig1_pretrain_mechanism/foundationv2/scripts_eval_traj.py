"""Print the eval trajectory from a run's train.log (CPU job). Shows when families diverged."""
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/corpus/results/univ_p4_v8/train.log"
KEYS = ["class-avg", "airfrans", "geofno_airfoil", "geofno_elasticity", "geofno_pipe",
        "shapenet_car", "drivaernet_pressure", "Poisson-Gauss", "NS-Sines"]

rows = []
for line in open(path, errors="ignore"):
    if "[EVAL]" not in line:
        continue
    vals = {}
    m = re.search(r"class-avg rel-L2 ([0-9.]+)", line)
    if m:
        vals["class-avg"] = float(m.group(1))
    for k in KEYS[1:]:
        m = re.search(rf"{re.escape(k)}=([0-9.]+)", line)
        if m:
            vals[k] = float(m.group(1))
    rows.append(vals)

n = len(rows)
print(f"total evals: {n}  (eval_every=500 -> step ~= idx*500)")
hdr = "idx  " + "  ".join(f"{k[:9]:>9}" for k in KEYS)
print(hdr)
# print every ~4th eval + the last few
idxs = list(range(0, n, 4)) + list(range(max(0, n - 4), n))
for i in sorted(set(idxs)):
    r = rows[i]
    print(f"{i:>3}  " + "  ".join(f"{r.get(k, float('nan')):>9.1f}" for k in KEYS))
