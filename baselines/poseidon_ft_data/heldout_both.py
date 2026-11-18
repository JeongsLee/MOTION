"""REVISION-R1 10-24: trajectories of the six pretraining families that NEITHER model trained on.
MOTION-IVP-S: hash split of its manifests (/corpus/meta/splits/<fam>.json, entries "chunk:row"; chunk = index of the
file in lexicographic order, rows of a chunk follow the previous chunks, as in the HF assembly). Poseidon-T: trained on
the original indices [0, N_max - 360); official val = next 120, test = last 240.
Held out by both = (MOTION val or test) and (Poseidon val or test). Written to /corpus/meta/splits/heldout_both/<fam>.json
with the MOTION entries (for score_ivp_l1.py) and the original indices + Poseidon split (for the scOT per-trajectory
results, row in the pre6 file = 1 + orig - (N_max - 360))."""
import h5py, glob, json, os
FILES = {"CE-RP": "/eu/data/poseidon/_assembled/CE-RP.nc", "CE-CRP": "/eu/data/poseidon/_assembled/CE-CRP.nc",
         "CE-KH": "/eu/data/poseidon/_assembled/CE-KH.nc", "CE-Gauss": "/eu/data/poseidon/_assembled/CE-Gauss.nc",
         "NS-Sines": "/eu/data/poseidon/_assembled/NS-Sines.nc", "NS-Gauss": "/eu/data/poseidon_raw/NS-Gauss"}
NMAX = {"CE-RP": 10000, "CE-CRP": 10000, "CE-KH": 10000, "CE-Gauss": 10000, "NS-Sines": 20000, "NS-Gauss": 20000}
OUT = "/corpus/meta/splits/heldout_both"; os.makedirs(OUT, exist_ok=True)
for fam, p in FILES.items():
    fs = sorted(glob.glob(p + "/*.nc")) if os.path.isdir(p) else [p]
    off, o = [], 0
    for f in fs:
        with h5py.File(f, "r") as h:
            k = max((k for k in h if hasattr(h[k], "ndim")), key=lambda k: h[k].ndim); off.append(o); o += h[k].shape[0]
    man = json.load(open(f"/corpus/meta/splits/{fam}.json"))
    orig = lambda e: off[int(e.split(":")[0])] + int(e.split(":")[1])
    pv0, pt0 = NMAX[fam] - 360, NMAX[fam] - 240
    psplit = lambda g: "train" if g < pv0 else ("val" if g < pt0 else "test")
    keep = [(e, orig(e), sp, psplit(orig(e))) for sp in ("val", "test") for e in man["splits"][sp] if orig(e) >= pv0]
    ntr_in = sum(orig(e) >= pv0 for e in man["splits"]["train"])
    res = {"family": fam, "rows_total": o, "N_max": NMAX[fam],
           "entries": [k[0] for k in keep], "orig_index": [k[1] for k in keep],
           "motion_split": [k[2] for k in keep], "poseidon_split": [k[3] for k in keep],
           "note": "held out by both MOTION-IVP-S (hash val/test) and Poseidon (official val/test)"}
    json.dump(res, open(f"{OUT}/{fam}.json", "w"))
    c = {(a, b): sum(1 for k in keep if (k[2], k[3]) == (a, b)) for a in ("val", "test") for b in ("val", "test")}
    print(f"{fam:9s} rows {o:6d}  held-out-by-both {len(keep):3d}  (MOTION,Poseidon) {c}  "
          f"MOTION-train rows inside Poseidon val/test {ntr_in}", flush=True)
print("HELDOUT_DONE", flush=True)
