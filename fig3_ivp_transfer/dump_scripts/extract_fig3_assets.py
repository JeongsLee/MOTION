"""Extract Fig3 render inputs to /seoul: training curves (a,b) + selected contour
samples (d). Selection logic copied verbatim from render_fig3d_v4.py."""
import re, json, numpy as np, os

def parse_ours(p):
    S = []
    for L in open(p, errors="ignore"):
        m = re.search(r"\[EVAL step (\d+)\].*overall=([\d.]+)%", L)
        if m: S.append((int(m.group(1)) * 8, float(m.group(2))))
    return S

FAMS = ["shallow_water", "com_ns", "incom_ns", "pdearena_ns", "cfdbench", "pdearena_uncond"]
INV = {"shallow_water": "shallow_water", "com_ns": "com_ns", "incom_ns": "incom_ns",
       "incom_ns_arena": "pdearena_ns", "cfdbench": "cfdbench",
       "incom_ns_arena_u": "pdearena_uncond"}

def parse_fair(p):
    ep, per = -1, {}
    for L in open(p, errors="ignore"):
        m = re.search(r"End of epoch (\d+)", L)
        if m: ep = int(m.group(1)); continue
        m = re.search(r"FULLTEST_RELNORM\s+(\S+)\s+n=100\s+mean=([\d.]+)%", L)
        if m and m.group(1) in INV:
            per.setdefault(ep, {}).setdefault(INV[m.group(1)], []).append(float(m.group(2)))
    S = []
    for e in sorted(per):
        d = per[e]
        if e < 0 or not all(f in d for f in FAMS): continue
        S.append(((e + 1) * 16000, float(np.mean([d[f][0] for f in FAMS]))))
    return S

curves = {
    "ours": parse_ours("/eu/results/prose_150M_fluid5_6x_unified_task2_advb2_arh5_bylfa/train.log"),
    "bcat": parse_fair("/eu/results/p1_bcat_fair/v1/train.log"),
    "prose": parse_fair("/eu/results/p1_prose_fair/v1/train.log"),
}
os.makedirs("/seoul/fig3_assets", exist_ok=True)
json.dump(curves, open("/seoul/fig3_assets/curves.json", "w"))
print("curves saved:", {k: len(v) for k, v in curves.items()}, flush=True)

ROWS = [("shallow_water", "shallow_water", [5], 0, 10, "maxerr"),
        ("com_ns", "com_ns", [0, 1, 3, 4], 3, 10, "maxerr"),
        ("pdearena_ns", "incom_ns_arena", [0, 1, 2], 2, 10, "maxstd"),
        ("cfdbench", "cfdbench", [0, 1, 2], 0, 10, "maxerr")]
out = {}
for fa, fb, slots, pc, F, rule in ROWS:
    A = np.load(f"/eu/_bfa_{fa}.npz")
    aP = A["pred"][..., slots][:, :F]; aG = A["gt"][..., slots][:, :F]
    Bz = np.load(f"/eu/_bcatf_{fb}.npz"); nc = len(slots)
    bP = Bz["pred"][..., :nc][:, :F]
    ns = min(len(aP), len(bP)); fr = F - 1
    def rl2(p, i):
        o = p[i, fr]; t = aG[i, fr]
        return 100 * np.sqrt(((o - t) ** 2).sum()) / np.sqrt((t ** 2).sum())
    if rule == "maxstd":
        s = int(np.argmax([aG[i, fr, ..., pc].std() for i in range(ns)]))
    else:
        s = int(np.argmax([rl2(aP, i) + rl2(bP, i) for i in range(ns)]))
    out[f"{fa}_gt"] = aG[s, fr, ..., pc]
    out[f"{fa}_a"] = aP[s, fr, ..., pc]
    out[f"{fa}_b"] = bP[s, fr, ..., pc]
    out[f"{fa}_rla"] = np.float32(rl2(aP, s))
    out[f"{fa}_rlb"] = np.float32(rl2(bP, s))
    print(f"{fa}: sample={s} rla={rl2(aP,s):.2f} rlb={rl2(bP,s):.2f}", flush=True)
np.savez_compressed("/seoul/fig3_assets/fig3d_sel.npz", **{k: np.asarray(v, np.float32) for k, v in out.items()})
print("EXTRACT_DONE", flush=True)
