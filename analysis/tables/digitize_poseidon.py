"""A23 (2026-10-24): digitize the scaling plots of Herde et al. (Poseidon, arXiv 2405.19101v2), SM Figs. 7-21:
median relative L1 error (paper protocol: final time, functions of interest of Table 6) vs number of fine-tuning
trajectories/samples, per model.

The figures are vector graphics. In each figure:
  * data points are small filled marker paths (3x3 pt; circles = bezier 'c' items, triangles = 'lll');
  * solid/dashed polylines are the authors' power-law fits (81) (NOT data);
  * axis tick marks are short dark (0.15 grey) line segments; tick labels are text words.
Markers are mapped to data coordinates via the tick marks (each label snapped to its tick mark): log2 axis in x,
log10 axis in y (y is shared between the two subplots; each subplot is calibrated with its own tick marks).
Colours are mapped to models via the legend swatch (marker located immediately left of each legend word).

Outputs (in this folder): poseidon_curves.json, poseidon_published_table.md, verify_fig7.png
Usage: python3 -I digitize_poseidon.py [PDF]
"""
import fitz, re, json, sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = sys.argv[1] if len(sys.argv) > 1 else "poseidon_paper.pdf"
MODELS = ("Poseidon-B", "Poseidon-L", "Poseidon-T", "FNO", "scOT", "CNO-FM", "CNO", "MPP-B")
TICK = (0.15, 0.15, 0.15)
TIME_INDEP = ("SE-AF", "Poisson-Gauss", "Helmholtz")
d = fitz.open(PDF)


def col(c):
    return None if c is None else tuple(round(v, 3) for v in c)


def ylabels(W, xmax):
    """y tick labels left of x=xmax: plain decimals ('0.010', '0.10', '0.000') or 'a × 10−k' (three words)."""
    ws = [w for w in W if w[2] <= xmax + 1]
    out, used = [], set()
    for w in ws:
        if w[4] != "×":
            continue
        a = [v for v in ws if v[2] <= w[0] + 1 and abs(v[1] - w[1]) < 3 and w[0] - v[2] < 4]
        b = [v for v in ws if v[0] >= w[2] - 1 and abs(v[1] - w[1]) < 3 and v[0] - w[2] < 4]
        if a and b:
            m = re.fullmatch(r"10([−-]?\d+)", b[0][4])
            if m:
                val = float(a[0][4]) * 10 ** int(m.group(1).replace("−", "-"))
                out.append((val, (a[0][1] + a[0][3]) / 2, a[0][4] + "x" + b[0][4]))
                used |= {id(a[0]), id(b[0])}
    for w in ws:
        if id(w) not in used and re.fullmatch(r"\d\.\d+", w[4]):
            out.append((float(w[4]), (w[1] + w[3]) / 2, w[4]))
    return out


def snap(v, cands, tol=4.0):
    if not cands:
        return None
    c = min(cands, key=lambda t: abs(t - v))
    return c if abs(c - v) < tol else None


res, diag = {}, {}
for pno, p in enumerate(d):
    caps = re.findall(r"Figure (\d+): ([A-Za-z\-]+)\. Number of (?:trajectories|samples)", p.get_text())
    if not caps:
        continue
    words = p.get_text("words")
    dr = p.get_drawings()
    capy = sorted(w[1] for w in words if w[4] == "Figure")
    for k, (fnum, task) in enumerate(caps):
        if not 7 <= int(fnum) <= 21:  # only the downstream-task scaling plots (Figs 24+ reuse task names)
            continue
        ytop = capy[k - 1] + 12 if k else 0
        ybot = capy[k]
        W = [w for w in words if ytop < w[1] < ybot]
        G = [g for g in dr if ytop < g["rect"].y0 and g["rect"].y1 < ybot + 1]
        # ---- axes boxes from the long dark spines
        hsp = [g["rect"] for g in G if col(g.get("color")) == TICK and g["rect"].height < 0.5 and g["rect"].width > 100]
        vsp = [g["rect"] for g in G if col(g.get("color")) == TICK and g["rect"].width < 0.5 and g["rect"].height > 80]
        lefts = sorted({round(r.x0, 1) for r in vsp})
        bottom = max(r.y0 for r in hsp)
        top = bottom - max(r.height for r in vsp)
        boxes = [fitz.Rect(xl, top, xl + max(r.width for r in hsp if abs(r.x0 - xl) < 1), bottom) for xl in lefts]
        assert len(boxes) == 2, (task, boxes)
        ticks = [g for g in G if col(g.get("color")) == TICK and len(g["items"]) == 1 and g["items"][0][0] == "l"
                 and max(g["rect"].width, g["rect"].height) < 4]
        # ---- legend: marker swatch immediately left of each legend word
        leg = [w for w in W if w[4] in MODELS and w[3] < top]
        markers = [g for g in G if g.get("fill") is not None and col(g["fill"]) != TICK
                   and 2 < g["rect"].width < 5 and 2 < g["rect"].height < 5]
        cmap, legids = {}, set()
        for w in leg:
            wy = (w[1] + w[3]) / 2
            sw = [g for g in markers if abs((g["rect"].y0 + g["rect"].y1) / 2 - wy) < 2.5 and 0 < w[0] - g["rect"].x1 < 40]
            sw = min(sw, key=lambda g: w[0] - g["rect"].x1)
            key = col(sw["fill"])
            assert key not in cmap, (task, w[4], key)
            cmap[key] = (w[4], "circle" if sw["items"][0][0] == "c" else "triangle", round(w[0] - sw["rect"].x1, 1))
            legids.add(id(sw))
        task_res = {}
        task_diag = {"figure": int(fnum), "page": pno + 1, "subplots": [],
                     "legend": {m: {"rgb": list(c), "marker": s, "swatch_to_text_gap_pt": gp} for c, (m, s, gp) in cmap.items()}}
        for si, box in enumerate(boxes):
            side = ["left", "right"][si]
            # x calibration: integer labels under the axis, snapped to vertical tick marks on the bottom spine
            xlab = [w for w in W if box.x0 - 5 < (w[0] + w[2]) / 2 < box.x1 + 5 and bottom < w[1] < bottom + 12
                    and re.fullmatch(r"\d+", w[4])]
            vt = [(g["rect"].x0 + g["rect"].x1) / 2 for g in ticks if g["rect"].width < 0.5
                  and abs(g["rect"].y0 - bottom) < 4 and box.x0 - 1 < g["rect"].x0 < box.x1 + 1]
            xs, xv = [], []
            for w in xlab:
                t = snap((w[0] + w[2]) / 2, vt)
                if t is not None:
                    xs.append(t); xv.append(np.log2(int(w[4])))
            ax, bx = np.polyfit(xs, xv, 1)
            xres = float(np.max(np.abs(np.polyval([ax, bx], xs) - xv)))
            # y calibration: labels left of the left subplot (shared y), snapped to this subplot's horizontal ticks
            yl = ylabels(W, boxes[0].x0)
            ht = [(g["rect"].y0 + g["rect"].y1) / 2 for g in ticks if g["rect"].height < 0.5 and abs(g["rect"].x0 - box.x0) < 1.5]
            ys, yv, ylab = [], [], []
            for val, yc, s in yl:
                t = snap(yc, ht)
                if t is not None and val > 0:
                    ys.append(t); yv.append(np.log10(val)); ylab.append(s)
            ay, by = np.polyfit(ys, yv, 1)
            yres = float(np.max(np.abs(np.polyval([ay, by], ys) - yv)))
            ign = [s for val, yc, s in yl if val <= 0]
            # ---- data markers inside this box
            pts = {}
            for g in markers:
                if id(g) in legids:
                    continue
                r = g["rect"]
                cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
                key = col(g["fill"])
                if key not in cmap or not (box.x0 - 2 < cx < box.x1 + 2 and box.y0 - 2 < cy < box.y1 + 2):
                    continue
                l2 = float(np.polyval([ax, bx], cx))
                N = int(2 ** round(l2))
                m = cmap[key][0]
                Ev = float(10 ** np.polyval([ay, by], cy))
                if N in pts.get(m, {}):  # the PDF contains a few duplicated markers at the same N: keep all, use geo-mean
                    e = pts[m][N]
                    e["all_markers"] = e.get("all_markers", [e["err"]]) + [Ev]
                    e["err"] = float(np.exp(np.mean(np.log(e["all_markers"]))))
                    e["duplicate_marker"] = True
                    continue
                pts.setdefault(m, {})[N] = {"err": Ev, "log2N_offset": round(l2 - round(l2), 3),
                                            "clipped_at_edge": bool(cy < box.y0 + 1.6 or cy > box.y1 - 1.6)}
            # ---- authors' power-law fit polylines, by stroke colour
            fits = {}
            for g in G:
                c = col(g.get("color"))
                if g.get("fill") is not None or c not in cmap or len(g["items"]) < 2:
                    continue
                r = g["rect"]
                if not (box.x0 - 2 < r.x0 and r.x1 < box.x1 + 2):
                    continue
                q = [it[1] for it in g["items"] if it[0] == "l"] + [g["items"][-1][2]]
                lx = np.polyval([ax, bx], [v.x for v in q]); ly = np.polyval([ay, by], [v.y for v in q])
                sl, ic = np.polyfit(lx * np.log10(2), ly, 1)  # log10 E = ic + sl*log10 N
                fits.setdefault(cmap[c][0], []).append(  # several segments per model for biphasic fits (82)
                    {"C": float(10 ** ic), "alpha": float(-sl), "N_range": [float(2 ** lx.min()), float(2 ** lx.max())]})
            names = sorted(set(pts) | set(fits))
            task_diag["subplots"].append({"side": side, "x_ticks": [int(round(2 ** v)) for v in xv], "x_calib_maxres_log2": round(xres, 4),
                                          "y_tick_labels_used": ylab, "y_calib_maxres_log10": round(yres, 4),
                                          "y_labels_ignored": ign, "models": names,
                                          "calib": {"log2N = ax*x + bx": [float(ax), float(bx)], "log10E = ay*y + by": [float(ay), float(by)]}})
            for m in names:
                e = task_res.setdefault(m, {"points": {}, "fit": {}})
                e["points"][side] = {str(n): v for n, v in sorted(pts.get(m, {}).items())}
                if m in fits:
                    e["fit"][side] = fits[m]
        res[task], diag[task] = task_res, task_diag

# ---- consolidate per (model, N); left subplot preferred; record left/right consistency
curves = {}
for task, tr in res.items():
    curves[task] = {"figure": diag[task]["figure"], "pdf_page": diag[task]["page"], "diagnostics": diag[task], "models": {}}
    for m, e in tr.items():
        merged, src, lr = {}, {}, []
        for side in ("left", "right"):
            for n, v in e["points"].get(side, {}).items():
                if n in merged:
                    lr.append(abs(np.log(v["err"] / merged[n])))
                else:
                    merged[n], src[n] = v["err"], side
        curves[task]["models"][m] = {
            "median_rel_L1": {n: merged[n] for n in sorted(merged, key=int)},
            "subplots_present": sorted(s for s in e["points"] if e["points"][s]),
            "left_right_max_rel_diff": (float(np.exp(max(lr)) - 1) if lr else None),
            "points_by_subplot": e["points"], "authors_powerlaw_fit": e["fit"]}

# ---- Tables 8 and 1 (text)
tasks = list(curves)
t8 = [pg for pg in d if "Table 8: Efficiency gain (EG) and Accuracy Gain (AG) for the POSEIDON" in pg.get_text()][0].get_text().split("\n")
t1 = [pg for pg in d if "Table 1: Efficiency gain EG" in pg.get_text()][0].get_text().split("\n")


def parse_table(lines, ncols):
    out = {}
    for t in tasks:
        i = lines.index(t)
        out[t] = [float(x) for x in lines[i + 1:i + 1 + ncols]]
    return out


T8 = parse_table(t8, 8)   # L EG,AG | B EG,AG | T EG,AG | FNO EG,AG
T1 = parse_table(t1, 12)  # L | CNO-FM | MPP-B | CNO | scOT | FNO  (EG, AG each)
AG = {t: {"Poseidon-L": T8[t][1], "Poseidon-B": T8[t][3], "Poseidon-T": T8[t][5], "CNO-FM": T1[t][3],
          "MPP-B": T1[t][5], "CNO": T1[t][7], "scOT": T1[t][9]} for t in tasks}
assert all(T8[t][1] == T1[t][1] for t in tasks)  # Poseidon-L AG identical in Tables 1 and 8


def E(task, m, n):
    return curves[task]["models"].get(m, {}).get("median_rel_L1", {}).get(str(n))


def Efit(task, m, n):
    fd = curves[task]["models"].get(m, {}).get("authors_powerlaw_fit", {})
    fd = fd.get("left") or fd.get("right")
    if not fd:
        return None
    seg = min(fd, key=lambda q: 0 if q["N_range"][0] * 0.99 <= n <= q["N_range"][1] * 1.01
              else min(abs(np.log(n / q["N_range"][0])), abs(np.log(n / q["N_range"][1]))))
    return seg["C"] * n ** (-seg["alpha"])


check = {}
for t in tasks:
    S = 512 if t in TIME_INDEP else 128
    ef, eff = E(t, "FNO", S), Efit(t, "FNO", S)
    row = {"S": S, "E_FNO_point": ef, "E_FNO_fitline": eff}
    for m in ("Poseidon-B", "Poseidon-L", "CNO-FM", "MPP-B", "CNO", "scOT"):
        em, fm = E(t, m, S), Efit(t, m, S)
        tab = AG[t][m]
        cand = {"pt/pt": (ef / em if ef and em else None), "fit/fit": (eff / fm if eff and fm else None),
                "FNOpt/modelfit": (ef / fm if ef and fm else None), "FNOfit/modelpt": (eff / em if eff and em else None)}
        row[m] = {"E_point": em, "E_fitline": fm, "AG_table": tab,
                  "AG_digitized": cand, "rel_dev_vs_table": {k: (v / tab - 1 if v else None) for k, v in cand.items()}}
    row["Poseidon-T"] = {"AG_table": AG[t]["Poseidon-T"], "EG_table": T8[t][4],
                         "E_derived": ef / AG[t]["Poseidon-T"] if ef else None}
    check[t] = row

# which AG recipe reproduces the tables? (summary over all models/tasks)
recipe = {}
for k in ("pt/pt", "fit/fit", "FNOpt/modelfit", "FNOfit/modelpt"):
    for grp, ts in (("time_dependent", [t for t in tasks if t not in TIME_INDEP]), ("time_independent", list(TIME_INDEP))):
        dv = [check[t][m]["rel_dev_vs_table"][k] for t in ts for m in ("Poseidon-B", "Poseidon-L", "CNO-FM", "MPP-B", "CNO", "scOT")
              if check[t][m]["rel_dev_vs_table"][k] is not None]
        a = np.abs(np.log1p(dv))
        recipe.setdefault(k, {})[grp] = {"n": len(dv), "median_abs_pct": round(100 * float(np.median(a)), 2),
                                         "max_abs_pct": round(100 * float(np.max(a)), 1), "frac_within_5pct": round(float(np.mean(a < 0.05)), 2)}

json.dump({"source": "Herde et al., Poseidon, arXiv:2405.19101v2, SM Figs 7-21 (vector-digitized data markers + authors' fit lines)",
           "units": "median relative L1 error as a fraction (paper protocol: final time, Table 6 functions of interest)",
           "keys": "tasks[task].models[model].median_rel_L1[N] = error at data marker; N = number of trajectories "
                   "(samples for SE-AF, Poisson-Gauss, Helmholtz); authors_powerlaw_fit = E = C N^-alpha segments read from the fit polylines",
           "tasks": curves, "AG_check": check, "AG_recipe_summary": recipe,
           "table8": {t: dict(zip(["L_EG", "L_AG", "B_EG", "B_AG", "T_EG", "T_AG", "FNO_EG", "FNO_AG"], T8[t])) for t in tasks},
           "table1": {t: dict(zip(["L_EG", "L_AG", "CNOFM_EG", "CNOFM_AG", "MPPB_EG", "MPPB_AG", "CNO_EG", "CNO_AG",
                                   "scOT_EG", "scOT_AG", "FNO_EG", "FNO_AG"], T1[t])) for t in tasks}},
          open(os.path.join(HERE, "poseidon_curves.json"), "w"), indent=1)


# ---- markdown table
def f(x):
    return "n/p" if x is None else (f"{x:.4f}" if x >= 0.001 else f"{x:.2e}")


def pct(r, k):
    v = r["AG_digitized"][k]
    return "n/p" if v is None else f"{v:.2f} ({100 * r['rel_dev_vs_table'][k]:+.1f}%)"


L = ["# Poseidon (Herde et al., arXiv 2405.19101v2): errors digitized from SM Figs 7-21", "",
     "Median relative L1 error as a fraction (paper protocol: final time, Table 6 functions of interest).",
     "S = 128 for time-dependent tasks, S = 512 for time-independent tasks (SE-AF, Poisson-Gauss, Helmholtz).", "",
     "Subplots: left = Poseidon-B, Poseidon-L, CNO-FM, MPP-B, FNO; right = Poseidon-B, Poseidon-L, CNO, scOT, FNO.",
     "Poseidon-B, Poseidon-L and FNO are drawn in BOTH subplots; their markers are identical in the two (max left/right",
     "difference 0.0%), values below are read from the LEFT subplot. CNO-FM/MPP-B: left; CNO/scOT: right. n/p = not plotted.", "",
     "* `pt` = data marker; `fit` = the authors' power-law fit line (81)/(82) evaluated at N=S.",
     "* The paper's AG is reproduced by AG = E_FNO(S)[marker] / E_model(S)[fit line] (see README); hence",
     "  **T @S (derived) = E_FNO(S)[marker] / AG_T(Table 8) is the value of Poseidon-T's fitted scaling law at N=S**, comparable to",
     "  the `B fit@S` / `L fit@S` columns, not to a single run. Poseidon-T is not plotted in the paper.", "",
     "| Task | Fig | S | B pt@64 | B pt@128 | B pt@512 | L pt@64 | L pt@128 | L pt@512 | B fit@S | L fit@S | FNO pt@S | **T @S (derived)** | AG_T (T8) |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for t in tasks:
    c = check[t]; S = c["S"]
    L.append(f"| {t} | {curves[t]['figure']} | {S} | {f(E(t, 'Poseidon-B', 64))} | {f(E(t, 'Poseidon-B', 128))} | "
             f"{f(E(t, 'Poseidon-B', 512)) if S == 512 else '-'} | {f(E(t, 'Poseidon-L', 64))} | {f(E(t, 'Poseidon-L', 128))} | "
             f"{f(E(t, 'Poseidon-L', 512)) if S == 512 else '-'} | {f(c['Poseidon-B']['E_fitline'])} | {f(c['Poseidon-L']['E_fitline'])} | "
             f"{f(c['E_FNO_point'])} | **{f(c['Poseidon-T']['E_derived'])}** | {AG[t]['Poseidon-T']} |")
L += ["", "## AG check: digitized AG (deviation from Table 8 / Table 1)", "",
      "| Task | AG_B T8 | AG_B FNOpt/fit | AG_B pt/pt | AG_L T1/T8 | AG_L FNOpt/fit | AG_L pt/pt |", "|---|---|---|---|---|---|---|"]
for t in tasks:
    b, l = check[t]["Poseidon-B"], check[t]["Poseidon-L"]
    L.append(f"| {t} | {b['AG_table']} | {pct(b, 'FNOpt/modelfit')} | {pct(b, 'pt/pt')} | {l['AG_table']} | {pct(l, 'FNOpt/modelfit')} | {pct(l, 'pt/pt')} |")
L += ["", "Baselines (Table 1), recipe FNOpt/modelfit:", "", "| Task | CNO-FM | MPP-B | CNO | scOT |", "|---|---|---|---|---|"]
for t in tasks:
    L.append(f"| {t} | " + " | ".join(f"{check[t][m]['AG_table']} -> {pct(check[t][m], 'FNOpt/modelfit')}" for m in ("CNO-FM", "MPP-B", "CNO", "scOT")) + " |")
L += ["", "Recipe summary (all 6 models with a Table-1/8 AG, |log ratio| in %):", "", "| recipe | group | n | median | max | within 5% |", "|---|---|---|---|---|---|"]
for k, v in recipe.items():
    for g, s in v.items():
        L.append(f"| {k} | {g} | {s['n']} | {s['median_abs_pct']} | {s['max_abs_pct']} | {s['frac_within_5pct']} |")
L += ["", "## Full digitized Poseidon-B / Poseidon-L / FNO data markers", ""]
for t in tasks:
    L.append(f"**{t}** (Fig {curves[t]['figure']}, PDF p.{curves[t]['pdf_page']})  ")
    for m in ("Poseidon-B", "Poseidon-L", "FNO"):
        mm = curves[t]["models"].get(m, {})
        v = mm.get("median_rel_L1", {})
        fl = mm.get("authors_powerlaw_fit", {}).get("left", [])
        L.append(f"- {m}: " + ", ".join(f"{n}: {f(e)}" for n, e in v.items())
                 + "  (fit: " + "; ".join(f"{q['C']:.4g} N^-{q['alpha']:.3f} on [{q['N_range'][0]:.0f},{q['N_range'][1]:.0f}]" for q in fl) + ")")
    L.append("")
open(os.path.join(HERE, "poseidon_published_table.md"), "w").write("\n".join(L) + "\n")

# ---- console diagnostics
for t in tasks:
    dg = diag[t]
    print(f"{t:14s} fig{dg['figure']} p{dg['page']} xres={[s['x_calib_maxres_log2'] for s in dg['subplots']]} "
          f"yres={[s['y_calib_maxres_log10'] for s in dg['subplots']]} ylab={dg['subplots'][0]['y_tick_labels_used']} "
          f"ign={dg['subplots'][0]['y_labels_ignored']} xt={dg['subplots'][0]['x_ticks']}")
    print("   legend:", {m: (v["marker"][0], v["swatch_to_text_gap_pt"]) for m, v in dg["legend"].items()})
    allp = [v for m in res[t].values() for s in m["points"].values() for v in s.values()]
    print("   max|log2N off|:", max(abs(v["log2N_offset"]) for v in allp), " clipped:",
          [(m, s, n) for m, e in res[t].items() for s, pp in e["points"].items() for n, v in pp.items() if v["clipped_at_edge"]])
    print("   L/R diff:", {m: (round(v["left_right_max_rel_diff"], 4) if v["left_right_max_rel_diff"] is not None else None)
                           for m, v in curves[t]["models"].items()})
    print("   Ns:", {m: list(v["median_rel_L1"]) for m, v in curves[t]["models"].items()})
    c = check[t]
    print("   AG pt/pt, FNOpt/fit, table:", {m: (round(c[m]["AG_digitized"]["pt/pt"] or -1, 2), round(c[m]["AG_digitized"]["FNOpt/modelfit"] or -1, 2), c[m]["AG_table"])
                                     for m in ("Poseidon-B", "Poseidon-L", "CNO-FM", "MPP-B", "CNO", "scOT")})

# ---- visual check: re-plot every digitized NS-PwC marker (inverse calibration) as a red ring on Fig 7 (in memory only)
pg = d[diag["NS-PwC"]["page"] - 1]
sh = pg.new_shape()
for si, sp in enumerate(diag["NS-PwC"]["subplots"]):
    (ax, bx), (ay, by) = sp["calib"]["log2N = ax*x + bx"], sp["calib"]["log10E = ay*y + by"]
    for m, e in res["NS-PwC"].items():
        for n, v in e["points"].get(sp["side"], {}).items():
            sh.draw_circle(fitz.Point((np.log2(int(n)) - bx) / ax, (np.log10(v["err"]) - by) / ay), 2.6)
sh.finish(color=(1, 0, 0), width=0.4); sh.commit()
pg.get_pixmap(dpi=300, clip=fitz.Rect(100, 70, 510, 290)).save(os.path.join(HERE, "verify_fig7_overlay.png"))
