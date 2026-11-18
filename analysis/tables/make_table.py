# SI table for Supplementary Note 18 from v5_h0.json.
import json
R = {r['fam']: r for r in json.load(open('v5_h0.json'))}
NAMES = [("shallow_water", "SWE"), ("diff_react", "Diff-React"), ("com_ns", "Comp. NS"), ("incom_ns", "Incomp. NS"),
         ("pdearena_ns", "PDEArena"), ("pdearena_uncond", "PDEArena-u"), ("CE-RP", "CE-RP"), ("CE-CRP", "CE-CRP"),
         ("CE-KH", "CE-KH"), ("CE-Gauss", "CE-Gauss"), ("CE-RM", "CE-RM"), ("NS-Sines", "NS-Sines"),
         ("NS-Gauss", "NS-Gauss"), ("ACE", "ACE"), ("Wave-Layer", "Wave-Layer"), ("Poisson-Gauss", "Poisson"),
         ("pdebench3d_cns_Rand_M0.1_Eta1e-08_Zeta1e-08", "3D M0.1"), ("pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08", "3D M1.0"),
         ("pdebench3d_cns_Turb_M1.0_Eta1e-08_Zeta1e-08", "3D Turb")]
def p(x): return "$<$0.001" if x < 1e-3 else f"{x:.3f}"
def ko(r, h):
    a, b, q = r['ko_' + h]
    if abs(a) < 0.5 and abs(b) < 0.5: return "---"
    f = lambda v: "0" if round(v) == 0 else f"{v:+.0f}"
    return f"${f(a)}\\rightarrow{f(b)}$" + ("" if q < 0.05 else "$^{\\mathrm{ns}}$")
out = []
for k, n in NAMES:
    r = R[k]
    out.append(f"{n} & {r['n']} & {r['h0share']:.2f} & {r['err_br']:.2f} & {r['err_v5']:.2f} & ${r['rel']:+.1f}$ & {p(r['p_holm'])} & "
               f"{ko(r,'reaction')} & {ko(r,'reaction_nl')} & {ko(r,'coefficient')}\\\\")
open('table_s18.tex', 'w').write("\n".join(out) + "\n")
print("\n".join(out))
