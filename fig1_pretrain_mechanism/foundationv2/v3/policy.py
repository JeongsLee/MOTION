"""Per-family training policy for the MOTION-NCS run (motion_ncs/PLAN_NCS.md, 2026-08-06).

Two measured facts fix the temporal mode per family (paper 1 §4.4 + user observation 08-06):
  * AR depth is monotone on advection-dominated families (bylfa: 3.175 -> 2.774 going 2 -> 10
    segments) -> full AR, one frame per call.
  * Smooth families prefer NO autoregression: a single whole-horizon ADA segment sees the
    global temporal structure at once, and for second-order-in-time physics (Wave-Layer) the
    per-frame re-anchor actively hurts because it discards the velocity state.
Poisson-Gauss is steady and keeps the K_relax pseudo-transient loop (its own third mode).

The panel count does NOT vary per family: panels are Dense width here (out = d_w * n_p), so
n_p = 16 serves both modes at the same matmul cost — an AR call oversamples its one-frame
segment (harmless) while a single-shot call gets 1.6 panels/frame over the 10-frame horizon,
exactly the measured minimum (07-20 sweep). What differs per family is the MODE, which is
where the wall-clock lives: single-shot runs core+banks ONCE per sample instead of 10x.
"""
from __future__ import annotations

# "ar"     — full AR: L=1 frame per segment, K = horizon model calls (bylfa paradigm)
# "ss"     — single-shot: one model call, panels span the horizon, decode all frames
# "steady" — K_relax pseudo-transient relaxation (registry time == "steady" wins anyway)
AR_MODE = {
    "shallow_water": "ar", "com_ns": "ar", "incom_ns": "ar",
    "pdearena_ns": "ar", "pdearena_uncond": "ar",
    "CE-RP": "ar", "CE-CRP": "ar", "CE-KH": "ar", "CE-Gauss": "ar", "CE-RM": "ar",
    "NS-Sines": "ar", "NS-Gauss": "ar",
    "pdebench3d_cns_Rand_M0.1_Eta1e-08_Zeta1e-08": "ar",
    "pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08": "ar",
    "pdebench3d_cns_Turb_M1.0_Eta1e-08_Zeta1e-08": "ar",
    "ACE": "ss", "diff_react": "ss", "Wave-Layer": "ss",
    "Poisson-Gauss": "steady",
}


def ar_mode(family, steady):
    if steady:
        return "steady"
    return AR_MODE.get(family, "ar")
