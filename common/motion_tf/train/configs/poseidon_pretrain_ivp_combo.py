"""PHASE-2 IVP pretraining — COMBO architecture (Phase-1 best, single-shot) on Poseidon's 6-family set.
= poseidon_pretrain_ivp scaffold (T_in=1, Nt=21, all2all, CE/NS-6, rel-L1, 8-slot, train_ivp.py) with the
Phase-1 COMBO arch swapped in: single UNIFIED expert + LEARNED tendency heads (NO explicit operator bank),
adaptive per-pixel Fourier/Legendre basis mix, and a semi-Lagrangian warp — single-shot (pw_batched).
Tests the Phase-2 hypothesis: the less-fluid-specialized learned-head arch should transfer to OOD physics
(ACE, Wave) better than the old explicit-operator base while keeping the in-family NS win. Run with
train_ivp.py (NOT train_prose_mgpu). Combo arch flags below; everything else inherited (IVP/data/capacity).
"""
import os
from .poseidon_pretrain_ivp import Cfg as _Base


class Cfg(_Base):
    # ---- COMBO architecture (replaces the old 6-expert mixture) ----
    experts = ("unified",)         # single physics expert (transformer = encoder, no mixture routing)
    unified_expert = True
    lambda_gate = 0.0              # no routing → no load-balance reg
    unified_ops = False            # DROP explicit Δ/∇/advection bank → learned architecture bias only
    slot_structured = True         # clean physical channels → heads target real quantities
    unified_adapter = True         # transport/multi-scale head
    unified_reaction = True        # reaction/source head
    unified_buoyancy = True        # scalar→velocity (Boussinesq) head (∂_y seed)
    unified_wave = True            # cross-channel ∇ (wave) head
    unified_gradx = True           # horizontal-shear head (∂_x seed)
    adaptive_mix = True            # per-pixel/per-channel Fourier↔Legendre basis mix
    warp_head = True               # semi-Lagrangian IC transport (warp_steps default 1)
    # ---- single-shot fast rollout ----
    parallel_w = True
    pw_batched = True
    step_embed = True
    w_feedback = False
    save_dir = "results/" + os.environ.get("AB_SAVE", "poseidon_pretrain_ivp_combo")
