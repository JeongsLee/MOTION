"""C — Full PDE Operator Library base (transfer-favorable, Poseidon-B weight class).

Same 6 fluid pretraining families + same budget as the cont base, but with the FULL operator vocabulary:
  - FD analytic (param-free): Δ,∇,advection,ω,div (have) + Δ²,∂²xy,∂³x,∂³y,|∇| (hibank)
  - learned physics: reaction (R(u)), wave (cross-channel ∇), shock (∇·F̂), multigrid (Δ⁻¹ elliptic)
  - NO generic adapter (A ablation: no OOD transfer benefit)
Capacity offset: geo_layers 14→12 (−18.9M) → ~157M ≈ Poseidon-B (157.7M).

Hypothesis: a base whose conv head is pretrained to consume the full PDE operator vocabulary transfers to
OOD physics (ACE, Wave, ...) better than the fluid-specialized cont base — at the SAME data + weight class.

Run (2 GPU): PT_STEPS=... PT_SAVE=... bash _run_ivp_eu.sh motion_tf.train.configs.poseidon_pretrain_ivp_unified_158m_C
"""
import os
from .poseidon_pretrain_ivp_unified_158m import Cfg as _Base


class Cfg(_Base):
    geo_layers = 12                       # 14→12 (−~18.9M) to offset the operator library, stay Poseidon-B class
    # operator library
    unified_hibank = True                 # +Δ²,∂²xy,∂³,|∇| FD ops (param-free)
    unified_reaction = True               # +R(u) reaction (reaction-diffusion)
    unified_wave = True                   # +cross-channel ∇ (wave/acoustic)
    unified_shock = True                  # +∇·F̂ conservation/shock (active on compressible Euler)
    unified_multigrid = True              # +Δ⁻¹ V-cycle elliptic/Poisson (active on incompressible NS pressure)
    unified_adapter = False               # generic adapter OFF (ablation: no transfer help)
    steps = int(os.environ.get("PT_STEPS", "320000"))   # match Poseidon-B budget; lower for pilot
    if os.environ.get("PT_INIT"):
        init_from = os.environ["PT_INIT"]
    save_dir = "results/" + os.environ.get("PT_SAVE", "pretrain_ivp_158m_C")
