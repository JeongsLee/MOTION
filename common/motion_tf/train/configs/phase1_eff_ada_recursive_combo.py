"""Phase-1 ADA — ACCURACY-FIRST COMBO: the RECURSIVE baseline (z-Euler recursion kept = max accuracy, no
speed sacrifice) + the three CONFIRMED orthogonal improvements stacked:
  • cfdbench_geom : CFDBench boundary → geom_mask (slot-2 cleanup; data-level, helps pdearena)
  • bank_time_deriv: z0 = ic_enc([last, ∂u/∂t]) → bank sees acceleration (buoyancy; pdea −1.7 @4000 vs cfdgeom)
  • panel_attn (strong): bidirectional panel-axis attention coupling the per-panel W (now wired into the
    recursive while_loop path too — applied to the stacked W at 128²).
NO parallel_w / pw_batched / fixedpoint — this is the full-accuracy recursive model, aiming to BEAT the
4.43% baseline by adding the recursion-orthogonal physics/data gains. Slower (~recursive + attn) but accuracy
is the objective. A/B vs the plain recursive baseline (job-q6029b64qnt9, 4.43%).
Launch (2 GPU): bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_recursive_combo
"""
import os
from .phase1_eff_ada import Cfg as _Base


class Cfg(_Base):
    # recursive baseline kept (parallel_w / pw_batched / fixedpoint all OFF by inheritance)
    cfdbench_geom = True
    bank_time_deriv = True
    panel_attn = True
    panel_attn_dim = 64
    panel_attn_heads = 4
    panel_attn_layers = 3
    panel_attn_mlp_ratio = 2
    save_dir = "results/" + os.environ.get("COMBO_SAVE", "phase1_int_ada_152m_6fam_recursive_combo")
