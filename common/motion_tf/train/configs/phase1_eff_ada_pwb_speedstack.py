"""Phase-1 ADA — SPEED-FIRST CHAMPION: the no-recursion pw_batched (~1 s/step) stacked with ALL THREE confirmed
~1s orthogonal levers — cfdbench_geom (slot-2 cleanup) + bank_time_deriv (∂u/∂t → buoyancy) + panel_attn
(strong, panel-axis coupling). Each beat cfdgeom individually; this stacks them. The speed-first counterpart to
the accuracy-first recursive combo (rec_btd_cfd). Stays batchable/fast (panel_attn at latent res in the _pwb
path). A/B vs cfdgeom (no-recursion base) and vs the recursive variants — best fast accuracy at ~1s.
Launch (2 GPU): bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwb_speedstack
"""
import os
from .phase1_eff_ada_pwbatched_cfdgeom import Cfg as _Base


class Cfg(_Base):
    # inherits parallel_w + pw_batched + cfdbench_geom (no recursion, ~1s)
    bank_time_deriv = True       # ∂u/∂t → z0 (buoyancy)
    panel_attn = True            # panel-axis coupling (strong)
    panel_attn_dim = 64
    panel_attn_heads = 4
    panel_attn_layers = 3
    panel_attn_mlp_ratio = 2
    save_dir = "results/" + os.environ.get("SS_SAVE", "phase1_int_ada_152m_6fam_pwb_speedstack")
