"""Phase-1 ADA — archbias + UPWIND advection-bias head (#3) + strong panel_attn (#4). Extends the session-best
archbias (learned multi-bias-heads, Order A, no explicit ops, no recursion) with two parallel additions:
  #3 advbias : a learned UPWIND/directional advection-bias head (sign-aware upwind −(u·∇^up)z = the paper's
               upwind-conv architectural bias, learned head, additive zero-init branch).
  #4 panel_attn (strong): bidirectional panel-axis attention coupling the N_p panels.
Both target the remaining PDEArena gap (forced advection cascade) on the frozen-IC no-recursion backbone.
A/B vs archbias.  Launch (2 GPU): bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwb_archbias_adv
"""
import os
from .phase1_eff_ada_pwb_archbias import Cfg as _Base


class Cfg(_Base):
    unified_advbias = True       # #3 learned upwind advection-bias head (parallel branch)
    panel_attn = True            # #4 panel-axis attention (strong)
    panel_attn_dim = 64
    panel_attn_heads = 4
    panel_attn_layers = 3
    panel_attn_mlp_ratio = 2
    save_dir = "results/" + os.environ.get("ABA_SAVE", "phase1_int_ada_152m_6fam_pwb_archbias_adv")
