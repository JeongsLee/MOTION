"""Phase-1 ADA — STRENGTHENED panel-axis attention (on cfdgeom). The first panel_attn was weak (1 layer,
attn width = oc = 16, NO FFN) and washed out to a tie with cfdgeom by step 5000. This version makes the
panel-axis coupler a proper L-layer transformer: wider attention dim, multi-head, + FFN per block. Goal:
give the parallel inter-panel coupling enough capacity to actually recover the recursion's PDEArena gain
(at pw_batched speed). Final output proj stays zero-init → starts == cfdgeom, learns the coupling. Stacks on
cfdgeom so the A/B vs cfdgeom isolates the (stronger) attention contribution.
Launch (2 GPU): bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwb_panelattn_strong
"""
import os
from .phase1_eff_ada_pwbatched_cfdgeom import Cfg as _Base


class Cfg(_Base):
    panel_attn = True
    panel_attn_dim = 64        # attention width (was 16=oc) → much more capacity
    panel_attn_heads = 4
    panel_attn_layers = 3      # stacked transformer blocks (was 1)
    panel_attn_mlp_ratio = 2   # + FFN per block (was none)
    save_dir = "results/" + os.environ.get("PAS_SAVE", "phase1_int_ada_152m_6fam_pwb_panelattn_strong")
