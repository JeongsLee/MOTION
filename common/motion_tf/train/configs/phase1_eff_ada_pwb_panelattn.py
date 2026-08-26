"""Phase-1 ADA — APPROACH ②: pw_batched + PANEL-AXIS ATTENTION. The N_p panels are computed independently
(batched, fast), then a SINGLE bidirectional self-attention over the panel axis (per pixel) couples them so
each panel's W is conditioned on all others — the PARALLEL replacement for the sequential W-recursion (no
teacher-forcing needed since W is latent → bidirectional, not causal). Captures inter-panel self-consistency
(the benefit recursion gave PDEArena) while staying batchable → keeps the ~1 s/step pw_batched speed. Panel
attention is zero-init → starts identical to pw_batched, learns coupling. A/B vs pw_batched (no coupling) and
vs wrec (sequential W-recursion) — does parallel attention recover the recursion's PDEArena gain at speed?
Launch (2 GPU): bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwb_panelattn
"""
import os
from .phase1_eff_ada import Cfg as _Base


class Cfg(_Base):
    parallel_w = True
    pw_batched = True         # batched panel rollout (the _pwb path panel_attn hooks into)
    panel_attn = True         # bidirectional self-attention over the N_p panels (parallel coupling)
    panel_attn_heads = 4
    step_embed = True
    w_feedback = False        # panel_attn REPLACES the sequential recursion (must stay off for the _pwb path)
    swap_memory = False
    save_dir = "results/" + os.environ.get("PA_SAVE", "phase1_int_ada_152m_6fam_pwb_panelattn")
