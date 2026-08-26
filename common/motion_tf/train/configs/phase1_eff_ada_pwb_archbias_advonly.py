"""Phase-1 ADA — archbias + advbias ONLY (no panel_attn). Isolates #3 (learned upwind advection-bias head) on
top of the session-best archbias, WITHOUT the expensive 3-layer panel_attn (#4). Keeps the ~1.1 s/step
no-recursion speed (advbias is ~free). A/B vs archbias (does the upwind advection-bias head alone improve
pdea?) and vs archbias_adv (is panel_attn worth its 2× cost?).
Launch (2 GPU): bash _run_prose_stream_eu.sh motion_tf.train.configs.phase1_eff_ada_pwb_archbias_advonly
"""
import os
from .phase1_eff_ada_pwb_archbias import Cfg as _Base


class Cfg(_Base):
    unified_advbias = True       # #3 learned upwind advection-bias head only (no panel_attn)
    save_dir = "results/" + os.environ.get("ABAO_SAVE", "phase1_int_ada_152m_6fam_pwb_archbias_advonly")
