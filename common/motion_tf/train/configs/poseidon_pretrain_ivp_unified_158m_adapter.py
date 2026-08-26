"""C — re-pretrain a TRANSFER-FAVORABLE base on the SAME 6 fluid families (CE-RP/KH/CRP/Gauss + NS-Sines/Gauss,
= Poseidon-B's exact pretraining set) but with the generic learnable local-operator adapter ACTIVE from the
start. Hypothesis: a base whose representation is less fluid-hardcoded (the adapter carries general spatial
capacity, co-trained on fluids) transfers to OOD physics (ACE, Wave) better than the plain cont base — closing
the gap to Poseidon-B WITHOUT changing the pretraining data. This isolates ARCHITECTURE transferability.

Cold by default (fair vs Poseidon budget). PT_INIT (named cont base) → cheaper warm-continue variant.
Run: PT_STEPS=... PT_SAVE=... bash _run_ivp_eu.sh motion_tf.train.configs.poseidon_pretrain_ivp_unified_158m_adapter
"""
import os
from .poseidon_pretrain_ivp_unified_158m import Cfg as _Base


class Cfg(_Base):
    unified_adapter = True
    unified_adapter_hidden = int(os.environ.get("PT_AD_HIDDEN", "48"))
    steps = int(os.environ.get("PT_STEPS", "320000"))           # full = Poseidon budget; lower for a pilot
    # from-scratch by default: only define init_from when PT_INIT is set (else inherit base = cold; defining
    # init_from=None would be stringified to "None" by the runner and mis-staged).
    if os.environ.get("PT_INIT"):
        init_from = os.environ["PT_INIT"]                       # warm-continue (named ckpt → name-keyed load)
    save_dir = "results/" + os.environ.get("PT_SAVE", "pretrain_ivp_158m_adapter")
