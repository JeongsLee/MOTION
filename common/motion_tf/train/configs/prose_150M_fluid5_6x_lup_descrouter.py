"""Phase-1 Task-1: attn-router + DESCRIPTOR (coexist) + lup + masking. The data-only attn-router COLLAPSED
(no family signal -> diffusive-only); instead of reverting to a descriptor-only FamilyGate, KEEP the
field-based attn-router AND feed it the equation descriptor (router_descriptor=True -> broadcast+concat to the
router input). Family prior prevents collapse; spatial/regime field routing retained. 6-expert mixture,
learned-upsample, instance-norm masking. Compare to the 114M unified-lup masking best (class-avg 6.31)."""
from .prose_150M_fluid5_6x_lup import Cfg as _Base


class Cfg(_Base):
    router_descriptor = True      # feed equation descriptor INTO the attn-router (prevent collapse, keep field routing)
    # attn_router=True, field_gate=False, experts=6, learned_upsample=True, instance_norm all inherited
    save_dir = "results/prose_150M_fluid5_6x_lup_descrouter_logit"
