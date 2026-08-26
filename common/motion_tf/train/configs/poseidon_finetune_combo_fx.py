"""Finetune the combo learned-head pretrain WITH an added FIXED exact-physics-operator branch
(unified_fixedops): Δ, −(u·∇), ∇·, R(u)=u−u³, c²Δ — zero-init coefficients, so it warm-starts identically
from the learned-head pretrain and finetune turns on the task-appropriate PDE terms. Same FT_TASK-parametrized
N-shot protocol as poseidon_finetune_combo; init_from must be a NAMED ckpt (positional 305k migrated via
motion_tf.utils._migrate_ckpt) so the new fixedops vars fresh-init instead of corrupting a positional load.
"""
from .poseidon_finetune_combo import Cfg as _Base


class Cfg(_Base):
    unified_fixedops = True
