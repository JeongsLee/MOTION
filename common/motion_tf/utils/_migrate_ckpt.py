"""One-time POSITIONAL -> NAMED ckpt migration so an architecturally-extended model (e.g. + reaction
branch) can load the cont base BY NAME (new vars fresh-init-skipped instead of corrupting via index shift).

Build the SAME-arch model as the one that saved the positional ckpt (NO reaction branch), positional-load,
then name-save.
Run: python -u -m motion_tf.utils._migrate_ckpt <cfg_module> <pos_ckpt.npz> <out_named.npz>
e.g. python -u -m motion_tf.utils._migrate_ckpt motion_tf.train.configs.poseidon_pretrain_ivp_unified_158m_cont \
       /eu/results/poseidon_pretrain_ivp_unified_158m_p0_cont260k_lr5e6/ckpt_60000.npz \
       /eu/results/migrated/cont260k_lr5e6_ckpt60000_named.npz
"""
from __future__ import annotations
import importlib, os, sys
import tensorflow as tf
from ..model import PhysicsOperatorMixture
from ..utils import ckpt_named


def main(cfg_path, pos_path, out_path):
    cfg = importlib.import_module(cfg_path).Cfg
    if bool(getattr(cfg, "bf16", False)):
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
    # CRITICAL: build with the SAME arch that saved the positional ckpt, but WITHOUT the newly-added branch.
    # The combo pretrain HAS reaction/wave/adapter/buoyancy/gradx ON (part of the saved arch → must stay ON);
    # only the finetune-added branch (fixedops) must be OFF so the migrated var set matches the ckpt.
    for fl in ("unified_fixedops",):
        assert not bool(getattr(cfg, fl, False)), f"migration cfg must have {fl} OFF (match saved ckpt arch)"
    C = int(getattr(cfg, "n_channels", 0))
    if not C:
        try:
            from ..data.prose import C_MAX as C            # prose/Phase-1 channel count (default 6)
        except Exception:
            C = 8
    Nx = int(getattr(cfg, "Nx", 128)); Tin = int(getattr(cfg, "T_in", 1)); D = int(getattr(cfg, "desc_dim", 3))
    u0 = tf.zeros((1, Tin, Nx, Nx, C)) if Tin > 1 else tf.zeros((1, Nx, Nx, C))   # multi-frame (prose) vs IVP
    model = PhysicsOperatorMixture(cfg)
    _ = model.call_with_gate(u0, tf.zeros((1, D)), tf.zeros((1, 4)))
    nvar = len(model.trainable_variables)
    print(f"[migrate] built {cfg_path}: {nvar} trainable vars; loading {pos_path}", flush=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ckpt_named.migrate(model, pos_path, out_path)
    print(f"[migrate] DONE -> {out_path}  ({nvar} named vars)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
