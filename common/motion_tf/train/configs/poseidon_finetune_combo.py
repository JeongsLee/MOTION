"""Phase-2 downstream finetune — COMBO base (learned-head, single-shot) from poseidon_pretrain_ivp_combo.
= the COMBO arch (so the combo-IVP ckpt loads positionally — finetune arch == pretrain arch, no new heads,
no name-migration) + the SAME N-shot finetune protocol as poseidon_finetune_158m (FT_TASK-parametrized,
first N train / last 128 test, rel-L1 loss, eff-batch 40 via grad-accum). Tests the Phase-2 hypothesis on
OOD physics (ACE/Wave) and in-family NS, head-to-head vs the old explicit-operator base + Poseidon-B.

Launch (2 GPU), e.g. ACE:
  FT_TASK=ACE FT_NSHOT=64 FT_UV_ONLY=0 FT_IC_FRAC=0 FT_TAG=_combo \
  FT_INIT_PATH=/eu/results/poseidon_pretrain_ivp_combo/ckpt_120000.npz \
  bash /code-vol/code/runners/_run_ivp_eu.sh motion_tf.train.configs.poseidon_finetune_combo
"""
import os
from .poseidon_pretrain_ivp_combo import Cfg as _Base

_N = int(os.environ.get("FT_NSHOT", "64"))
_TASK = os.environ.get("FT_TASK", "ACE")   # NS-PwC | CE-RM | ACE | Wave-Layer


class Cfg(_Base):
    datasets = (_TASK,)
    n_shot = _N                          # downstream: train first N, reserve last test_per_fam
    test_per_fam = 128
    Nt = {"ACE": 20, "Poisson-Gauss": 2}.get(_TASK, 21)   # ACE=20 frames; steady elliptic=2 (IC/source -> solution)
    max_ic_frac = float(os.environ.get("FT_IC_FRAC", "0.0"))  # 0 = IC0-only (concentrate 64 traj on eval config)
    det_ic = bool(int(os.environ.get("FT_DET_IC", "0")))
    uv_only = bool(int(os.environ.get("FT_UV_ONLY", "0")))    # ACE: 0 = full-channel (c_mask → ACE solution slot)
    pad_zero = False
    rho_const = 0.0
    grad_accum = 10
    batch = 4                            # eff-batch = 40
    loss_rell1 = float(os.environ.get("FT_LOSS_CAP", "4.0"))   # rel-L1 clamp cap; raise for extreme-OOD tasks
    # (Poisson steady: initial error ~37x > cap=4 -> clamped flat -> ZERO gradient; FT_LOSS_CAP=100 unblocks)
    loss_mse = False
    lr = float(os.environ.get("FT_LR", "5e-5"))
    lr_decay = True
    warmup = 100
    steps = int(os.environ.get("FT_STEPS", "3200"))
    ckpt_every = int(os.environ.get("FT_CKPT_EVERY", "200"))
    eval_every = int(os.environ.get("FT_EVAL_EVERY", "0")) or ckpt_every
    init_from = os.environ.get("FT_INIT_PATH") or "/eu/results/poseidon_pretrain_ivp_combo/ckpt_120000.npz"
    save_dir = f"results/ftcombo_{_TASK}_N{_N}{os.environ.get('FT_TAG','')}"
