"""Downstream NS-PwC/64 finetune from the 158M (Poseidon-B parity, dummy-supervised) pretrain — even
mid-training, since its NS is already FAR below the old 114M phase2 base (NS-Sines 27% vs 39.9%). Arch MUST
match the 158M checkpoint (geo_layers=14, router_depth=4, n_latent=24), so this extends the 158M pretrain
config and applies the SAME finetune protocol as the old 6.58% run (uv-only velocity task, eff-batch 40 via
grad-accum, 2000 epochs = 3200 steps, rel-L1 loss, half all2all). Head-to-head: old-114M-base 6.58% vs this
158M-base vs Poseidon-B 3.67%."""
import os
from .poseidon_pretrain_ivp_unified_158m import Cfg as _Base

_N = int(os.environ.get("FT_NSHOT", "64"))
_STEP = os.environ.get("FT_INIT_STEP", "145000")


_TASK = os.environ.get("FT_TASK", "NS-PwC")   # NS-PwC | CE-RM | ACE | Wave-Layer (downstream)


class Cfg(_Base):
    datasets = (_TASK,)
    n_shot = _N                          # downstream mode: train first N, reserve last test_per_fam
    test_per_fam = 128
    Nt = {"ACE": 20}.get(_TASK, 21)      # ACE trajectories are 20 frames; others 21
    max_ic_frac = float(os.environ.get("FT_IC_FRAC", "0.5"))  # 0=IC0-only (concentrate 64 traj on eval config), 0.5=half all2all
    det_ic = bool(int(os.environ.get("FT_DET_IC", "0")))      # 1=deterministic all2all IC cycling (uniform, paper-clean); 0=random
    uv_only = bool(int(os.environ.get("FT_UV_ONLY", "1")))  # 1=velocity task; 0=full-channel (uv+tracer for NS-PwC)
    pad_zero = False                     # uv-only masking (override the 158M pretrain pad_zero)
    rho_const = 0.0
    grad_accum = 10
    batch = 4                            # eff-batch = 4 x 10 = 40
    loss_rell1 = 4.0                     # relative-L1 (takes precedence over inherited relclamp); best loss
    loss_mse = False
    lr = float(os.environ.get("FT_LR", "5e-5"))   # FT_LR=5e-6 → one order lower (cleaner convergence from a good base)
    lr_decay = True
    warmup = 100
    steps = int(os.environ.get("FT_STEPS", "3200"))
    ckpt_every = int(os.environ.get("FT_CKPT_EVERY", "200"))   # raise for long all2all runs (limit disk)
    eval_every = int(os.environ.get("FT_EVAL_EVERY", "0")) or ckpt_every  # watch curve more often than ckpt
    init_from = os.environ.get("FT_INIT_PATH") or f"/eu/results/poseidon_pretrain_ivp_unified_158m_p0/ckpt_{_STEP}.npz"
    save_dir = f"results/ft158m_{_TASK}_N{_N}_uv_l1_s{_STEP}_ic{os.environ.get('FT_IC_FRAC','05').replace('.','')}{os.environ.get('FT_TAG','')}"
