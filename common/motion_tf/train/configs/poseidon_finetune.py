"""DOWNSTREAM TRANSFER finetune (Poseidon protocol head-to-head).

ONE env-driven config serves every (task, N-shot) pair so we don't write 8 files:
  FT_TASK   in {NS-PwC, CE-RM, ACE, Wave-Layer}   (default NS-PwC)
  FT_NSHOT  number of finetune trajectories         (default 128)
Warm-starts our finished phase-2 pretrained weights (ckpt_320000, on /eu) and finetunes on the FIRST
FT_NSHOT trajectories of one downstream task, reserving the last `test_per_fam` as a fixed test set.
Eval = Poseidon metric (MEDIAN relative-L1 at final time, physical). Compare to Poseidon-B finetuned
identically (scOT) at the same N on the same data/split. Run via _run_ivp_eu.sh (capella, POSE_ASSEMBLED=/eu)."""
import os
from .poseidon_pretrain_ivp_unified import Cfg as _Base

_TASK = os.environ.get("FT_TASK", "NS-PwC")
_N = int(os.environ.get("FT_NSHOT", "128"))
_NT = {"ACE": 20}.get(_TASK, 21)        # ACE trajectories are 20 frames; others 21


class Cfg(_Base):
    datasets = (_TASK,)
    n_shot = _N                          # train on first N trajectories (downstream-mode loader)
    Nt = _NT
    test_per_fam = 128                   # fixed test set = last 128 trajectories (median rel-L1@final)
    init_from = "/eu/results/poseidon_pretrain_ivp_unified_eu2/ckpt_320000.npz"  # finished phase-2 weights
    decoder_only = bool(int(os.environ.get("FT_DECODER_ONLY", "0")))  # freeze backbone, train decoder only
    lr_decoder = float(os.environ.get("FT_LR_DEC", "0"))  # >0 = Poseidon-style discriminative LR (decoder/head high)
    uv_only = bool(int(os.environ.get("FT_UV_ONLY", "0")))  # supervise velocity only (match Poseidon's per-task focus)
    grad_accum = int(os.environ.get("FT_GRAD_ACCUM", "1"))  # micro-batches per update; effective batch = batch*grad_accum
    max_ic_frac = float(os.environ.get("FT_MAX_IC", "0.5"))  # 1.0 = full all2all (IC i in [0,Nt-1])
    pad_zero = bool(int(os.environ.get("FT_PAD0", "0")))  # 0-in-0-out: supervise unused slots->0 (AR-stable output)
    pad_const = float(os.environ.get("FT_PAD_CONST", "0"))  # const-in-const-out: unused slots->const (channel ALIVE, not killed)
    loss_rell1 = float(os.environ.get("FT_LOSS_RELL1", "0"))  # >0 = relative-L1 loss (Poseidon family + eval-aligned)
    loss_perch = bool(int(os.environ.get("FT_PERCH", "0")))  # 2-term loss: uv per-channel rel-L1 + dummy abs-L1 (uv undiluted)
    dummy_w = float(os.environ.get("FT_DUMMY_W", "0.1"))  # weight on the dummy abs-L1 term (perch only)
    loss_mse = bool(int(os.environ.get("FT_LOSS_MSE", "0")))  # 1 = masked global-norm MSE (= phase2-pretrain loss)
    lr = 5e-5                            # finetune LR (below pretrain 1e-4); cosine + short warmup
    lr_decay = True
    warmup = 100
    # MATCHED to Poseidon: same batch (FT_BATCH) AND same total optimizer steps (FT_STEPS).
    batch = int(os.environ.get("FT_BATCH", "4"))
    steps = int(os.environ.get("FT_STEPS", str(max(200, 200 * _N // batch))))  # default = 200 epochs
    ckpt_every = max(50, steps // 16)    # ~16 evals -> take best
    _tag = ('_dec' if decoder_only else ('_disc' if lr_decoder > 0 else '')) + ('_uv' if uv_only else '') + ('_l1' if loss_rell1 > 0 else ('_mse' if loss_mse else '')) + ('_perch' if loss_perch else '') + ('_a2a' if max_ic_frac > 0.5 else '') + ('_p0' if pad_zero else '') + (f'_pc{pad_const:g}' if pad_const else '')
    save_dir = f"results/ft_{_TASK}_N{_N}_b{batch}s{steps}{_tag}"   # distinct dir (no mixed ckpts)
