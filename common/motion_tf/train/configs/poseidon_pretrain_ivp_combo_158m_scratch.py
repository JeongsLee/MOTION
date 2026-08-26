"""CAPACITY-SCALED combo pretrain (~162M ≈ Poseidon-B 157.7M): the 113.5M combo floored at ~16% in-distribution
on the 6 fluid families; every non-capacity lever (LR, batch, recursion-was-too-slow, fixed-ops-were-redundant)
left it there, so this tests whether Poseidon-level CAPACITY closes the gap while keeping the OOD-reaction win.
Combo arch (learned heads, unified_ops=False, adaptive_mix, warp, all-IC ic_reuse) UNCHANGED — only depth/width
raised (geo_layers 9→14, router_depth 2→4, n_latent 16→24), matching the old 158M explicit base's capacity knobs.
Fresh scratch (n_latent width change → cannot warm-start from the 113M ckpt). 2x H100, low batch (162M: batch12
would OOM ~83GB; batch8 → eff-batch 16 ~55GB safe). steps is a ceiling — watch the eval vs the 113M run and
early-stop once the capacity effect (below/at the ~16% floor) is clear.
"""
from .poseidon_pretrain_ivp_combo_scratch import Cfg as _Base


class Cfg(_Base):
    geo_layers = 14                # depth ↑  (Poseidon-parity capacity)
    router_depth = 4               # transformer-encoder depth ↑
    n_latent = 24                  # most-compressed hidden width ↑  -> ~162M
    batch = 8                      # eff-batch 16 on 2 GPU (162M; batch12 OOMs)
    steps = 800000                 # ceiling; early-stop when the 113M-vs-162M signal is clear
    warmup = 2000
    ckpt_every = 10000
    eval_every = 5000
    save_dir = "results/poseidon_pretrain_ivp_combo_158m_scratch"
