"""PHASE-2 IVP COLD pretrain SCALED to Poseidon-B parity (~162M vs Poseidon-B 158M), capacity added via
DEPTH (geo_layers 9->14, router_depth 2->4) + the most-compressed-hidden width (n_latent 16->24) — the
levers that helped. loss = task2 clamped rel-L2 (loss_relclamp=4.0, loss_mse=False, inherited+honored by the
patched train_ivp). 2xH100 (gb8). Goal: same-capacity, same-loss head-to-head vs Poseidon-B on downstream
transfer (NS-PwC velocity etc.). Cold (no init_from). Run: _run_ivp_eu.sh."""
from .poseidon_pretrain_ivp_unified import Cfg as _Base


class Cfg(_Base):
    geo_layers = 14            # depth ↑
    router_depth = 4           # transformer-encoder depth ↑
    n_latent = 24              # most-compressed hidden width ↑  -> ~162M params (Poseidon-B = 158M)
    pad_zero = True            # supervise unused slots -> 0 (Poseidon-style dummy supervision; not masking)
    rho_const = 1.0            # incompressible (NS) density slot -> const 1.0 (const-in-const-out, like Poseidon rho)
    max_ic_frac = 1.0          # FULL all2all: IC frame i ∈ [0,20] (late starts -> few supervised frames,
                               # masked out; matches Poseidon's full-pair all2all, more diverse)
    steps = 320000             # match Poseidon-B pretraining budget (~39 epochs, gb8)
    warmup = 1000
    seed = 5
    ckpt_every = 5000
    save_dir = "results/poseidon_pretrain_ivp_unified_158m_p0"   # fresh dir (full-all2all cold start, no resume from 0.5-ckpt)
    # loss = clamped rel-L2 (task2) inherited; cold (no init_from)
