"""CONTINUE the 158M Phase-2 IVP pretrain from ckpt_260000 — the last VALID checkpoint (steps 265k-320k were
LOST to a full /eu volume: the background ckpt sync wrote 0-byte files once cephfs hit 100%). The run itself
DID reach step 320000 (final eval 8.873%) but only ckpt_260000 (81%) survived on disk, so we resume from
there and re-run the remaining ~60k steps, now saving properly (volume freed to 78G).

WARM-RESTART (train_ivp.py:252): fresh save_dir (no in-place ckpts) + init_from → loads weights with a FRESH
step counter and a FRESH LR cosine over cfg.steps. REDUCED peak LR 3e-5 (vs original 1e-4): the original
CosineDecay(1e-4, 320000, alpha=0.05) was already ~1.3e-5 at step 260k, so 3e-5 is a gentle re-warm — enough
to keep learning without replaying the high-LR phase. Everything else (arch geo14/router4/nlat24, pad_zero,
rho_const=1.0, FULL all2all max_ic_frac=1.0, clamped rel-L2 loss) inherited from the 158M config. 2xH100."""
from .poseidon_pretrain_ivp_unified_158m import Cfg as _Base


class Cfg(_Base):
    init_from = "/eu/results/poseidon_pretrain_ivp_unified_158m_p0/ckpt_260000.npz"
    steps = 60000              # remaining budget (260k -> 320k); fresh cosine over these
    # NO re-warm: peak 5e-6 = the original cosine FLOOR (alpha 0.05 x 1e-4). Starting AT the floor (well
    # below the @260k tail ~1.3e-5) continues the decayed trajectory with minimal disruption — pure gentle
    # polish. (3e-5 re-warmed ABOVE the tail and BUMPED eval 9.3%->11.1%; original @275k was 9.309%.)
    lr = 5e-6
    warmup = 0                 # no warmup ramp -> no re-warm spike
    eval_every = 1000          # watch the recovery curve frequently
    ckpt_every = 2000          # save often-ish (~30 ckpts x 648MB ≈ 19G), bounded vs /eu headroom
    save_dir = "results/poseidon_pretrain_ivp_unified_158m_p0_cont260k_lr5e6"
