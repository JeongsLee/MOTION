"""Phase-1 TASK-2 (full-data convergence) for the UNIFIED single-expert architecture — the best
performance this arch can reach, trained on PROSE's full 6 families to compare head-to-head with
PROSE-FD / BCAT (each on its own 2×H100, exp_id=task2).

Changes vs Task-1 unified (prose_150M_fluid5_6x_unified):
 - datasets: 5 fluid families + pdearena_uncond (PROSE's incom_ns_arena_u). uncond is 14-frame, cached
   at t20 by clip-repeat padding; a PROSE-style TEMPORAL validity mask (SPEC valid_t=14, threaded through
   stream.py + per_sample_masked_rel_l2 t_mask) excludes the padded output frames from loss/eval — so
   t>=14 is left UNSUPERVISED exactly as PROSE does (data_mask_len).
 - steps: 160k = PROSE-FD's 40 epochs x 4000 steps (arXiv 2409.09811, max_epoch=40). warmup 4000.
 - test_max_per_fam: 100 (6 families x 100 ~ 600 eval samples).
"""
from .prose_150M_fluid5_6x_unified import Cfg as _Base


class Cfg(_Base):
    datasets = ("shallow_water", "com_ns", "incom_ns", "pdearena_ns", "cfdbench", "pdearena_uncond")
    n_channels = 6
    desc_dim = 6
    steps = 160000                 # PROSE-FD: 40 epochs x 4000 steps
    warmup = 4000
    test_max_per_fam = 100
    save_dir = "results/prose_150M_fluid5_6x_unified_task2"
    ckpt_every = 2000              # crash-safe: resume from latest ckpt_*.npz
