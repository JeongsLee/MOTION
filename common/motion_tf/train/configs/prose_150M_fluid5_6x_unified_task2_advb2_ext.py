"""CONVERGENCE EXTENSION of the advb2 arm toward literature budgets (BCAT 128x4000x40 = 20.5M views,
PROSE-FD 176x4000x40 = 28.2M): +625k steps x 8 views/step = +5M views on top of advb2's 1.28M.

Protocol notes:
 - Warm-start from the FINISHED advb2 run's ckpt_160000 (weights only; optimizer state is not
   checkpointed). LR CONTINUES at the level the 160k run ended at (5e-6 = 1e-4 x alpha 0.05), with a
   2k warmup to rebuild Adam moments gently and lr_alpha=0.5 so the cosine stays in [2.5e-6, 5e-6] --
   no re-heat, no loss shock. The paper's matched-budget (<=1M views) curves are NOT touched by this
   run -- it lives in its own folder; the advb2 folder keeps all ckpts.
 - Disk: ckpt_keep=3 -> only the newest 3 checkpoints are retained (locally AND remotely via the
   _keep runner), so the 62-ckpt / ~40GB accumulation of a naive 625k run cannot recur.
 - Inline [EVAL] every 10k steps on the same test-600 -> the convergence curve extends the paper's
   Fig. data-efficiency axis directly; rescore T=4 (papers' protocol) at ~2M/4M/6M views milestones.
Launch:  bash _run_prose_stream_eu_keep.sh motion_tf.train.configs.prose_150M_fluid5_6x_unified_task2_advb2_ext
"""
from .prose_150M_fluid5_6x_unified_task2_advb2 import Cfg as _Base


class Cfg(_Base):
    steps = 625000                 # +5M views @ 8 views/step
    warmup = 2000
    lr = 5e-6                      # = the LR the 160k run ENDED at (1e-4 x alpha 0.05): no re-heat shock;
    lr_alpha = 0.5                 # cosine floor 2.5e-6 -> whole run stays in the ending-LR band, so the
                                   # optimizer never jumps above where the finished run left off
    init_ckpt = "/eu/results/prose_150M_fluid5_6x_unified_task2_advb2/ckpt_160000.npz"
    save_dir = "results/prose_150M_fluid5_6x_unified_task2_advb2_ext"
    ckpt_every = 10000
    eval_every = 10000
    ckpt_keep = 3
