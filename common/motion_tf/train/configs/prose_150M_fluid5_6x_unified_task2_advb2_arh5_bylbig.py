"""arh5_bylbig — byl STANDARD + the capacity/feedback endgame + lf2 (user: "lup=2").
= byl + arh5_big config (5-seg AR, N_p8/modes8, geo_layers 13 -> 152M ~ BCAT 155.6M) + latent_factor=2
(64^2 latent, 2x upsample instead of 4x: attacks the velocity-amplitude smoothing of coarse-latent
synthesis - the remaining pdea velocity-phase channel). Judged vs byl and byl5s (bylbig - byl5s ~
capacity+lf2).
  bash runners/_run_prose_stream_eu.sh motion_tf.train.configs.prose_150M_fluid5_6x_unified_task2_advb2_arh5_bylbig
"""
from .prose_150M_fluid5_6x_unified_task2_advb2_arh5_byl import Cfg as _Base


class Cfg(_Base):
    Nt = 3
    T_final = 0.2
    ar_seg = 5
    N_p = 8
    n_modes = 8
    geo_layers = 13
    latent_factor = 2
    save_dir = "results/prose_150M_fluid5_6x_unified_task2_advb2_arh5_bylbig"
