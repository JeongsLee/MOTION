"""arh5_byl — buoyancy branch + NATIVE Lagrangian scalar decode (user design 07-23).
Physics-split output channels: the TRANSPORTED scalar (slot 2: pdea/uncond smoke, incom particles) is
produced by the particle projection itself — ADA models x_p(t), y_p(t) per particle (basis trajectories,
ballistic warm-start with the FIXED axis convention) and the field output is the bilinear splat of carried
values at particle positions. Structural replacement, NOT a residual: all 7 residual arms tied with arh5
(fungible with existing capacity); a native output structure cannot be re-absorbed. Field scalars (SW h,
com density/pressure) stay Eulerian (smooth/elliptic: already strong). Buoyancy branch stays ON (base =
arh5_by; learned physics from the bank first). stride-1 + bilinear => tau=0 identity => hard-IC exact.
cfdbench (same desc as the trio, static mask slot) excluded by input-window temporal-variance detection.
  bash runners/_run_prose_stream_eu.sh motion_tf.train.configs.prose_150M_fluid5_6x_unified_task2_advb2_arh5_byl
"""
from .prose_150M_fluid5_6x_unified_task2_advb2_arh5_by import Cfg as _Base


class Cfg(_Base):
    ar_fair_loss = True   # ONE masked-mean over all AR segments: fixes ~2x uncond under-weighting
    lag_channel = True       # build the Lagrangian machinery
    lag_native = True        # slot-2 output = splat projection (not residual)
    lag_stride = 1           # 128^2 particles: bilinear splat is the identity at tau=0 (hard-IC exact)
    lag_modes = 6
    save_dir = "results/prose_150M_fluid5_6x_unified_task2_advb2_arh5_byl"
