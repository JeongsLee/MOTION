"""arh5_bgd — byl standard + desc-GROUP DECODE un-sharing (user hypothesis 07-23): even with the same
anti-derivative synthesis, splitting the final latent->physical readout per desc-group lets each group learn
its own way of USING the shared physics bank. Distinct from the bank-side splits that tied (slot-disjoint,
hsplit3): the vocabulary stays shared, only the grammar (readout weights) is un-shared -> structural
conflict removal, not absorbable zero-init capacity. Groups: g0=incompressible/advective trio(+cfd),
g1=compressible(com: acoustic/shock momentum separated from incompressible momentum readout), g2=SW.
+~2.1k params. Base = arh5_byl (STANDARD: native Lag scalar decode + buoyancy + ar_fair_loss).
  bash runners/_run_prose_stream_eu.sh motion_tf.train.configs.prose_150M_fluid5_6x_unified_task2_advb2_arh5_bgd
"""
from .prose_150M_fluid5_6x_unified_task2_advb2_arh5_byl import Cfg as _Base


class Cfg(_Base):
    group_decode = True      # per-desc-group readout of the shared bank
    save_dir = "results/prose_150M_fluid5_6x_unified_task2_advb2_arh5_bgd"
