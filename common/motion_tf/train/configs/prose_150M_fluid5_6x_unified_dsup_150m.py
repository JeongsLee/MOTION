"""Phase-1 Task-1 SCALED to ~150M (BCAT/PROSE param class) + DUMMY-SUPERVISION (rho=const only).
Inherits prose_150M_fluid5_6x_unified_dsup (learned-upsample, dummy_slots=[3] rho_const=1.0, pressure/others
MASKED) and scales the encoder depth (geo_layers 9->12) to ~150M. Goal: the density-only dummy-supervision
recipe that kept WORKING at 114M (overall+incom+pdearena+cfd all below the masking baseline, cfd stable),
now at the BCAT(156M)/PROSE-FD(165M) param scale. Same lup + current code so a paired masking-150M run is a
clean A/B (no upsample/code confound, unlike the Jun-13 bilinear ours_eval.txt)."""
from .prose_150M_fluid5_6x_unified_dsup import Cfg as _Base


class Cfg(_Base):
    geo_layers = 13               # 9 -> 13 encoder depth: ~114M -> ~155M (BCAT/PROSE class; verify via log)
    save_dir = "results/prose_150M_fluid5_6x_unified_dsup_155m"
