"""Phase-1 Task-1 MASKING baseline, paired control for the dummy-supervision ablation. IDENTICAL to
prose_150M_fluid5_6x_unified (masking: unused slots c_mask=0) — only the save_dir differs so it doesn't
clobber the original result. Run side-by-side with prose_150M_fluid5_6x_unified_dsup (same code, same
hardware, same seed) so masking vs dummy-supervision is a clean A/B with no code/version confound."""
from .prose_150M_fluid5_6x_unified import Cfg as _Base


class Cfg(_Base):
    save_dir = "results/prose_150M_fluid5_6x_unified_maskbase"
