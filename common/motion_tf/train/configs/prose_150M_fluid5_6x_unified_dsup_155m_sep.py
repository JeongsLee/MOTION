"""152M dsup + SEPARATED DENSITY: give com_ns its OWN density slot. com_ns real ρ stays in slot 3; the
incompressible families' const-ρ marker goes to a DEDICATED slot 6 (needs PROSE_CMAX=7, n_channels=7). The
slot-3 decoder is then trained ONLY on com_ns's varying density (no '4 const vs 1 varying' majority collapse)
-> com_ns should recover toward masking's 1.17, while incompressible families keep the clean const-ρ
auxiliary (in slot 6). Direct structural fix for the com_ns trade-off (fill-value/shock/FiLM-so-far did not
fix it). Same geo_layers=13 + lup as the flagship. RUN WITH env PROSE_CMAX=7. 1 GPU."""
from .prose_150M_fluid5_6x_unified_dsup_150m import Cfg as _Base


class Cfg(_Base):
    n_channels = 7                # +1 slot (6) for the separated incompressible density marker
    incomp_rho_slot = 6           # incompressible const-rho -> slot 6 (com_ns real rho keeps slot 3 alone)
    dummy_slots = []              # use the SEPARATION branch, not the shared-slot dummy
    rho_const = 1.0
    save_dir = "results/prose_150M_fluid5_6x_unified_dsup_155m_sep"
