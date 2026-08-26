"""152M dsup (rho=1) + REGIME FiLM: condition h_geom on the field regime (_regime_features) so the shared
decoder can predict VARYING density for compressible (com_ns) vs CONSTANT for incompressible — the
architectural fix for the dummy-supervision com_ns trade-off (fill-value & shock could NOT fix it; it is a
'shared decoder collapses to the ρ=const majority' problem). regime_film works WITHOUT a router (unified).
Same scale (geo_layers=13) + lup + rho=1 as the flagship -> a clean test: does com_ns recover (toward
masking's 1.17) while incom/pdearena gains hold? 1 GPU."""
from .prose_150M_fluid5_6x_unified_dsup_150m import Cfg as _Base


class Cfg(_Base):
    regime_film = True            # FiLM(γ,β) on h_geom from field regime -> per-regime decoder split
    save_dir = "results/prose_150M_fluid5_6x_unified_dsup_155m_film"
