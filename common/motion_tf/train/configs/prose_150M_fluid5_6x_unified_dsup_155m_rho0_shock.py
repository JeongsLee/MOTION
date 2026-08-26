"""152M dsup + SHOCK feature ON. The UnifiedExpert merges all 6 physics operators into one bank; with
unified_shock=True it adds a shock-capturing feature (was OFF by default in every prior dsup/unified run).
Motivation: com_ns (compressible, shocks) is the lone family the dummy-supervision REGRESSED, and the
known remaining gap is shock/Riemann velocity (CE-RP/CRP) — shock handling was simply never enabled.
Combine with rho=0 (rho=0 ~= rho=1 for com_ns, so the density-fill value isn't the lever; shock is).
Same scale (geo_layers=13, ~152M+) + lup. 1 GPU."""
from .prose_150M_fluid5_6x_unified_dsup_155m_rho0 import Cfg as _Base


class Cfg(_Base):
    unified_shock = True          # +shock-capturing feature in the UnifiedExpert bank (was default False)
    save_dir = "results/prose_150M_fluid5_6x_unified_dsup_155m_rho0_shock"
