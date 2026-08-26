"""152M dsup variant: density fill = 0 instead of 1. Hypothesis (user): instance_norm centers EVERY family's
density at per-sample mean 0, so the rho=1 fill sits at +1sigma of com_ns's normalized density -> biases the
shared decoder upward -> the com_ns regression. Filling the incompressible density at 0 (= com_ns's center)
removes that offset, so com_ns should recover; the open question is whether the cross-family benefit on the
incompressible families (incom/pdearena/cfd) survives at 0 (it is still SUPERVISED-to-a-constant, c_mask=1,
which differs from masking's free/garbage output). Same scale (geo_layers=13, ~152M) + lup as the rho=1
flagship -> a clean rho=1 vs rho=0 head-to-head. 1 GPU."""
from .prose_150M_fluid5_6x_unified_dsup_150m import Cfg as _Base


class Cfg(_Base):
    rho_const = 0.0               # density slot -> 0 (matches instance_norm center -> no offset bias on com_ns)
    save_dir = "results/prose_150M_fluid5_6x_unified_dsup_155m_rho0"
