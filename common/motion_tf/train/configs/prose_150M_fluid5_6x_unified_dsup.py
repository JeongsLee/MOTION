"""Phase-1 Task-1 ABLATION: DUMMY-SUPERVISION (CROSS-FAMILY PHYSICS slots only) vs MASKING.

Identical to prose_150M_fluid5_6x_unified in EVERY way (arch, datasets, steps, loss, norm, batch, lr) except
how a family's MISSING physics channels are handled. Instead of MASKING them out (c_mask=0 -> garbage in the
shared backbone), the cross-family physics slots are SUPERVISED to a CONSISTENT physical constant when a
family lacks them, so every fluid family shares the same target structure in rho/p:
  - density slot (3): rho_const=1.0 (incompressible rho=const, NON-ZERO/ALIVE)
  - pressure slot (4): p_const=0.0  (Poseidon incompressible convention)
ONLY slots 3,4 (NOT every unused slot — the earlier blanket "others=0" wrongly zeroed FOREIGN slots like
scalar-h on NS and SW's real velocity = many dead channels -> divergence). Families that HAVE rho/p (com_ns)
are untouched; all other unused slots stay MASKED. TEST/eval c_mask stays ORIGINAL (real channels) so the
metric is comparable to the masking baseline (phase1task1/raw_logs/ours_eval.txt, 1-GPU). RUN ON 1 GPU to
match that baseline's trajectories-per-step (avoid the 2-GPU 2x-data confound)."""
from .prose_150M_fluid5_6x_unified import Cfg as _Base


class Cfg(_Base):
    dummy_slots = [3]             # DENSITY ONLY: families lacking density are incompressible -> rho=const is
                                  # PHYSICALLY TRUE. Pressure is LEFT MASKED (not supervised): incompressible
                                  # pressure is a nonzero Lagrange-multiplier field, so p=0 is a PHYSICALLY
                                  # FALSE target that destabilized a physics-informed model (cfdbench rose
                                  # 3.4->5.2 under p=0). No false constraint -> only the true one (rho=const).
    rho_const = 1.0               # density slot -> 1.0 (ALIVE) for families lacking density (incompressible)
    save_dir = "results/prose_150M_fluid5_6x_unified_dsup_rho1"
