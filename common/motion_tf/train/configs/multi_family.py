"""Multi-family config: Heat · Advection · ReacDiff · ADR on a common grid.
Headline goal: the family gate produces a per-family operator fingerprint (DESIGN.md §7)."""


class Cfg:
    Nx = 32
    Nt = 16
    T_final = 1.0
    N_p = 16
    n_modes = 8
    max_order = 8
    d_geom = 24
    experts = ("convective", "diffusive", "reaction")
    families = ("heat", "advection", "reacdiff", "adr")
    n_per_train = 96          # per family
    n_per_val = 32
    seed = 0
    steps = 1200
    batch = 32
    lr = 1e-3
    lambda_gate = 1e-3
    jit_compile = True
    save_dir = "results/multi_family"
