"""Single source of truth for every corpus family (foundationv2).

One FamilySpec per dataset fixes: geometry (K), raw format+location, the channel
map into the UNIVERSAL slot layout, the symbolic governing equation + parameter
extraction rule, axis roles, boundary-condition type per axis, and the split
source. Preprocessing adapters read ONLY this — no per-family logic elsewhere.

Universal slot layout (physical-quantity channels; a family fills a subset):
  0 vx   1 vy   2 vz   3 density   4 pressure   5 energy/temperature
  6 scalarA (smoke / particles / activator / concentration)
  7 scalarB (inhibitor)   8 height (shallow water)
Unused slots are dummy-filled per training convention (not this file's concern).

Axis roles (dimension-agnostic operators key off role, not index):
  0 = x (streamwise/horizontal)  1 = y (horizontal)  2 = z (GRAVITY/vertical)
BC per spatial axis: "periodic" | "wall" (Dirichlet/Neumann via projector) | "open".
"""
from __future__ import annotations

from dataclasses import dataclass

# 9 -> 12: SEMANTIC SLOT SPLIT. Slots are the model's universal output channels, and two of them
# had become grab-bags of physically unrelated quantities — slot 4 carried thermodynamic pressure
# AND an elliptic potential AND a Mach number AND von Mises stress AND car-surface Cp, slot 6
# carried advected/reactive scalars AND wave displacement. All five of our worst families sat on
# those two slots (drivaernet 30.6, Wave-Layer 26.8, airfrans 23.6, geofno_airfoil 17.8,
# shapenet_car 14.1) while raw contention COUNT predicted nothing (Pearson +0.07 against
# headroom), so what hurts is incompatible physics on one channel, not many families on one
# channel. The split also resolves the K=3 collision: those five were the only users of the 3D
# axial weights and every one of them wrote slot 4.
#   9  = car-surface static pressure   (airfrans, shapenet_car, drivaernet_pressure)
#   10 = wave displacement             (Wave-Layer)
#   11 = steady non-pressure scalar    (geofno_airfoil Mach, geofno_elasticity stress)
# Slot 8 was deliberately NOT reused: shallow_water is the one family on a private slot and is
# already at its representation floor, so it has nothing to gain and something to lose.
#   9  = car-surface static pressure   (airfrans, shapenet_car, drivaernet_pressure)
#   10 = wave displacement             (Wave-Layer)
#   11 = steady non-pressure scalar    (geofno_airfoil Mach, geofno_elasticity stress)
#   12..15 = MEMORY. Never scored, never owned by a family, always fed back.
# The memory channels exist because making the semantic slots independent removes something the
# model was actually using. Each AR segment re-anchors on the hard IC and rebuilds the latent box
# from scratch, so the ONLY state that survives from one segment to the next is the window — and
# the window's unused channels were carrying it. Cutting them costs nothing for local dynamics
# (ACE +0.0, shallow_water +0.2, drivaernet +0.0) and wrecks precisely the globally-coupled ones,
# which need to iterate information across the whole domain: Poisson-Gauss 4.6 -> 35.0,
# Wave-Layer 26.8 -> 39.3, geofno_elasticity 7.4 -> 15.0. Dedicated channels give that iterate a
# home that no other family can collide with.
SEMANTIC_SLOTS = 12
NUM_SLOTS = 16
GRAVITY_ROLE = 2


@dataclass(frozen=True)
class FamilySpec:
    name: str
    K: int                                  # spatial dimension (2 or 3)
    fmt: str                                # adapter key: grid_npy | grid_nchw | pdebench3d | mesh_*
    location: str                           # path under /corpus
    channels: tuple                         # raw-channel-index -> universal slot, in raw order
    symbolic: str                           # governing equation, tokenizer template
    operators: tuple                        # operator tokens present (for symbolic encoder)
    param_names: tuple = ()                 # scalar equation parameters (log-Fourier embedded)
    param_source: str = "none"              # "filename" | "attr:<key>" | "const" | "per_sample" | "none"
    param_const: tuple = ()                 # values when param_source == "const"
    roles: tuple = (0, 1, 2)                # axis -> role (first K used)
    bc: tuple = ("periodic", "periodic", "periodic")   # per spatial axis
    time: str = "transient"                 # "transient" | "steady" (relaxation-trajectory)
    unit: str = "trajectory"                # split unit: trajectory | design | case
    split_source: str = "derived"           # "official:<ref>" | "derived"
    stage: str = "pretrain"                 # "pretrain" | "downstream"
    t_valid: int | None = None              # valid frames (pdearena_uncond padding)
    geom_kind: str = "none"                 # "sdf" (signed-distance -> BL mask + near-wall colloc) |
                                            # "mask" (material/boundary indicator, passed as-is) | "none"
    nc_keys: tuple = ()                     # (field_key, source_key) for elliptic BVPs: predict the
                                            # field from the source term fed as a conditioning channel
    coef_file: tuple = ()                   # (path_under_corpus, key) for a SEPARATE per-sample
                                            # COEFFICIENT FIELD, sample-index aligned with the
                                            # solution. Without it the IVP is not closed: the same
                                            # IC under a different coefficient field has a different
                                            # solution, so the error has an irreducible floor.
    notes: str = ""


# --------------------------------------------------------------------------- #
# PROSE/BCAT 6-family grid corpus (native 128², frames-as-time)
# --------------------------------------------------------------------------- #
_PROSE = [
    FamilySpec("shallow_water", 2, "grid_npy", "cache/prose128/shallow_water_n100000_t20.npy",
               channels=(8,), symbolic="dt(h) = -div(h*u); shallow-water",
               operators=("advection", "divergence"), bc=("periodic",) * 3,
               split_source="derived"),
    FamilySpec("diff_react", 2, "grid_npy", "cache/prose128/diff_react_n100000_t20.npy",
               channels=(6, 7), symbolic="dt(u) = Du*lap(u) + R(u,v); dt(v) = Dv*lap(v) + R(u,v); FitzHugh-Nagumo",
               operators=("diffusion", "reaction"), param_names=("Du", "Dv", "k"),
               param_source="const", param_const=(1e-3, 5e-3, 5e-3), bc=("periodic",) * 3),
    FamilySpec("com_ns", 2, "grid_npy", "cache/prose128/com_ns_n100000_t20_shards",
               channels=(0, 1, 3, 4), symbolic="dt(rho,rho*u,E) + div(fluxes) = 0; compressible Navier-Stokes",
               operators=("advection", "diffusion", "divergence", "pressure_grad"),
               param_names=("mach", "eta", "zeta"), param_source="attr:mach",
               bc=("periodic",) * 3),
    FamilySpec("incom_ns", 2, "grid_npy", "cache/prose128/incom_ns_n100000_t20_shards",
               channels=(0, 1, 6), symbolic="dt(u) + (u.grad)u = -grad(p) + nu*lap(u); div(u)=0; incompressible NS + passive scalar",
               operators=("advection", "diffusion", "pressure_projection"),
               param_names=("nu",), param_source="per_sample", bc=("periodic",) * 3),
    FamilySpec("cfdbench", 2, "grid_npy", "cache/prose128/cfdbench_n100000_t20.npy",
               channels=(0, 1), geom_kind="mask",
               symbolic="dt(u) + (u.grad)u = -grad(p) + nu*lap(u); div(u)=0; incompressible NS (walls)",
               operators=("advection", "diffusion", "pressure_projection"),
               bc=("wall", "wall", "periodic"),
               notes="raw ch2 = STATIC boundary mask -> geom interior-mask channel (slot-6 field "
                     "contamination fix, 2026-07-31: it was being supervised as a field AND "
                     "colliding with smoke/activator physics in slot 6)"),
    FamilySpec("pdearena_ns", 2, "grid_npy", "cache/prose128/pdearena_ns_n100000_t20_shards",
               channels=(0, 1, 6), symbolic="dt(u)+(u.grad)u = -grad(p)+nu*lap(u)+f; div(u)=0; NS + buoyant scalar",
               operators=("advection", "diffusion", "pressure_projection", "buoyancy"),
               param_names=("nu", "buoyancy"), param_source="per_sample", bc=("periodic",) * 3),
    FamilySpec("pdearena_uncond", 2, "grid_npy", "cache/prose128/pdearena_uncond_n100000_t20_shards",
               channels=(0, 1, 6), symbolic="dt(u)+(u.grad)u = -grad(p)+nu*lap(u)+f; div(u)=0; NS (unconditioned)",
               operators=("advection", "diffusion", "pressure_projection", "buoyancy"),
               bc=("periodic",) * 3, t_valid=14),
]

# --------------------------------------------------------------------------- #
# Poseidon operators (nc: sample,time,[channel],x,y). NS/CE pretrain; others downstream.
# --------------------------------------------------------------------------- #
def _pose(name, ch, sym, ops, stage="pretrain", params=(), psrc="none", bc=("periodic",) * 3,
          coef_file=()):
    return FamilySpec(name, 2, "grid_nchw", f"raw/poseidon_assembled/{name}.nc",
                      channels=ch, symbolic=sym, operators=ops, param_names=params,
                      param_source=psrc, bc=bc, stage=stage, split_source="official:poseidon",
                      coef_file=coef_file)

_POSEIDON = [
    _pose("CE-RP", (3, 0, 1, 4, 5), "compressible Euler, Riemann problem",
          ("advection", "divergence", "pressure_grad")),
    _pose("CE-CRP", (3, 0, 1, 4, 5), "compressible Euler, curved Riemann problem",
          ("advection", "divergence", "pressure_grad")),
    _pose("CE-KH", (3, 0, 1, 4, 5), "compressible Euler, Kelvin-Helmholtz",
          ("advection", "divergence", "pressure_grad", "shear")),
    _pose("CE-Gauss", (3, 0, 1, 4, 5), "compressible Euler, Gaussian ICs",
          ("advection", "divergence", "pressure_grad")),
    _pose("NS-Sines", (0, 1), "incompressible NS, sinusoidal forcing",
          ("advection", "diffusion", "pressure_projection")),
    FamilySpec("NS-Gauss", 2, "grid_nchw", "raw/poseidon/NS-Gauss",
               channels=(0, 1), symbolic="incompressible NS, Gaussian forcing",
               operators=("advection", "diffusion", "pressure_projection"),
               split_source="official:poseidon"),
    # downstream-only (held out from pretraining)
    # NOTE (2026-07-14): all families moved to pretrain — "train on everything, hold out only each
    # family's test split". OOD generalization is instead probed on an EXTERNAL novel scenario
    # (e.g. pitching / unsteady airfoil), testing emergence rather than held-out-family transfer.
    _pose("ACE", (6,), "dt(u) = eps*lap(u) + u - u^3; Allen-Cahn", ("diffusion", "reaction"),
          params=("eps",), psrc="const"),
    # The layered wave speed c(x) ships as a SEPARATE file and was previously unused, leaving the
    # IVP open: identical ICs under different layer structures have different solutions, so no
    # amount of capacity could close the gap (this was our worst family at ~32%). c_0.nc is
    # sample-index aligned with the assembled solution (assemble_data.py writes solution and c into
    # one index space in sorted file order; our assembled file holds the first 3504 = solution_0's
    # block). Alignment re-checked empirically: corr(Var_t u, c) is -0.12 +- 0.04 aligned vs
    # +0.02 +- 0.04 shuffled, the physically correct sign (fast layers = less temporal variance).
    # It enters through the zero-init geom gate, so a mis-pairing would simply never open.
    _pose("Wave-Layer", (6,), "dtt(u) = c(x)^2 * lap(u); wave, layered speed",
          ("wave",), coef_file=("raw/poseidon/Wave-Layer/c_0.nc", "c")),
    FamilySpec("Poisson-Gauss", 2, "grid_nchw", "raw/poseidon_assembled/Poisson-Gauss.nc",
               channels=(4,), symbolic="lap(u) = f; Poisson (steady, elliptic)",
               operators=("laplacian",), time="steady", nc_keys=("solution", "source"),
               split_source="official:poseidon"),
    _pose("CE-RM", (3, 0, 1, 4, 5), "compressible Euler, Richtmyer-Meshkov",
          ("advection", "divergence", "pressure_grad")),
]

# --------------------------------------------------------------------------- #
# 3D volumetric (PDEBench compressible NS, K=3)
# --------------------------------------------------------------------------- #
_PDEBENCH3D = [
    FamilySpec(f"pdebench3d_cns_{tag}", 3, "pdebench3d",
               f"raw/pdebench/3d/3D_CFD_{tag}_periodic_Train.hdf5",
               channels=(0, 1, 2, 3, 4),          # Vx,Vy,Vz,density,pressure -> slots 0,1,2,3,4
               symbolic="dt(rho,rho*u,E) + div(fluxes) = 0; 3D compressible Navier-Stokes",
               operators=("advection", "diffusion", "divergence", "pressure_grad"),
               param_names=("mach", "eta", "zeta"), param_source="filename",
               roles=(0, 1, 2), bc=("periodic", "periodic", "periodic"))
    for tag in ("Rand_M0.1_Eta1e-08_Zeta1e-08", "Rand_M1.0_Eta1e-08_Zeta1e-08",
                "Turb_M1.0_Eta1e-08_Zeta1e-08")
]

# --------------------------------------------------------------------------- #
# Mesh / geometry (steady RANS or elliptic; point-cloud native)
# --------------------------------------------------------------------------- #
_MESH = [
    FamilySpec("airfrans", 2, "mesh_airfrans", "raw/mesh/airfrans/Dataset",
               channels=(0, 1, 4), symbolic="RANS: (u.grad)u = -grad(p)+div((nu+nu_t)grad u); div(u)=0; airfoil",
               operators=("advection", "diffusion", "pressure_projection", "turbulence"),
               param_names=("reynolds", "aoa"), param_source="filename",
               bc=("wall", "open", "open"), time="steady", unit="case",
               geom_kind="sdf", split_source="official:airfrans"),
    FamilySpec("geofno_airfoil", 2, "mesh_geofno_airfoil", "raw/mesh/geofno/airfoil",
               channels=(4,), symbolic="transonic potential/Euler flow over airfoil; Mach field",
               operators=("advection", "pressure_grad"), bc=("wall", "open", "open"),
               time="steady", unit="design", split_source="official:geofno"),   # geom: C-grid wall ambiguous -> none
    FamilySpec("geofno_elasticity", 2, "mesh_geofno_elasticity", "raw/mesh/geofno/elasticity",
               channels=(4,), symbolic="div(sigma)=0; sigma=C:eps(u); linear elasticity, stress field",
               operators=("divergence", "elliptic"), bc=("wall", "open", "open"),
               time="steady", unit="design", geom_kind="sdf", split_source="official:geofno"),
    FamilySpec("geofno_pipe", 2, "mesh_geofno_pipe", "raw/mesh/geofno/pipe",
               channels=(0,), symbolic="incompressible NS in a parametrized pipe; velocity",
               operators=("advection", "diffusion", "pressure_projection"),
               bc=("wall", "wall", "open"), time="steady", unit="design",
               geom_kind="sdf", split_source="official:geofno"),
    FamilySpec("shapenet_car", 3, "mesh_shapenet_car", "raw/mesh/shapenet_car/mlcfd_data",
               channels=(4,), symbolic="steady incompressible RANS around a car; surface pressure",
               operators=("advection", "diffusion", "pressure_projection", "turbulence"),
               param_names=("inlet_velocity",), param_source="const", param_const=(20.0,),
               bc=("wall", "open", "open"), time="steady", unit="design", geom_kind="surface",
               split_source="derived", stage="pretrain"),   # mainstream mesh benchmark: trained, not held out
    FamilySpec("drivaernet_pressure", 3, "mesh_drivaernet", "raw/mesh/drivaernet/pressure",
               channels=(4,), symbolic="steady RANS (k-omega SST) around a car; surface pressure",
               operators=("advection", "diffusion", "pressure_projection", "turbulence"),
               param_names=("inlet_velocity",), param_source="const", param_const=(38.9,),
               bc=("wall", "open", "open"), time="steady", unit="design", geom_kind="surface",
               split_source="official:drivaernet"),   # mainstream: trained once materialized
]

# --------------------------------------------------------------------------- #
# Fig2 downstream (NOT in MOTION_FAMILIES / pretraining): few-shot transfer targets.
# APPENDED LAST so every pretrain family keeps its fam_id — warm-started per-family
# heads/embeddings stay aligned; the new families get fresh rows at the end.
# --------------------------------------------------------------------------- #
_FIG2_DOWNSTREAM = [
    _pose("NS-PwC", (0, 1), "incompressible NS, piecewise-constant vorticity ICs",
          ("advection", "diffusion", "pressure_projection"), stage="downstream"),
    # GCE-RT: gravitational compressible Euler (Rayleigh-Taylor). Raw nc channel order per
    # scOT RayleighTaylor: [rho, u, v, p, g]; the 6th file channel is unused (scOT
    # input_dim=5). g (gravitational potential) lands on the scalar slot 2 -> feeds the
    # buoyancy head: the OOD-recombination story (buoyancy x compressibility) of Fig2b.
    FamilySpec("GCE-RT", 2, "grid_nchw", "raw/poseidon/GCE-RT",
               channels=(3, 0, 1, 4, 2),
               symbolic="gravitational compressible Euler, Rayleigh-Taylor instability",
               operators=("advection", "divergence", "pressure_grad", "buoyancy"),
               stage="downstream", split_source="official:poseidon"),
    # PoolBoil (BubbleML Subcooled 2D): 10 trajectories (Twall 79-110C), phase-change boiling.
    # Fig2c: the NEW-mechanism insertion testbed (arm B adds a phase_interface bank).
    # Channels [velx, vely, temperature, dfun(SDF interface)] -> slots (0,1,5,6).
    # 08-11: pressure DROPPED — rms 6.95 dominates the joint metric while being unpredictable
    # (rel 2.4; Walrus measured 1.23 and excluded it too). 4-field set = Walrus canon.
    FamilySpec("PoolBoil-Sub", 2, "grid_nchw", "raw/bubbleml/poolboil_combo",
               channels=(0, 1, 5, 6),
               symbolic="incompressible two-phase boiling with phase change; subcooled pool",
               operators=("advection", "diffusion", "pressure_projection", "buoyancy"),
               param_names=("twall",), param_source="none",
               bc=("wall", "open", "periodic"), stage="downstream",
               split_source="official:combo-2.0train-1.0valtest"),
    # arm B: SAME data, operators additionally declare phase_change -> the op-mask opens the
    # new phase_interface bank. arm A (PoolBoil-Sub) never sees it (masked).
    FamilySpec("PoolBoil-SubX", 2, "grid_nchw", "raw/bubbleml/poolboil_combo",
               channels=(0, 1, 5, 6),
               symbolic="incompressible two-phase boiling with phase change; subcooled pool",
               operators=("advection", "diffusion", "pressure_projection", "buoyancy",
                          "phase_change"),
               param_names=("twall",), param_source="none",
               bc=("wall", "open", "periodic"), stage="downstream",
               split_source="official:combo-2.0train-1.0valtest"),
]

# Walrus-aligned respin of M1.0 (08-12): SAME raw file, split from the leakage audit --
# test = Walrus held-out tail, so both models are clean on it. Lazy path only (no cache),
# appended LAST so every existing fam_id is untouched; head row warmstarted from the
# original family (cold-row trap).
_WALIGN = [
    FamilySpec("pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08_W", 3, "pdebench3d",
               "raw/pdebench/3d/3D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5",
               channels=(0, 1, 2, 3, 4),
               symbolic="dt(rho,rho*u,E) + div(fluxes) = 0; 3D compressible Navier-Stokes",
               operators=("advection", "diffusion", "divergence", "pressure_grad"),
               param_names=("mach", "eta", "zeta"), param_source="filename",
               roles=(0, 1, 2), bc=("periodic", "periodic", "periodic"),
               split_source="official:walrusleak", stage="downstream"),
]

# PB arm C (08-12): SAME data/vocab as SubX, plus the evolving-interface geometry
# pathway (anchor-frame dfun -> geom features + sdf_box + interface-biased sampling).
_ARMC = [
    FamilySpec("PoolBoil-SubG", 2, "grid_nchw", "raw/bubbleml/poolboil_combo",
               channels=(0, 1, 5, 6),
               symbolic="incompressible two-phase boiling with phase change; subcooled pool",
               operators=("advection", "diffusion", "pressure_projection", "buoyancy",
                          "phase_change"),
               param_names=("twall",), param_source="none",
               bc=("wall", "open", "periodic"), stage="downstream",
               geom_kind="interface",
               split_source="official:combo-2.0train-1.0valtest"),
]

FAMILIES = {f.name: f for f in (_PROSE + _POSEIDON + _PDEBENCH3D + _MESH + _FIG2_DOWNSTREAM
                                + _WALIGN + _ARMC)}


# MOTION-NCS corpus (2026-08-06, motion_ncs/PLAN_NCS.md): field-based grid physics only,
# 2D + 3D, one weight set. Geometry (mesh/surface) is a separate later paper; cfdbench is
# excluded because its persistence floor (0.198%) leaves 0.07 points of learnable signal, so
# no number computed on it is interpretable. 19 families = 16 x 2D + 3 x 3D.
MOTION_FAMILIES = (
    "shallow_water", "diff_react", "com_ns", "incom_ns", "pdearena_ns", "pdearena_uncond",
    "CE-RP", "CE-CRP", "CE-KH", "CE-Gauss", "CE-RM", "NS-Sines", "NS-Gauss",
    "ACE", "Wave-Layer", "Poisson-Gauss",
    "pdebench3d_cns_Rand_M0.1_Eta1e-08_Zeta1e-08",
    "pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08",
    "pdebench3d_cns_Turb_M1.0_Eta1e-08_Zeta1e-08",
)


def motion_families():
    return [FAMILIES[n] for n in MOTION_FAMILIES]


def pretrain_families():
    return [f for f in FAMILIES.values() if f.stage == "pretrain"]


def downstream_families():
    return [f for f in FAMILIES.values() if f.stage == "downstream"]


def by_dim(K):
    return [f for f in FAMILIES.values() if f.K == K]
