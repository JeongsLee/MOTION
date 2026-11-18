"""Configuration of the PyTorch port: every environment switch of the TF reference (motion_tf) that changes the
architecture, the conditioning layout or the data normalization is an explicit field here.

The TF code reads these switches from os.environ at IMPORT time; the port reads them from a MotionConfig object
(`MotionConfig.from_env()` reproduces the TF behaviour for a given environment). Presets for the released models are
in PRESETS (see README.md for the mapping).
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field


def _b(name, default="0"):
    return os.environ.get(name, default) == "1"


@dataclass
class MotionConfig:
    # ---------------------------------------------------------------- architecture (V3Model constructor args)
    d: int = 384
    depth: int = 8
    d_w: int = 16
    n_p: int = 16
    d_cond: int = 128
    t_in: int = 10
    S: int = 16                       # data.registry.NUM_SLOTS
    n_semantic: int = 12              # data.registry.SEMANTIC_SLOTS
    dims2: tuple = (64, 64)
    dims3: tuple = (32, 32, 32)
    transport: bool = True
    gate_cond: int = 1
    fam_head: int = 1
    dec_hidden: int = 96
    n_fams: int = 32                  # family rows of dec_famW: 32 (MOTION-S/-L/-NS), 47 (MOTION-IVP: + the 15 FT-* tasks)
    # ---------------------------------------------------------------- environment switches (TF env name in brackets)
    legacy_vocab: bool = False        # [MOTION_LEGACY_VOCAB] 13-token operator vocabulary (no 'phase_change')
    vortex2d_legacy: bool = False     # [MOTION_VORTEX2D_LEGACY] evaluate the vortex head (bias) also for K=2
    warp_live: bool = False           # [MOTION_WARP_LIVE] init-only (disp random init); no effect on a loaded ckpt
    bank_ngeneric: int = 0            # [MOTION_BANK_NGENERIC] number of generic bank heads
    bank_generic_merge: bool = False  # [MOTION_BANK_GENERIC_MERGE] generic heads merged behind one gate
    box_uniform: int = 0              # [MOTION_BOX_UNIFORM] uniform latent box side (else per-family BOX_TABLE)
    input_zscore: bool = False        # [MOTION_INPUT_ZSCORE] instance z-score (mean removed, carried in 'shift')
    cond_stats: bool = False          # [MOTION_COND_STATS] append window statistics to cond (+2*S dims)
    global_norm: bool = False         # [MOTION_GLOBAL_NORM] fixed per-family mean/std (needs input_zscore)
    force_mode: str = ""              # [MOTION_FORCE_MODE] e.g. "ss": every transient family single-shot
    ss_warp: str = "0"                # [MOTION_SS_WARP] "0" | "1" (taylor) | "blend"
    ft_nometa: bool = False           # [MOTION_FT_NOMETA] metadata-free examples (all op tokens on, no params)
    rh_fams: tuple = ()               # [MOTION_RH_FAMS] families in receding-horizon mode
    no_dead_repair: bool = True       # [NO_DEAD_REPAIR] restore the checkpoint exactly (see convert.py)
    ti_fixed: int = 0                 # [MOTION_TI_FIXED]
    # unsupported-in-port switches (asserted off): MOTION_CORE, MOTION_STD_DECODER, MOTION_BANK_GENERIC,
    # MOTION_BANK_NOMETA, MOTION_BANK_NOH0, MOTION_BANK_SINGLE, MOTION_BANK_PHYSGATE, MOTION_WARP_MODE!=blend,
    # MOTION_FAST_ATTN (exact re-association, outputs equal to float tolerance: the port uses the reference path).
    name: str = "custom"

    # ---------------------------------------------------------------- derived
    @property
    def n_ops(self) -> int:
        return 13 if self.legacy_vocab else 14

    @property
    def cond_dim(self) -> int:
        # len(OPERATOR_VOCAB) + MAX_PARAMS*PARAM_FEAT_DIM + len(PARAM_VOCAB) + 3*3 + 1 + 2 (+ 2*S)
        return self.n_ops + 6 * 18 + 13 + 9 + 1 + 2 + (2 * self.S if self.cond_stats else 0)

    @property
    def ss_warp_on(self) -> bool:
        return self.transport and self.ss_warp in ("1", "blend")

    @property
    def ss_warp_mode(self) -> str:
        return "blend" if self.ss_warp == "blend" else "taylor"

    def replace(self, **kw) -> "MotionConfig":
        return dataclasses.replace(self, **kw)

    def env(self) -> dict:
        """The TF environment that reproduces this configuration (for the reference implementation)."""
        e = {}
        if self.legacy_vocab: e["MOTION_LEGACY_VOCAB"] = "1"
        if self.vortex2d_legacy: e["MOTION_VORTEX2D_LEGACY"] = "1"
        if self.warp_live: e["MOTION_WARP_LIVE"] = "1"
        if self.bank_ngeneric: e["MOTION_BANK_NGENERIC"] = str(self.bank_ngeneric)
        if self.bank_generic_merge: e["MOTION_BANK_GENERIC_MERGE"] = "1"
        if self.box_uniform: e["MOTION_BOX_UNIFORM"] = str(self.box_uniform)
        if self.input_zscore: e["MOTION_INPUT_ZSCORE"] = "1"
        if self.cond_stats: e["MOTION_COND_STATS"] = "1"
        if self.global_norm: e["MOTION_GLOBAL_NORM"] = "1"
        if self.force_mode: e["MOTION_FORCE_MODE"] = self.force_mode
        if self.ss_warp != "0": e["MOTION_SS_WARP"] = self.ss_warp
        if self.ft_nometa: e["MOTION_FT_NOMETA"] = "1"
        if self.rh_fams: e["MOTION_RH_FAMS"] = ",".join(self.rh_fams)
        if self.no_dead_repair: e["NO_DEAD_REPAIR"] = "1"
        if self.ti_fixed: e["MOTION_TI_FIXED"] = str(self.ti_fixed)
        return e

    @classmethod
    def from_env(cls, **arch) -> "MotionConfig":
        for k in ("MOTION_CORE",):
            assert os.environ.get(k, "axial") == "axial", f"{k} not supported by the port"
        for k in ("MOTION_STD_DECODER", "MOTION_BANK_GENERIC", "MOTION_BANK_NOMETA", "MOTION_BANK_NOH0",
                  "MOTION_BANK_PHYSGATE"):
            assert not _b(k), f"{k}=1 not supported by the port"
        assert not os.environ.get("MOTION_BANK_SINGLE"), "MOTION_BANK_SINGLE not supported by the port"
        assert os.environ.get("MOTION_WARP_MODE", "blend") == "blend", "MOTION_WARP_MODE != blend not supported"
        return cls(legacy_vocab=_b("MOTION_LEGACY_VOCAB"), vortex2d_legacy=_b("MOTION_VORTEX2D_LEGACY"),
                   warp_live=_b("MOTION_WARP_LIVE"), bank_ngeneric=int(os.environ.get("MOTION_BANK_NGENERIC", "0")),
                   bank_generic_merge=_b("MOTION_BANK_GENERIC_MERGE"),
                   box_uniform=int(os.environ.get("MOTION_BOX_UNIFORM", "0")),
                   input_zscore=_b("MOTION_INPUT_ZSCORE"), cond_stats=_b("MOTION_COND_STATS"),
                   global_norm=_b("MOTION_GLOBAL_NORM"), force_mode=os.environ.get("MOTION_FORCE_MODE", ""),
                   ss_warp=os.environ.get("MOTION_SS_WARP", "0"), ft_nometa=_b("MOTION_FT_NOMETA"),
                   rh_fams=tuple(f for f in os.environ.get("MOTION_RH_FAMS", "").split(",") if f),
                   no_dead_repair=bool(os.environ.get("NO_DEAD_REPAIR")),
                   ti_fixed=int(os.environ.get("MOTION_TI_FIXED", "0")), name="env", **arch)


PRESETS = {
    # paper MOTION-S (m1f, 20.7M): 19 MOTION families, AR / ss / steady per v3/policy.py, per-family box table
    "motion_s": MotionConfig(name="motion_s", legacy_vocab=True, vortex2d_legacy=True, no_dead_repair=True),
    # paper MOTION (m2s2, 156.9M): same switches as MOTION-S, wider/deeper
    "motion": MotionConfig(name="motion", d=896, depth=12, d_w=48, legacy_vocab=True, vortex2d_legacy=True,
                           no_dead_repair=True),
    # MOTION-NS (zsc, 21.8M): 6 joint-benchmark NS families, z-scored inputs, uniform 64^2 box, merged generic heads
    # (NO_DEAD_REPAIR was not set for the reference runs; for this checkpoint the repair only touches zero-init
    # tensors, so the converted weights are identical either way -- see convert.py)
    "motion_ns": MotionConfig(name="motion_ns", warp_live=True, bank_ngeneric=8, bank_generic_merge=True,
                              box_uniform=64, input_zscore=True, cond_stats=True, no_dead_repair=False),
    # MOTION-IVP-S (not released locally): single-shot from one IC frame, 20 leads, fixed per-family normalization
    # (jobs/run_ivps_ss.sh: --n_p 32 --t_in 1 --k_fut 20 --box_table 0 --dims2 64, merged generic heads, metadata off)
    "motion_ivp_s": MotionConfig(name="motion_ivp_s", n_fams=47, n_p=32, t_in=1, force_mode="ss", ss_warp="blend",
                                 global_norm=True, input_zscore=True, warp_live=True, bank_ngeneric=8,
                                 bank_generic_merge=True, box_uniform=64, ft_nometa=True, no_dead_repair=False),
}
