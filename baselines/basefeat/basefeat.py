"""REVISION-R1 (R2 comment: 'add the same physical features to BCAT and PROSE-FD'), 2026-11-04.

Physical operator features computed INSIDE the baseline model, on the current input frames, before the patch
embedding. Targets, loss and the autoregressive rollout are unchanged; features are recomputed from the model's own
predictions during generation, and they transform with the input under the dihedral group.

  BASEFEAT=off  (default) -> features(x) returns x unchanged and n_extra() = 0: byte-identical baseline.
  BASEFEAT=phys           -> x is concatenated with 4C+1 feature maps (C = state channels, tie_fields layout):
        gradient   d_H q, d_W q          (2C)   ~ MOTION directional gradient
        laplacian  lap q                 (C)    ~ diffusion / coefficient-weighted diffusion
        advection  u . grad q            (C)    ~ advection / upwind transport
        divergence div u                 (1)    ~ dilatation
  BASEFEAT_KO=grad,lap,...  zeroes the named groups (evaluation-time feature-group knockout).

Layout facts (checked on prebuilt_n900, 2026-11-04): slots 0/1 = (Vx, Vy) in every fluid family except shallow water
(slot 0 = h); Vx runs along the first spatial axis (H): |div u| with (u0~H, u1~W) is 4-9x smaller than with the
swapped convention on incom_ns and pdearena_ns. BCAT is data-only, so the same slot convention is applied to every
family (for shallow water u = (h, 0); documented, not corrected, to avoid giving BCAT family information).
Differences: central, index units, replicate padding (the families mix periodic and wall boundaries). Each feature map
is scaled by its per-frame RMS over the grid (detached), so the scaling is identical in training (whole sequence) and in
generation (one new frame at a time).
"""
import os
import torch
import torch.nn.functional as F

MODE = os.environ.get("BASEFEAT", "off")
KO = {g for g in os.environ.get("BASEFEAT_KO", "").split(",") if g}
GROUPS = ("grad", "lap", "adv", "div")


def n_extra(c):
    return 4 * c + 1 if MODE == "phys" else 0


def group_slices(c):
    """channel ranges of each group inside the appended block (after the C state channels)."""
    return {"grad": (0, 2 * c), "lap": (2 * c, 3 * c), "adv": (3 * c, 4 * c), "div": (4 * c, 4 * c + 1)}


def features(x):
    """x: (N, C, H, W) -> (N, C + n_extra(C), H, W)."""
    if MODE != "phys":
        return x
    xf = x.float()
    c = xf.shape[1]
    p = F.pad(xf, (1, 1, 1, 1), mode="replicate")
    dH = 0.5 * (p[:, :, 2:, 1:-1] - p[:, :, :-2, 1:-1])
    dW = 0.5 * (p[:, :, 1:-1, 2:] - p[:, :, 1:-1, :-2])
    lap = p[:, :, 2:, 1:-1] + p[:, :, :-2, 1:-1] + p[:, :, 1:-1, 2:] + p[:, :, 1:-1, :-2] - 4.0 * xf
    u = xf[:, 0:1]
    v = xf[:, 1:2] if c > 1 else torch.zeros_like(u)
    adv = u * dH + v * dW
    div = dH[:, 0:1] + (dW[:, 1:2] if c > 1 else torch.zeros_like(u))
    f = torch.cat([dH, dW, lap, adv, div], 1)
    f = f / f.pow(2).mean((2, 3), keepdim=True).add(1e-6).sqrt().detach()
    if KO:
        sl = group_slices(c)
        keep = torch.ones(f.shape[1], device=f.device, dtype=f.dtype)
        for g in KO:
            a, b = sl[g]
            keep[a:b] = 0
        f = f * keep.view(1, -1, 1, 1)
    return torch.cat([x, f.to(x.dtype)], 1)
