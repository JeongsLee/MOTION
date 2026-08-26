"""PHASE-1 Task-1 (UNIFIED) — add an FNO-style SPECTRAL global branch to the UnifiedExpert head.

Replaces the weak global-mean scalar with an FNO SpectralConv2d (2D FFT → keep low modes → learned complex
per-mode channel mixing → iFFT), parallel to the local dilated-conv. Global, parameter-efficient, the
canonical fluid-PDE inductive bias (FNO). CAVEAT (user's prior observation): the ADA basis is TEMPORAL
Fourier, so this stacks spatial+temporal Fourier ("double sin") which previously meshed poorly — this run
EMPIRICALLY tests it on the current unified arch (all other structural tweaks ~tied the baseline). zero-init
fno_proj → safe start. vs unified Task-1 baseline (pdf-fluid5-unified2, @15k overall 10.43 / pdearena 22.10)."""
from .prose_150M_fluid5_6x_unified import Cfg as _Base


class Cfg(_Base):
    unified_fno = True
    unified_fno_modes = 12     # low modes per H-corner / W (latent grid 32² → max 16)
    unified_fno_width = 64
    save_dir = "results/prose_150M_fluid5_6x_unified_fno"
