"""Phase-1 Task-1 advect, latent-ON + MUSCL (minmod) self-advection — tests whether a 2nd-order TVD
advection scheme (low numerical diffusion) lets the explicit −u·∇z BEAT the baseline, vs the 1st-order
upwind arm which was ~tied/slightly-behind (1st-order upwind's numerical diffusion, amplified by the
coarse latent_factor=4 grid, over-smooths turbulent NS). Same as advect_lat except minmod instead of
upwind. Compare per-family physical rel-L2 (esp pdearena_ns / incom_ns) vs baseline-lup AND vs the
upwind advect_lat arm at matched steps."""
from .prose_150M_fluid5_6x_lup import Cfg as _Base


class Cfg(_Base):
    conv_self_advect = True
    conv_muscl = True              # minmod 2nd-order TVD (takes precedence over upwind)
    save_dir = "results/prose_150M_fluid5_6x_advect_muscl"
