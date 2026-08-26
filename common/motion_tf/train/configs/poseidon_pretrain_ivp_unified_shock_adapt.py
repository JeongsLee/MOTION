"""PHASE-2 IVP — full NTO-ADA shock recipe ported: gated-upwind flux (use_shock) + SHOCK-ADAPTIVE
RESOLUTION (shock_adapt). The Burgers paper's shock capture came from (a) gated upwind conv AND (b)
ν-scaled positional encoding (resolution adapts to the shock-sharpness regime) — NOT the PI residual
(which was optional/ineffective). use_shock alone (= the gated-upwind part) was ~tied at matched samples;
this adds the missing piece — a DATA-derived shock sensor (|∇ IC-velocity|, no equation key) that amplifies
the learned high-freq upsample residual where shocks are, i.e. the data-only analog of ν-scaled PE.
loss = rel-L2 (inherited, now active). cold 320k, 2×H100. Compare to no-shock baseline at MATCHED SAMPLES
(this is 2-GPU gb8: step S = no-shock-1g step 2S)."""
from .poseidon_pretrain_ivp_unified import Cfg as _Base


class Cfg(_Base):
    unified_shock = True       # gated-upwind flux feature (Burgers UpwindConv1D 2D analog)
    shock_adapt = True         # shock-sensor-driven adaptive hf resolution (ν-scaled-PE analog)
    steps = 320000
    seed = 4
    save_dir = "results/poseidon_pretrain_ivp_unified_shock_adapt"
