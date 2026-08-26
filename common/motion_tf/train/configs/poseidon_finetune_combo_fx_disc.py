"""Physics-block transfer finetune (Poseidon-style DISCRIMINATIVE LR): from the 305k combo pretrain, train
the PHYSICS BLOCK (learned tendency heads reaction/wave/gradx/buoyancy/adapter/advbias + the FIXED exact-op
branch fixedops + the latent decoder) at a HIGH lr, while the transformer BACKBONE (encoder + main conv + W
head + upsample) stays at a LOW lr — nothing frozen, just differential. This is how Poseidon transfers
(decoder/recovery head high, backbone low). Tests whether concentrating the task adaptation in the physics
operators (rather than letting the whole net re-fit, which left fixedops unused) improves OOD/in-family
transfer. init_from must be the NAMED 305k ckpt (fixedops vars fresh-init). FT_LR_DEC / FT_LR override.
"""
import os
from .poseidon_finetune_combo_fx import Cfg as _Base   # unified_fixedops=True


class Cfg(_Base):
    decoder_keys = ("u_rx", "u_wv", "u_ad", "u_by", "u_gx", "u_av", "u_fx",   # physics tendency heads + fixed ops
                    "decode_h", "decode_out", "w_up")                          # latent decoder + upsample
    lr_decoder = float(os.environ.get("FT_LR_DEC", "5e-5"))                    # physics block + decoder: HIGH lr
    lr = float(os.environ.get("FT_LR", "5e-6"))                               # transformer backbone: LOW lr
