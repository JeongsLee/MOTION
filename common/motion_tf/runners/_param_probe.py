"""Build PhysicsOperatorMixture under candidate capacity knobs; print param count to hit ~158M (Poseidon-B)."""
import os, importlib, numpy as np, tensorflow as tf
tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
Base = importlib.import_module("motion_tf.train.configs.poseidon_pretrain_ivp_unified").Cfg
from motion_tf.model import PhysicsOperatorMixture

def count(overrides, tag):
    class C(Base): pass
    for k,v in overrides.items(): setattr(C, k, v)
    m = PhysicsOperatorMixture(C)
    _ = m.call_with_gate(tf.zeros((1,C.Nx,C.Nx,C.n_channels)), tf.zeros((1,C.desc_dim)), tf.zeros((1,4)), training=False)
    n = int(sum(np.prod(v.shape) for v in m.trainable_variables))
    print(f"  {tag}: {n:,}  ({overrides})", flush=True)
    return n

print("=== param counts ===", flush=True)
count({}, "BASELINE")
count({"geo_layers":14, "router_depth":4, "n_latent":24}, "A depth(geo14,rdep4,nlat24)")
count({"expert_hidden":1280, "n_latent":32}, "B width(eh1280,nlat32)")
count({"geo_layers":13, "router_depth":4, "expert_hidden":1152, "n_latent":24}, "C mix")
count({"geo_layers":16, "router_depth":6}, "D depth-only(geo16,rdep6)")
print("PROBE_DONE", flush=True)
