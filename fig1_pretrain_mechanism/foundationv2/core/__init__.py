"""foundationv2 P2 core: dimension-agnostic latent-box operators.

One parameter set serves K=2 and K=3: every spatial operator is built from
(a) a SHARED 1D primitive applied along each axis in turn, and (b) pointwise
channel mixing — K enters only as the number of axis applications. Attention
QKV/MLP weights are dimension-blind by construction (token-set operations).
"""
from .axops import AxialAttention, AxialConv, PointwiseMLP
from .latentbox import LatentBoxCore

__all__ = ["AxialConv", "AxialAttention", "PointwiseMLP", "LatentBoxCore"]
