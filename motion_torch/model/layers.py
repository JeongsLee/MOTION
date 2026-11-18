"""Basic layers of the PyTorch port. Parameters are stored in the TensorFlow/Keras LAYOUT (Dense kernel (in, out),
Conv1D kernel (k, in, out), MHA kernels (d, heads, key_dim)) so that weight conversion is a name-keyed copy; every
parameter records its TF variable name (`tf_names`, used by convert.py).

Activations: Keras "gelu" is the exact (erf) GELU -> F.gelu(approximate="none").
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_ROLES = 4


def gelu(x):
    return F.gelu(x)


class TFModule(nn.Module):
    """nn.Module whose parameters carry their TF variable names (local param name -> TF name)."""

    def __init__(self):
        super().__init__()
        self.tf_names = {}

    def tf_param(self, name, shape, tf_name, init="zeros"):
        if callable(init):
            t = init(shape)
        elif init == "zeros":
            t = torch.zeros(shape)
        elif init == "ones":
            t = torch.ones(shape)
        else:
            raise ValueError(init)
        p = nn.Parameter(t.float())
        self.register_parameter(name, p)
        self.tf_names[name] = tf_name
        return p


def glorot(shape):
    """Keras glorot_uniform for Dense (in, out) kernels (fresh init only; loaded weights overwrite it)."""
    if len(shape) == 2:
        fi, fo = shape
    elif len(shape) == 3:                       # conv (k, in, out) / einsum (d, h, k)
        fi, fo = shape[0] * shape[1], shape[0] * shape[2]
    else:
        fi = fo = int(np.prod(shape))
    lim = math.sqrt(6.0 / (fi + fo))
    return torch.empty(shape).uniform_(-lim, lim)


def normal(std):
    return lambda shape: torch.randn(shape) * std


class Dense(TFModule):
    """Keras Dense: y = act(x @ kernel + bias), kernel (in, out). `tf` = the layer's TF scope, e.g. "enc/enc_f"."""

    def __init__(self, n_in, n_out, tf, bias=True, act=None, init="glorot"):
        super().__init__()
        ki = glorot if init == "glorot" else ("zeros" if init == "zeros" else init)
        self.tf_param("kernel", (n_in, n_out), f"{tf}/kernel:0", ki)
        self.use_bias = bias
        if bias:
            self.tf_param("bias", (n_out,), f"{tf}/bias:0", "zeros")
        self.act = act

    def forward(self, x):
        y = torch.matmul(x, self.kernel)
        if self.use_bias:
            y = y + self.bias
        if self.act == "gelu":
            y = gelu(y)
        return y


class RankFreeLN(TFModule):
    """core/axops.RankFreeLN: LayerNorm over the last axis with (d,) gamma/beta, any rank."""

    def __init__(self, d, name, eps=1e-5):
        super().__init__()
        self.tf_param("g", (d,), f"{name}_g:0", "ones")
        self.tf_param("b", (d,), f"{name}_b:0", "zeros")
        self.eps = eps

    def forward(self, x):
        m = x.mean(-1, keepdim=True)
        v = torch.square(x - m).mean(-1, keepdim=True)
        return self.g * (x - m) * torch.rsqrt(v + self.eps) + self.b


def to_axis_major(x, axis):
    """(B, n1..nK, D), spatial axis a -> ((B*, n_a, D), restore). Same fold as core/axops._to_axis_major."""
    xt = x.movedim(1 + axis, -2)
    shp = xt.shape
    flat = xt.reshape(-1, shp[-2], shp[-1])

    def restore(y):
        return y.reshape(shp).movedim(-2, 1 + axis)

    return flat, restore


class AxialConv(TFModule):
    """core/axops.AxialConv: shared periodic 1-D conv (kernel 5) along every spatial axis + zero-init mix."""

    def __init__(self, d, name, kernel=5):
        super().__init__()
        self.k = kernel
        self.tf_param("role", (MAX_ROLES, d), f"{name}_role:0", "zeros")
        self.tf_param("conv_kernel", (kernel, d, d), f"{name}/{name}_c1d/kernel:0", glorot)
        self.tf_param("conv_bias", (d,), f"{name}/{name}_c1d/bias:0", "zeros")
        self.mix = Dense(d, d, f"{name}/{name}_mix", init="zeros")

    def forward(self, x, roles):
        K = x.dim() - 2
        p = self.k // 2
        w = self.conv_kernel.permute(2, 1, 0)          # (k, in, out) -> (out, in, k)
        acc = None
        for a in range(K):
            xa = x + self.role[roles[a] % MAX_ROLES]
            flat, restore = to_axis_major(xa, a)       # (B*, n, d)
            f = flat.transpose(1, 2)                   # (B*, d, n)
            f = torch.cat([f[..., -p:], f, f[..., :p]], -1)   # periodic wrap
            y = F.conv1d(f, w, self.conv_bias).transpose(1, 2)
            y = restore(y)
            acc = y if acc is None else acc + y
        h = gelu(acc / float(K))
        return x + self.mix(h)


class MHA(TFModule):
    """tf.keras.layers.MultiHeadAttention (self-attention use, no mask, no dropout) with Keras weight layout."""

    def __init__(self, d, heads, key_dim, tf):
        super().__init__()
        self.h, self.dk = heads, key_dim
        for nm in ("query", "key", "value"):
            self.tf_param(f"{nm}_kernel", (d, heads, key_dim), f"{tf}/{nm}/kernel:0", glorot)
            self.tf_param(f"{nm}_bias", (heads, key_dim), f"{tf}/{nm}/bias:0", "zeros")
        self.tf_param("out_kernel", (heads, key_dim, d), f"{tf}/attention_output/kernel:0", glorot)
        self.tf_param("out_bias", (d,), f"{tf}/attention_output/bias:0", "zeros")

    def forward(self, x):                              # x (B, T, d)
        q = torch.einsum("btc,chk->bthk", x, self.query_kernel) + self.query_bias
        k = torch.einsum("btc,chk->bthk", x, self.key_kernel) + self.key_bias
        v = torch.einsum("btc,chk->bthk", x, self.value_kernel) + self.value_bias
        q = q * (1.0 / math.sqrt(float(self.dk)))      # keras scales the query before the dot product
        s = torch.einsum("bshk,bthk->bhts", k, q)      # (B, H, Tq, Tk)
        a = torch.softmax(s, -1)
        o = torch.einsum("bhts,bshk->bthk", a, v)      # (B, Tq, H, dk)
        return torch.einsum("bthk,hkc->btc", o, self.out_kernel) + self.out_bias


class AxialAttention(TFModule):
    """core/axops.AxialAttention (reference path, MOTION_FAST_ATTN off)."""

    def __init__(self, d, name, heads=4):
        super().__init__()
        self.ln = RankFreeLN(d, f"{name}_ln")
        self.mha = MHA(d, heads, max(d // heads, 8), f"{name}/{name}_mha")
        self.tf_param("role", (MAX_ROLES, d), f"{name}_role:0", "zeros")
        self.tf_param("gate", (), f"{name}_gate:0", "zeros")

    def forward(self, x, roles):
        K = x.dim() - 2
        acc = None
        lx = self.ln(x)
        for a in range(K):
            xa = lx + self.role[roles[a] % MAX_ROLES]
            flat, restore = to_axis_major(xa, a)
            y = restore(self.mha(flat))
            acc = y if acc is None else acc + y
        return x + self.gate * (acc / float(K))


class PointwiseMLP(TFModule):
    def __init__(self, d, name, ratio=2):
        super().__init__()
        self.h = Dense(d, d * ratio, f"{name}/{name}_h", act="gelu")
        self.o = Dense(d * ratio, d, f"{name}/{name}_o", init="zeros")
        self.ln = RankFreeLN(d, f"{name}_ln")

    def forward(self, x):
        return x + self.o(self.h(self.ln(x)))


def fourier_freqs(n_freq=16):
    # tf.constant(2.0 ** np.arange(n), tf.float32) * np.pi  (float32 product; exact power-of-two scaling)
    return torch.from_numpy((np.float32(2.0) ** np.arange(n_freq, dtype=np.float32)) * np.float32(np.pi))


class CoordEncoder(TFModule):
    """core/pointio.CoordEncoder: shared per-coordinate Fourier features + axis-role embedding -> Dense(d)."""

    def __init__(self, d, name, scope="", n_freq=16):
        super().__init__()
        self.register_buffer("freq", fourier_freqs(n_freq), persistent=False)
        self.proj = Dense(2 * n_freq, d, f"{scope + '/' if scope else ''}{name}/{name}_p")
        self.tf_param("role", (MAX_ROLES, 2 * n_freq), f"{name}_role:0", "zeros")

    def forward(self, pts, roles):
        K = pts.shape[-1]
        acc = None
        for a in range(K):
            ph = pts[..., a:a + 1].float() * self.freq.float()      # f32 phase, as TF (2^15*pi*x needs it)
            ff = torch.cat([torch.sin(ph), torch.cos(ph)], -1).to(self.role.dtype) + self.role[roles[a] % MAX_ROLES]
            acc = ff if acc is None else acc + ff
        return self.proj(acc)
