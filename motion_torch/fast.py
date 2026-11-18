"""Accelerated execution of the MOTION trunk (same parameters, same function; revision R1, 2026-11-06).

  from motion_torch.fast import accelerate
  accelerate(model)                     # in place; parameters, names and checkpoints unchanged

Mathematically identical re-arrangements of the reference computation (only floating-point summation order changes):

  AxialConv       sum over axes a and taps j of roll(x + r_a, j-p, a) @ W_j  ->  ONE GEMM on the stacked shifted copies
                  [x, roll(x, +-1, a), roll(x, +-2, a), ...] @ [K*W_p; W_j ...] plus a constant (role and bias terms,
                  which the convolution maps to constants). The centre tap is shared by all axes (9 taps instead of
                  10 in 2-D, 13 instead of 15 in 3-D), and the K separate conv1d calls with their transposes become one
                  large matrix product.
  AxialAttention  the query/key/value projections of (LN(x) + r_a) are LN(x) W + r_a W: computed ONCE for all axes
                  (plus a per-axis constant); the K output projections sum to (sum_a o_a) W_o: ONE output projection.
                  The attention itself uses torch scaled_dot_product_attention (fused / flash kernels on GPU).

Everything else (encoder, FiLM, banks, decoder, transport) is unchanged. Combine with bf16 autocast and torch.compile
for the measurements of the response (bench_fast.py).
"""
from __future__ import annotations

import types

import torch
import torch.nn.functional as F

from .model.layers import MAX_ROLES, AxialAttention, AxialConv, gelu


def _shift(x, s, axis):
    """x[n + s] along spatial axis `axis` (periodic), x (B, n1..nK, d)."""
    return torch.roll(x, shifts=-s, dims=1 + axis)


def _conv_fast(self, x, roles):
    K = x.dim() - 2
    p = self.k // 2
    W = self.conv_kernel                                    # (k, in, out)
    fw = getattr(self, "_fast_W_frozen", None)               # set by freeze_fast_weights() before torch.compile
    if fw is not None:
        acc = torch.cat([x] + [_shift(x, j - p, a) for a in range(K) for j in range(self.k) if j != p], -1) @ fw
        acc = acc + self._fast_const
        return x + self.mix(gelu(acc / float(K)))
    if self.training or torch.compiler.is_compiling():
        # differentiable, no attribute writes (training; compiled graphs): fused weight built in the forward
        fwt = torch.cat([W[p] * float(K)] + [W[j] for a in range(K) for j in range(self.k) if j != p], 0)
        feats = [x] + [_shift(x, j - p, a) for a in range(K) for j in range(self.k) if j != p]
        const = sum(self.role[roles[a] % MAX_ROLES] for a in range(K)) @ W.sum(0) + float(K) * self.conv_bias
        acc = torch.cat(feats, -1) @ fwt.to(x.dtype) + const.to(x.dtype)
        return x + self.mix(gelu(acc / float(K)))
    key = (K, x.dtype, x.device, W._version)
    if getattr(self, "_fast_key", None) != key:
        blocks = [W[p] * float(K)]
        for a in range(K):
            for j in range(self.k):
                if j != p:
                    blocks.append(W[j])
        self._fast_W = torch.cat(blocks, 0)                 # ((1 + K(k-1)) d, d)
        self._fast_key = key
    feats = [x]
    for a in range(K):
        for j in range(self.k):
            if j != p:
                feats.append(_shift(x, j - p, a))
    acc = torch.cat(feats, -1) @ self._fast_W.to(x.dtype)
    Wsum = W.sum(0)                                         # constant input r maps to r @ sum_j W_j
    const = sum(self.role[roles[a] % MAX_ROLES] for a in range(K)) @ Wsum + float(K) * self.conv_bias
    acc = acc + const.to(x.dtype)
    h = gelu(acc / float(K))
    return x + self.mix(h)


def _attn_fast(self, x, roles):
    K = x.dim() - 2
    m = self.mha
    H, dk = m.h, m.dk
    lx = self.ln(x)
    shp = lx.shape
    d = shp[-1]
    Wqkv = torch.cat([m.query_kernel.reshape(d, H * dk), m.key_kernel.reshape(d, H * dk),
                      m.value_kernel.reshape(d, H * dk)], 1)
    bqkv = torch.cat([m.query_bias.reshape(-1), m.key_bias.reshape(-1), m.value_bias.reshape(-1)], 0)
    qkv0 = lx @ Wqkv.to(lx.dtype) + bqkv.to(lx.dtype)      # (B, n.., 3 H dk), once for all axes
    osum = None
    for a in range(K):
        qkv = qkv0 + (self.role[roles[a] % MAX_ROLES] @ Wqkv).to(lx.dtype)
        t = qkv.movedim(1 + a, -2)                          # (B, others.., n_a, 3Hdk)
        tshp = t.shape
        t = t.reshape(-1, tshp[-2], 3, H, dk)
        q, k_, v = (t[:, :, i].transpose(1, 2) for i in range(3))   # (B*, H, n_a, dk)
        o = F.scaled_dot_product_attention(q, k_, v)        # softmax(q k^T / sqrt(dk)) v, as Keras
        o = o.transpose(1, 2).reshape(*tshp[:-1], H * dk).movedim(-2, 1 + a)
        osum = o if osum is None else osum + o
    y = osum @ m.out_kernel.reshape(H * dk, d).to(x.dtype) + float(K) * m.out_bias.to(x.dtype)
    return x + self.gate * (y / float(K))


def _no_compile_fft():
    """The spectral filter uses complex FFTs, which TorchInductor does not code-generate: keep it eager (graph break)."""
    from .model import banks as _banks
    if not getattr(_banks.spectral_filter, "_eager", False):
        f = torch.compiler.disable(_banks.spectral_filter)
        f._eager = True
        _banks.spectral_filter = f


def freeze_fast_weights(model, roles, K=2, dtype=None):
    """Precompute the fused conv weights and constants for a fixed role layout (no attribute writes inside forward,
    as torch.compile(fullgraph=True) requires)."""
    for mod in model.modules():
        if isinstance(mod, AxialConv):
            W = mod.conv_kernel.detach()
            p = mod.k // 2
            blocks = [W[p] * float(K)] + [W[j] for a in range(K) for j in range(mod.k) if j != p]
            fw = torch.cat(blocks, 0)
            const = sum(mod.role[roles[a] % MAX_ROLES] for a in range(K)).detach() @ W.sum(0) + float(K) * mod.conv_bias.detach()
            if dtype is not None:
                fw, const = fw.to(dtype), const.to(dtype)
            mod._fast_W_frozen, mod._fast_const = fw, const


def accelerate(model):
    """Swap the forward of every AxialConv / AxialAttention of `model` for the fused version (in place)."""
    n = 0
    for mod in model.modules():
        if isinstance(mod, AxialConv):
            mod.forward = types.MethodType(_conv_fast, mod)
            n += 1
        elif isinstance(mod, AxialAttention):
            mod.forward = types.MethodType(_attn_fast, mod)
            n += 1
    _no_compile_fft()
    model._accelerated = True
    return n


# ---------------------------------------------------------------------------------------------------------------
# 11-07 (bench9): the spectral filter of the mechanism bank as real DFT matrix products. The eager complex FFT
# (_no_compile_fft) splits a compiled bank into ~76 graphs (70 graph breaks) and makes it 4-33x SLOWER than eager;
# with the filter below the bank compiles as ONE graph (fullgraph=True). Same operator as
# irfft2(rfft2(x) * g, s=dims) for K == 2, including the c2r treatment of the DC / Nyquist columns
# (rel. difference 3e-7 in float32, 3e-4 with TF32 matrix products).
_DFT = {}


def _dft_mats(H, W, device):
    key = (H, W, str(device))
    if key not in _DFT:
        import numpy as np
        Wh = W // 2 + 1
        n = torch.arange(W, dtype=torch.float64)
        k = torch.arange(Wh, dtype=torch.float64)
        ang = 2 * np.pi * n[:, None] * k[None, :] / W                     # (W, Wh)
        Cw, Sw = torch.cos(ang), torch.sin(ang)
        wk = torch.full((Wh,), 2.0, dtype=torch.float64)
        wk[0] = 1.0
        if W % 2 == 0:
            wk[-1] = 1.0
        Ci, Si = (wk[:, None] * Cw.T) / W, (wk[:, None] * Sw.T) / W      # (Wh, W): c2r along W
        m = torch.arange(H, dtype=torch.float64)
        angh = 2 * np.pi * m[:, None] * m[None, :] / H
        Ch, Sh = torch.cos(angh), torch.sin(angh)                          # (H, H), symmetric
        _DFT[key] = tuple(t.float().to(device) for t in (Cw, Sw, Ci, Si, Ch, Sh))
    return _DFT[key]


def spectral_filter_dft(f, g, K):
    """f (B,H,W,8) real, g (H,W//2+1,8) real transfer -> irfft2(rfft2(f) * g) (B,H,W,8), float32, compilable."""
    if K != 2:
        from .model.banks import _spectral_filter_fft
        return _spectral_filter_fft(f, g, K)
    with torch.autocast(f.device.type, enabled=False):              # the transform stays float32 (as TF)
        return _dft_filter_f32(f, g)


def _dft_filter_f32(f, g):
    H, W = f.shape[1], f.shape[2]
    Cw, Sw, Ci, Si, Ch, Sh = _dft_mats(H, W, f.device)
    x = f.float().permute(0, 3, 1, 2)                                      # (B,8,H,W)
    Xr, Xi = x @ Cw, -(x @ Sw)                                             # rfft along W
    hm = lambda M, X: torch.einsum("km,bcmw->bckw", M, X)
    Ar, Ai = hm(Ch, Xr) + hm(Sh, Xi), hm(Ch, Xi) - hm(Sh, Xr)              # fft along H
    gt = g.float().permute(2, 0, 1)[None]
    Zr, Zi = Ar * gt, Ai * gt
    Yr, Yi = (hm(Ch, Zr) - hm(Sh, Zi)) / H, (hm(Ch, Zi) + hm(Sh, Zr)) / H  # ifft along H
    return (Yr @ Ci - Yi @ Si).permute(0, 2, 3, 1)                         # c2r along W


def use_dft_spectral_filter():
    """Route the bank's spectral filter through spectral_filter_dft (replaces the eager-FFT graph break)."""
    from .model import banks as _banks
    if not hasattr(_banks, "_spectral_filter_fft"):
        _banks._spectral_filter_fft = getattr(_banks.spectral_filter, "__wrapped__", _banks.spectral_filter)
    _banks.spectral_filter = spectral_filter_dft



# ---------------------------------------------------------------------------------------------------------------
# 11-08: one entry point for the accelerated execution used in the response timings (bench_fast.py BENCH9,
# bench_ivp.py). mode="f32": float32 weights and activations (TF32 matrix products), everything compiled -- the
# reference inference setting (as the TensorFlow/XLA timings); mode="infer_bf16": the axial core and the bank run
# with bf16 copies of their weights (frozen fused conv weights) -- faster, but the error grows over long
# receding-horizon rollouts (MOTION-IVP); mode="train": float32 master weights, the fused conv weight is rebuilt
# inside the forward (gradients reach the original kernels), call the model under torch.autocast("cuda",
# torch.bfloat16); the encoder stays eager (its compiled graph fails under autocast in torch 2.4/2.5).
# The bank is ONE graph (spectral filter as DFT products). Fixed role layout and latent box per compiled model.
class _CoreFn(torch.nn.Module):
    def __init__(self, m, roles):
        super().__init__()
        self.convs, self.attns, self.mlps, self.cond, self.ln_out = m.convs, m.attns, m.mlps, m.cond, m.ln_out
        self.roles, self.depth = list(roles), m.depth

    def forward(self, h, e):
        for i in range(self.depth):
            h = self.convs[i](h, self.roles)
            h = self.attns[i](h, self.roles)
            h = self.mlps[i](h)
            h = self.cond.modulate(i, e, h)
        return self.ln_out(h)


def _freeze_kspace(bk, box_dims, device):
    names = {}
    for comps in (False, True):
        t = bk._kfeat(list(box_dims), comps, device).float()
        nm = f"_kc_{int(comps)}"
        bk.register_buffer(nm, t.clone(), persistent=False)
        names[(tuple(box_dims), comps)] = nm

    def _kfeat(dims, comps, device_):
        return getattr(bk, names[(tuple(dims), comps)]).to(bk.base.kernel.dtype)
    bk._kfeat = _kfeat


def compile_accelerated(model, roles, box_dims, mode="f32", K=2, encdec=True):
    """In place: accelerate(model) + compiled core / bank / encoder / decoder (see the block comment above)."""
    import copy
    if not getattr(model, "_accelerated", False):
        accelerate(model)
    use_dft_spectral_filter()
    import torch._dynamo as _dy
    _dy.config.cache_size_limit = max(_dy.config.cache_size_limit, 64)
    dev = next(model.parameters()).device
    assert mode in ("f32", "infer_bf16", "train"), mode
    if mode == "infer_bf16":
        cw = copy.deepcopy(_CoreFn(model, roles)).to(torch.bfloat16)
        freeze_fast_weights(cw, roles, K=K, dtype=torch.bfloat16)
        comp = torch.compile(cw, dynamic=False, fullgraph=True)
        model.core = lambda h, e, r: comp(h.to(torch.bfloat16), e.to(torch.bfloat16)).to(h.dtype)
        bk = copy.deepcopy(model.banks)
        _freeze_kspace(bk, box_dims, dev)
        bk.to(torch.bfloat16)
        inner = bk.forward

        def fwd(h, **kw):
            kw = {k: (v.to(torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in kw.items()}
            return inner(h.to(torch.bfloat16), **kw).float()
        bk.forward = fwd
        object.__setattr__(model, "banks", torch.compile(bk, dynamic=False, fullgraph=True))
    else:
        comp = torch.compile(_CoreFn(model, roles), dynamic=False, fullgraph=True)
        model.core = lambda h, e, r: comp(h, e)
        _freeze_kspace(model.banks, box_dims, dev)
        object.__setattr__(model, "banks", torch.compile(model.banks, dynamic=False, fullgraph=True))
    if encdec:
        if mode != "train":
            object.__setattr__(model, "enc", torch.compile(model.enc, dynamic=False))
        object.__setattr__(model, "dec", torch.compile(model.dec, dynamic=False))
    if mode == "f32":
        model.eval()
    return model
