"""Encoder (core/binenc.BinEncoder, n_slices=0), conditioning (v3/cond.CondEncoder) and decoder
(core/pointio.ContinuousDecoder: bias-free MLP, coord-Fourier / geometry / cond FiLM, unshared per-family head).
Paths not used by the released configurations (slice attention, dictionary decode, MOTION_STD_DECODER, shared head)
are not ported and are asserted off."""
from __future__ import annotations

import numpy as np
import torch

from .layers import MAX_ROLES, CoordEncoder, Dense, TFModule, fourier_freqs

GEOM_DIM = 8


class BinEncoder(TFModule):
    def __init__(self, d, n_feat, name="enc"):
        super().__init__()
        self.d = d
        self.coord = CoordEncoder(d, f"{name}_coord", scope=name)
        self.fproj = Dense(n_feat, d, f"{name}/{name}_f")
        self.out = Dense(d + 1, d, f"{name}/{name}_o")
        self.geom_mlp = Dense(GEOM_DIM, d, f"{name}/{name}_gm", act="gelu")
        self.geom_gate = Dense(d, d, f"{name}/{name}_gg", bias=False, init="zeros")

    def forward(self, coords, feats, dims, roles):
        """coords (B,N,K) in [0,1]^K, feats (B,N,F) -> box (B,*dims,d), geom_box (B,*dims,8). Scatter-mean per cell."""
        B, N, K = coords.shape
        ncell = int(np.prod(dims))
        tok = self.coord(coords, roles) + self.fproj(feats)
        tok = tok + self.geom_gate(self.geom_mlp(feats[..., -GEOM_DIM:]))
        idx = torch.zeros((B, N), dtype=torch.long, device=coords.device)
        stride = 1
        for a in reversed(range(K)):
            ci = torch.floor(coords[..., a] * dims[a]).long().clamp(0, dims[a] - 1)
            idx = idx + ci * stride
            stride *= dims[a]
        boff = torch.arange(B, device=coords.device)[:, None] * ncell
        flat_idx = (idx + boff).reshape(-1)
        flat_tok = tok.reshape(-1, self.d)
        summ = torch.zeros((B * ncell, self.d), dtype=tok.dtype, device=tok.device).index_add_(0, flat_idx, flat_tok)
        cnt = torch.zeros((B * ncell, 1), dtype=tok.dtype, device=tok.device).index_add_(
            0, flat_idx, torch.ones((flat_idx.shape[0], 1), dtype=tok.dtype, device=tok.device))
        mean = summ / torch.clamp(cnt, min=1.0)
        cov = torch.log1p(cnt)
        h = self.out(torch.cat([mean, cov], -1))
        g = feats[..., -GEOM_DIM:].reshape(-1, GEOM_DIM)
        gsum = torch.zeros((B * ncell, GEOM_DIM), dtype=g.dtype, device=g.device).index_add_(0, flat_idx, g)
        geom_box = (gsum / torch.clamp(cnt, min=1.0)).reshape(B, *dims, GEOM_DIM)
        return h.reshape(B, *dims, self.d), geom_box


class CondEncoder(TFModule):
    """cond (B, COND_DIM) -> e (B, d_cond); per-block FiLM. TF names: 'cond/cond_h1', but 'cond_film{i}' unscoped
    (Keras builds the FiLM Dense lazily outside the layer's call scope)."""

    def __init__(self, cond_dim, d_cond, d_core, n_blocks, name="cond"):
        super().__init__()
        self.h1 = Dense(cond_dim, d_cond, f"{name}/{name}_h1", act="gelu")
        self.h2 = Dense(d_cond, d_cond, f"{name}/{name}_h2", act="gelu")
        self.film = torch.nn.ModuleList([Dense(d_cond, 2 * d_core, f"{name}_film{i}", init="zeros")
                                         for i in range(n_blocks)])

    def forward(self, cond):
        return self.h2(self.h1(cond))

    def modulate(self, i, e, h):
        gb = self.film[i](e)
        d = h.shape[-1]
        gb = gb.reshape(gb.shape[0], *([1] * (h.dim() - 2)), 2 * d)
        return h * (1.0 + gb[..., :d]) + gb[..., d:]


class ContinuousDecoder(TFModule):
    """fam_only=True decode (fam_head=1): out = (FiLM-modulated gelu(z W_h)) @ famW[fam_id]. Bias-free -> D(0) = 0."""

    def __init__(self, d_w, d_out, hidden, n_fams, d_cond, n_freq=16, name="dec"):
        super().__init__()
        assert n_fams > 0
        self.hidden, self.d_out, self.n_fams = hidden, d_out, n_fams
        self.h = Dense(d_w, hidden, f"{name}/{name}_h", bias=False, act="gelu")
        self.cond_gate = Dense(d_cond, hidden, f"{name}/{name}_cgate", bias=False, init="zeros")
        self.tf_param("fam_table", (n_fams, hidden * d_out), f"{name}_famW:0",
                      lambda s: torch.randn(s) * hidden ** -0.5)
        self.register_buffer("ff_freq", fourier_freqs(n_freq), persistent=False)
        self.tf_param("ff_role", (MAX_ROLES, 2 * n_freq), f"{name}_ffrole:0", "zeros")
        self.ff_gate = Dense(2 * n_freq, hidden, f"{name}/{name}_ffgate", bias=False, init="zeros")
        self.geom_feat = Dense(GEOM_DIM, hidden, f"{name}/{name}_gf", act="gelu")
        self.geom_gate = Dense(hidden, hidden, f"{name}/{name}_gg", bias=False, init="zeros")

    def _fourier(self, coords, roles):
        acc = None
        for a in range(coords.shape[-1]):
            ph = coords[..., a:a + 1].float() * self.ff_freq.float()   # f32 phase, as TF
            ff = (torch.cat([torch.sin(ph), torch.cos(ph)], -1).to(self.ff_role.dtype)
                  + self.ff_role[roles[a] % MAX_ROLES])
            acc = ff if acc is None else acc + ff
        return acc

    def forward(self, z, coords, roles, geom, cond, fam_id):
        hid = self.h(z)
        hid = hid * (1.0 + self.ff_gate(self._fourier(coords, roles)))
        hid = hid * (1.0 + self.geom_gate(self.geom_feat(geom)))
        hid = hid * (1.0 + self.cond_gate(cond)[:, None, :])
        Wf = self.fam_table[fam_id].reshape(-1, self.hidden, self.d_out)
        return torch.einsum("bqh,bho->bqo", hid, Wf)
