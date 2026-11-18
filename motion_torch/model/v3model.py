"""V3Model (motion_tf/v3/model.py) in PyTorch: one AR step (`step`), single-shot (`step_ss`, optional `leads`), trunk.

  window at nodes -> BinEncoder (cell scatter-mean) [+ sdf gate] -> FiLM(e)-modulated axial core (depth blocks of
  AxialConv / AxialAttention / PointwiseMLP) -> TendencyBanks -> W panels (d_w x n_p)
  AR:  z = mean_p W_p, u_next(q) = u_prev(q) + D(interp(z, q)) (+ gated backward-warp transport on dense 2-D grids)
  SS:  z(k/kf) = exact partial-panel anti-derivative, frame k = u_prev + D(z_k) (+ MOTION_SS_WARP carrier)

Channels-last layout (B, *dims, C) exactly as the reference. Same code paths as the released configurations
(axial core, gate_cond=1, fam_head=1, dict_decode=0, n_slices=0, d_shape=0, moe_experts=0, transport=True,
MOTION_WARP_MODE=blend).
"""
from __future__ import annotations

import itertools

import numpy as np
import torch

from ..config import MotionConfig
from .banks import TendencyBanks
from .encdec import BinEncoder, CondEncoder, ContinuousDecoder
from .layers import AxialAttention, AxialConv, Dense, PointwiseMLP, RankFreeLN, TFModule, normal

ADV_OP = 1


def interp_box(box, q, dims):
    """Multilinear interpolation of box (B,*dims,C) at q (B,Q,K) in [0,1]^K, cell centres (i+0.5)/n, clamped edges."""
    K = len(dims)
    c = [q[..., a] * dims[a] - 0.5 for a in range(K)]
    i0f = [torch.floor(x) for x in c]
    fr = [x - f for x, f in zip(c, i0f)]
    i0 = [f.long() for f in i0f]
    B = q.shape[0]
    flat = box.reshape(B, -1, box.shape[-1])
    strides = [int(np.prod(dims[a + 1:])) for a in range(K)]
    bidx = torch.arange(B, device=q.device)[:, None]
    out = 0.0
    for off in itertools.product((0, 1), repeat=K):
        idx = 0
        w = 1.0
        for a in range(K):
            ia = torch.clamp(i0[a] + off[a], 0, dims[a] - 1)
            idx = idx + ia * strides[a]
            w = w * (fr[a] if off[a] else (1.0 - fr[a]))
        out = out + w[..., None].to(flat.dtype) * flat[bidx, idx]
    return out


def to_grid(x, grid_dims, S):
    """(B,N,S) node values -> (B,*grid_dims,S) (v3/train._grid)."""
    return x.reshape(x.shape[0], *grid_dims, S)


class V3Model(TFModule):
    def __init__(self, cfg: MotionConfig):
        super().__init__()
        self.cfg = cfg
        d, S = cfg.d, cfg.S
        self.S, self.d, self.depth = S, d, cfg.depth
        self.n_semantic = cfg.n_semantic
        self.d_w, self.n_p, self.t_in = cfg.d_w, cfg.n_p, cfg.t_in
        self.box_dims = {2: tuple(cfg.dims2), 3: tuple(cfg.dims3)}
        self.transport = bool(cfg.transport)
        self.enc = BinEncoder(d, cfg.t_in * S + cfg.t_in + 8, name="enc")
        self.cond = CondEncoder(cfg.cond_dim, cfg.d_cond, d, cfg.depth, name="cond")
        self.convs = torch.nn.ModuleList([AxialConv(d, f"b{i}_conv") for i in range(cfg.depth)])
        self.attns = torch.nn.ModuleList([AxialAttention(d, f"b{i}_attn") for i in range(cfg.depth)])
        self.mlps = torch.nn.ModuleList([PointwiseMLP(d, f"b{i}_mlp") for i in range(cfg.depth)])
        self.ln_out = RankFreeLN(d, "ln_out")
        self.banks = TendencyBanks(cfg.d_w * cfg.n_p, d, cfg, name="banks")
        self.dec = ContinuousDecoder(cfg.d_w, S, cfg.dec_hidden, cfg.n_fams, cfg.d_cond, name="dec")
        self.sdf_proj = Dense(1, d, "sdf_proj", act="gelu")
        self.sdf_gate = Dense(d, d, "sdf_gate", bias=False, init="zeros")
        if self.transport:
            self.disp = Dense(d, 2, "disp", init=(normal(1e-3) if cfg.warp_live else "zeros"))
            self.tgate = Dense(cfg.d_cond, S, "tgate", bias=False, init="zeros")
        self.ss_warp = cfg.ss_warp_on
        self.ss_warp_mode = cfg.ss_warp_mode
        if self.ss_warp and self.ss_warp_mode == "taylor":
            self.ss_wv = Dense(d, 2, "ss_warp_v", init="zeros")
            self.ss_wa = Dense(d, 2, "ss_warp_a", init="zeros")
        self.warp_scale = None              # eval-only runtime warp_off (eval_stats.py "warp_off")
        self.z_scale = None

    # ------------------------------------------------------------------ pieces
    def feats_from_win(self, win, fmask, geom_node):
        B, T, N, S = win.shape
        x = win.permute(0, 2, 1, 3).reshape(B, N, self.t_in * self.S)
        fm = fmask[:, None, :].expand(B, N, fmask.shape[-1])
        return torch.cat([x, fm, geom_node], -1)

    def core(self, h, e, roles):
        for i in range(self.depth):
            h = self.convs[i](h, roles)
            h = self.attns[i](h, roles)
            h = self.mlps[i](h)
            h = self.cond.modulate(i, e, h)
        return self.ln_out(h)

    def trunk(self, coords_node, win, fmask, geom_node, K, roles, cond, op_multihot, sdf_box=None, box_dims=None,
              steady=False):
        """window -> (h latent box, W tendencies (B,*dims,d_w*n_p), e)."""
        dims = tuple(box_dims) if box_dims else self.box_dims[K]
        feats = self.feats_from_win(win, fmask, geom_node)
        box, geom_box = self.enc(coords_node, feats, dims, roles)
        if sdf_box is not None:
            box = box + self.sdf_gate(self.sdf_proj(sdf_box))
            geom_box = torch.cat([sdf_box.to(geom_box.dtype), geom_box[..., 1:]], -1)
        e = self.cond(cond)
        h = self.core(box, e, roles)
        W = self.banks(h, K=K, roles=roles, op_multihot=op_multihot, geom_box=geom_box, steady=steady, e=e)
        return h, W, e, dims

    def _keep(self, x):
        keep = torch.cat([torch.ones(self.n_semantic, dtype=x.dtype, device=x.device),
                          torch.zeros(self.S - self.n_semantic, dtype=x.dtype, device=x.device)], 0)
        return x * keep[None, None, :]

    # ------------------------------------------------------------------ AR step
    def step(self, coords_node, win, fmask, geom_node, K, roles, cond, op_multihot, coords_q, geom_q, u_prev_q,
             steady=False, prev_grid=None, grid_dims=None, fam_id=None, sdf_box=None, box_dims=None, fbmask=None,
             prev2_grid=None, return_aux=False):
        """ONE AR step: next frame at coords_q (B,Q,S). prev_grid (B,H,W,S) for dense-2-D transport, else None.
        prev2_grid is accepted for API compatibility (only the dictionary decode, off here, reads it)."""
        roles = list(roles)
        h, W, e, dims = self.trunk(coords_node, win, fmask, geom_node, K, roles, cond, op_multihot, sdf_box,
                                   box_dims, steady)
        Wp = W.reshape(*W.shape[:-1], self.d_w, self.n_p)
        z = Wp.mean(-1)
        if self.z_scale is not None:
            z = z * self.z_scale
        zq = interp_box(z, coords_q, dims)
        delta = self.dec(zq, coords_q, roles, geom_q, e, fam_id)
        d = delta
        if fbmask is not None:
            d = d * fbmask[:, None, :]
        if self.n_semantic < self.S:
            u_prev_q = self._keep(u_prev_q)
        u_next = u_prev_q + d
        if self.transport and prev_grid is not None and K == 2:
            db = self.disp(h)
            dq = interp_box(db, coords_q, dims)
            warped = interp_box(prev_grid, coords_q - dq, grid_dims)
            g = torch.tanh(self.tgate(e))[:, None, :]
            g = g * op_multihot[:, None, ADV_OP:ADV_OP + 1]
            if self.warp_scale is not None:
                g = g * self.warp_scale
            u_next = u_next + g * (warped - u_prev_q)
        if return_aux:
            return u_next, {"h": h, "W": W, "e": e, "z": z, "zq": zq, "delta": delta}
        return u_next

    # ------------------------------------------------------------------ single shot
    def step_ss(self, coords_node, win, fmask, geom_node, K, roles, cond, op_multihot, coords_q, geom_q, u_prev_q,
                kf, fam_id=None, sdf_box=None, box_dims=None, prev_grid=None, grid_dims=None, leads=None):
        """Single-shot: one trunk call, frames k = 1..kf (or `leads`) from the exact partial-panel anti-derivative.
        Returns (B, len(leads or 1..kf), Q, S)."""
        roles = list(roles)
        h, W, e, dims = self.trunk(coords_node, win, fmask, geom_node, K, roles, cond, op_multihot, sdf_box,
                                   box_dims, False)
        Wq = interp_box(W, coords_q, dims)
        Wq = Wq.reshape(Wq.shape[0], -1, self.d_w, self.n_p)
        cum = torch.cumsum(Wq, -1) / float(self.n_p)
        if self.n_semantic < self.S:
            u_prev_q = self._keep(u_prev_q)
        warp = self.ss_warp and prev_grid is not None and K == 2
        if warp:
            if self.ss_warp_mode == "blend":
                vq = interp_box(self.disp(h), coords_q, dims)
                aq = None
                gw = torch.tanh(self.tgate(e))[:, None, :]
                gw = gw * op_multihot[:, None, ADV_OP:ADV_OP + 1]
                if self.warp_scale is not None:
                    gw = gw * self.warp_scale
            else:
                vq = interp_box(self.ss_wv(h), coords_q, dims)
                aq = interp_box(self.ss_wa(h), coords_q, dims)
            ic0 = interp_box(prev_grid, coords_q, grid_dims)
        outs = []
        for k in (range(1, kf + 1) if leads is None else leads):
            base = u_prev_q
            if warp:
                if aq is None:
                    dq = torch.clamp(vq * float(k), -0.5, 0.5)
                else:
                    tk = k / float(kf)
                    dq = torch.clamp(vq * tk + 0.5 * aq * (tk * tk), -0.5, 0.5)
                wk = interp_box(prev_grid, coords_q - dq, grid_dims)
                if self.n_semantic < self.S:
                    wk, ic0_ = self._keep(wk), self._keep(ic0)
                else:
                    ic0_ = ic0
                base = u_prev_q + ((gw * (wk - ic0_)) if aq is None else (wk - ic0_))
            t = k / float(kf) * self.n_p
            m = int(np.ceil(t)) - 1
            f = t - m
            z_k = (f / float(self.n_p)) * Wq[..., m]
            if m > 0:
                z_k = z_k + cum[..., m - 1]
            delta = self.dec(z_k, coords_q, roles, geom_q, e, fam_id)
            outs.append(base + delta)
        return torch.stack(outs, 1)

    # ------------------------------------------------------------------ names
    def tf_name_map(self):
        """{torch parameter name: TF variable name} for every parameter (used by convert.py)."""
        out = {}
        for mname, mod in self.named_modules():
            for pn, tn in getattr(mod, "tf_names", {}).items():
                out[f"{mname}.{pn}" if mname else pn] = tn
        names = dict(self.named_parameters())
        assert set(out) == set(names), (set(names) ^ set(out))
        return out
