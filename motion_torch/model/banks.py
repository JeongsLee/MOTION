"""core/banks.TendencyBanks (typed mechanism heads + optional generic heads), PyTorch port.

Ported code paths: typed vocabulary (13 heads in `names` + bernoulli, kinematic, vortex, reaction_nl, spectral,
coefficient, compressible), H0 base head, per-example gate modulation from e (gate_cond=1), operator-token masks,
steady drop, MOTION_VORTEX2D_LEGACY, MOTION_BANK_NGENERIC (+ MOTION_BANK_GENERIC_MERGE), the eval-only runtime
knock-out mask (ko_mask/ko_index, as eval_stats.py) and the `active` subset.
Not ported (asserted off in MotionConfig.from_env): MOTION_BANK_GENERIC (V1a), MOTION_BANK_NOMETA, NOH0, SINGLE,
PHYSGATE, gate_experts, dual source. Stencils are periodic cell differences without 1/dx (REF_DIMS empty).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .layers import Dense, TFModule, gelu, normal

GRAVITY_ROLE = 2
OP_INDEX = {"reaction": 3, "diffusion": 2, "buoyancy": 7, "shear": 8, "wave": 9,
            "advection": 1, "shock": 1, "elliptic": 11, "bernoulli": 6,
            "dilatation": 4, "reaction_nl": 3, "phase_interface": 13}
STEADY_DROP = ("wave", "shock", "buoyancy")
_MAX_ROLES = 4
NAMES = ["reaction", "diffusion", "buoyancy", "shear", "wave", "advection", "shock", "elliptic", "geometry",
         "boundary_layer", "dilatation", "gradvec", "phase_interface"]


def _roll(x, s, dim):
    return torch.roll(x, shifts=s, dims=dim)


def axis_grad(x, a):
    """0.5*(x[i+1]-x[i-1]) along spatial axis a (periodic)."""
    return 0.5 * (_roll(x, -1, 1 + a) - _roll(x, 1, 1 + a))


def axis_diff_fb(x, a):
    up, dn = _roll(x, -1, 1 + a), _roll(x, 1, 1 + a)
    return 1.0 * (up - x), 1.0 * (x - dn)


def laplacian(x, K):
    acc = 0.0
    for a in range(K):
        up, dn = _roll(x, -1, 1 + a), _roll(x, 1, 1 + a)
        acc = acc + (up - 2.0 * x + dn)
    return acc


def _psum(xs):
    """python sum() order: ((0 + x0) + x1) + ..."""
    acc = 0
    for x in xs:
        acc = acc + x
    return acc


class AxStencil(TFModule):
    """Learned per-channel axial stencil (w, d) per axis role, periodic shifts."""

    def __init__(self, d, w, name):
        super().__init__()
        self.w = int(w)
        self.tf_param("k", (_MAX_ROLES, self.w, d), f"{name}_k:0", normal(self.w ** -0.5))

    def forward(self, h, K, roles):
        c = self.w // 2
        out = None
        for a in range(K):
            ker = self.k[roles[a] % _MAX_ROLES]
            s = None
            for j in range(self.w):
                t = ker[j] * _roll(h, c - j, 1 + a)
                s = t if s is None else s + t
            out = s if out is None else out + s
        return out


class LowRank(TFModule):
    def __init__(self, d, r, name, scope):
        super().__init__()
        self.a = Dense(d, r, f"{scope}/{name}/{name}_a", bias=False)
        self.b = Dense(r, d, f"{scope}/{name}/{name}_b")

    def forward(self, h):
        return self.b(self.a(h))


def _kspace(dims, comps=False):
    ax = [np.fft.fftfreq(n) * n for n in dims[:-1]] + [np.arange(dims[-1] // 2 + 1)]
    mesh = np.meshgrid(*ax, indexing="ij")
    if comps:
        c = [(m / max(n, 1)).astype("float32") for m, n in zip(mesh, dims)]
        c = c + [np.zeros_like(c[0])] * (3 - len(c))
        return np.stack(c, -1)
    kmag = np.sqrt(sum((m / max(n, 1)) ** 2 for m, n in zip(mesh, dims))).astype("float32")
    return kmag[..., None]


_FFT_CPU = {}


def _fft_device(dev):
    """cuFFT of the CUDA 11.7 torch wheels fails on sm_89 GPUs (RTX 40xx: CUFFT_INTERNAL_ERROR). Probe once per
    device; if the GPU FFT is unusable, the (small: 8 channels) spectral transforms run on the CPU instead."""
    if dev.type != "cuda":
        return dev
    key = str(dev)
    if key not in _FFT_CPU:
        try:
            torch.fft.irfft2(torch.fft.rfft2(torch.ones(1, 1, 8, 8, device=dev)), s=[8, 8])
            _FFT_CPU[key] = False
        except RuntimeError:
            _FFT_CPU[key] = True
            print(f"[motion_torch] cuFFT unusable on {key} (torch {torch.__version__}, CUDA {torch.version.cuda}): "
                  f"spectral transforms fall back to the CPU", flush=True)
    return torch.device("cpu") if _FFT_CPU[key] else dev


def spectral_filter(f, g, K):
    """f (B,*dims,8) real, g (*kdims,8) real transfer -> filtered (B,*dims,8). Same transform sequence as TF."""
    dev = f.device
    fd = _fft_device(dev)
    if fd != dev:
        return spectral_filter(f.to(fd), g.to(fd), K).to(dev)
    rank = f.dim()
    dims = list(f.shape[1:-1])
    x = f.permute(0, rank - 1, *range(1, rank - 1))                # (B,8,*dims)
    if x.dtype in (torch.float16, torch.bfloat16):                  # FFTs in float32 (as TF under mixed policy)
        x = x.float()
    g = g.to(x.dtype)
    gt = g.permute(len(dims), *range(len(dims)))                    # (8,*kdims)
    gt = torch.complex(gt, torch.zeros_like(gt))
    if K == 2:
        X = torch.fft.rfft2(x)
        y = torch.fft.irfft2(X * gt[None], s=dims)
    else:
        X = torch.fft.rfft(x, dim=-1)
        X = torch.fft.fft2(X, dim=(-3, -2))
        X = X * gt[None]
        X = torch.fft.ifft2(X, dim=(-3, -2))
        y = torch.fft.irfft(X, n=dims[-1], dim=-1)
    return y.permute(0, *range(2, rank), 1)


class TendencyBanks(TFModule):
    def __init__(self, out, d, cfg, name="banks"):
        super().__init__()
        self.out, self.d = out, d
        self.cfg = cfg
        self.n_ops = cfg.n_ops
        self.gate_cond = int(cfg.gate_cond)
        dc = cfg.d_cond
        sc = name
        mk = lambda n_in, nm, **kw: Dense(n_in, out, f"{sc}/{sc}_{nm}", **kw)
        self.base = mk(d, "base", init=normal(1e-3))
        # typed heads; 'phase_interface' only exists when its token is in the vocabulary (Keras builds lazily)
        self.names = [m for m in NAMES if not (m in OP_INDEX and OP_INDEX[m] >= self.n_ops)]
        fin = {"dilatation": d + 1, "gradvec": 3 * d}
        self.heads = torch.nn.ModuleDict({m: mk(fin.get(m, d), m, init=normal(1e-3)) for m in self.names})
        rh = 2 * out if out < 128 else out
        self.reaction_hidden = Dense(d, rh, f"{sc}/{sc}_reaction_h", act="gelu")
        self.spec_kmlp = Dense(1, 16, f"{sc}/{sc}_spec_k", act="gelu")
        self.spec_kout = Dense(16, 8, f"{sc}/{sc}_spec_ko")
        self.spec_in = Dense(d, 8, f"{sc}/{sc}_spec_in")
        self.spec_head = mk(8, "spectral")
        self.coef_mlp = Dense(d, 8, f"{sc}/{sc}_coef_c", act="gelu")
        self.coef_lproj = Dense(d, 8, f"{sc}/{sc}_coef_l")
        self.coef_head = mk(16, "coefop")
        self.cmp_w = Dense(d, 1, f"{sc}/{sc}_cmp_w")
        self.cmp_head = mk(4 * d + 4, "compress")
        self.bern_head = mk(3, "bernoulli", init=normal(1e-3))
        self.kin_head = mk(3, "kinematic", init=normal(1e-3))
        self.vel = Dense(d, 3, f"{sc}/{sc}_vel")
        self.reaction_nl_head = mk(rh, "reaction_nl")
        self.vs_head = mk(3, "vortex")
        self.n_generic = int(cfg.bank_ngeneric)
        self.gen_merge = bool(cfg.bank_generic_merge) and self.n_generic > 0
        self._gen = (["generic"] if self.gen_merge else [f"generic{i}" for i in range(self.n_generic)])
        if self.n_generic:
            G = self.n_generic
            self.gen_S = torch.nn.ModuleList([AxStencil(d, 3, f"{sc}_genS{i}") for i in range(G)])
            self.gen_L = torch.nn.ModuleList([LowRank(d, 8, f"{sc}_genL{i}", sc) for i in range(G)])
            self.gen_si = torch.nn.ModuleList([Dense(d, 8, f"{sc}/{sc}_genSI{i}") for i in range(G)])
            self.gen_sk = torch.nn.ModuleList([Dense(3, 16, f"{sc}/{sc}_genSK{i}", act="gelu") for i in range(G)])
            self.gen_so = torch.nn.ModuleList([Dense(16, 8, f"{sc}/{sc}_genSO{i}") for i in range(G)])
            self.gen_head = torch.nn.ModuleList([
                mk((G if self.gen_merge else 1) * (d + 8), f"generic{i}", init=normal(1e-3))
                for i in range(1 if self.gen_merge else G)])
        self._gatenames = ([m for m in NAMES] + ["bernoulli", "kinematic", "spectral", "reaction_nl",
                                                 "coefficient", "compressible", "vortex"] + self._gen)
        for m in self._gatenames:                     # gate vectors exist for every name (incl. phase_interface)
            self.tf_param(f"gate_{m}", (out,), f"{sc}_gate_{m}:0", "zeros")
        if self.gate_cond:
            self.gcond = torch.nn.ModuleDict({m: Dense(dc, out, f"{sc}/{sc}_gcond_{m}", bias=False, init="zeros")
                                              for m in self._gatenames if m in self.names or m not in NAMES})
        # eval-only runtime switches (eval_stats.py): ko_mask (one entry per head, order ko_index) and `active`
        self.ko_index = None
        self.ko_mask = None
        self.active = None
        self.probe = None
        self._kcache = {}

    def all_heads(self):
        """eval_stats.ALL_HEADS order (knock-out index)."""
        return list(NAMES) + list(self._gen) + ["bernoulli", "kinematic", "vortex", "reaction_nl", "spectral",
                                                  "coefficient", "compressible"]

    # ------------------------------------------------------------------ helpers
    def _kfeat(self, dims, comps, device):
        key = (tuple(dims), comps, str(device))
        if key not in self._kcache:
            self._kcache[key] = torch.from_numpy(_kspace(dims, comps)).to(device)
        return self._kcache[key].to(self.base.kernel.dtype)

    def _gate(self, m, ce, rank):
        g = getattr(self, f"gate_{m}")
        if ce is not None:
            g = g + self.gcond[m](ce).reshape(-1, *([1] * (rank - 2)), self.out)
        return g

    def _ops(self, h, K, u, geom_box):
        gx = axis_grad(h, 0)
        gz = axis_grad(h, min(GRAVITY_ROLE, K - 1)) if K >= 2 else gx
        lap = laplacian(h, K)
        adv = 0.0
        shock = 0.0
        for a in range(K):
            ua = u[..., a:a + 1]
            adv = adv - ua * axis_grad(h, a)
            fwd, bwd = axis_diff_fb(h, a)
            shock = shock - ua * torch.where(ua > 0.0, bwd, fwd)
        rank = h.dim()
        ell = h.mean(dim=tuple(range(1, rank - 1)), keepdim=True) + 0.0 * h
        gmag = torch.sqrt(_psum(axis_grad(h, a) ** 2 for a in range(K)) + 1e-6)
        kappa = 0.0
        for a in range(K):
            kappa = kappa + axis_grad(axis_grad(h, a) / gmag, a)
        if geom_box is not None:
            sdf = geom_box[..., 0:1]
            ng = [axis_grad(sdf, a) for a in range(K)]
            nmag = torch.sqrt(_psum(g * g for g in ng) + 1e-6)
            dn_h = _psum((ng[a] / nmag) * axis_grad(h, a) for a in range(K))
            dn2 = _psum((ng[a] / nmag) * axis_grad(dn_h, a) for a in range(K))
            bl = torch.exp(-torch.abs(sdf) / 0.15) * (dn_h + dn2)
        else:
            bl = torch.zeros_like(h)
        divu = 0.0
        for a in range(K):
            divu = divu + axis_grad(u[..., a:a + 1], a)
        dil = torch.cat([divu, h * divu], -1)
        gv = [torch.zeros_like(h) for _ in range(3)]
        for a in range(K):
            gv[a if a < 3 else 2] = axis_grad(h, a)
        gradvec = torch.cat(gv, -1)
        gmag2 = torch.sqrt(gx * gx + gz * gz + 1e-12)
        return {"phase_interface": gmag2 * lap,
                "reaction": h, "diffusion": lap, "buoyancy": gz, "shear": gx, "wave": lap,
                "advection": adv, "shock": shock, "elliptic": ell, "geometry": kappa,
                "boundary_layer": bl, "dilatation": dil, "gradvec": gradvec}

    def _spectral_feat(self, h, K):
        f = self.spec_in(h)
        dims = list(h.shape[1:-1])
        g = self.spec_kout(self.spec_kmlp(self._kfeat(dims, False, h.device)))
        return spectral_filter(f, g, K).to(f.dtype)

    def _generic_spec(self, h, K, i):
        f = self.gen_si[i](h)
        dims = list(h.shape[1:-1])
        g = self.gen_so[i](self.gen_sk[i](self._kfeat(dims, True, h.device)))
        return spectral_filter(f, g, K).to(f.dtype)

    @staticmethod
    def _vortex_feat(u, K):
        z = torch.zeros_like(u[..., :1])

        def g(i, j):
            return axis_grad(u[..., i:i + 1], j) if (i < K and j < K) else z

        om = [g(2, 1) - g(1, 2), g(0, 2) - g(2, 0), g(1, 0) - g(0, 1)]
        return torch.cat([_psum(om[a] * g(i, a) for a in range(3)) for i in range(3)], -1)

    @staticmethod
    def _kin_feat(u, K):
        G = [[axis_grad(u[..., i:i + 1], j) for j in range(K)] for i in range(K)]
        S2 = _psum((0.5 * (G[i][j] + G[j][i])) ** 2 for i in range(K) for j in range(K))
        Om2 = _psum((0.5 * (G[i][j] - G[j][i])) ** 2 for i in range(K) for j in range(K))
        return torch.cat([S2, Om2, 0.5 * (Om2 - S2)], -1)

    def _rec(self, m, c):
        if self.ko_mask is not None and self.ko_index is not None and m in self.ko_index:
            c = c * self.ko_mask[self.ko_index[m]].to(c.dtype)
        if self.probe is not None:
            self.probe[m] = float(c.detach().abs().mean())
        return c

    def _on(self, m):
        return self.active is None or m in self.active

    # ------------------------------------------------------------------ forward
    def forward(self, h, K, roles, op_multihot, geom_box, steady=False, e=None):
        drop = set(STEADY_DROP) if steady else set()
        rank = h.dim()
        bshape = (-1,) + (1,) * (rank - 1)
        u = self.vel(h)
        feats = self._ops(h, K, u, geom_box)
        ce = e if self.gate_cond else None
        gate = lambda m: self._gate(m, ce, rank)
        W = self._rec("H0", self.base(h))
        for m in NAMES:
            if not self._on(m) or m in drop:
                continue
            if op_multihot is not None and m in OP_INDEX and OP_INDEX[m] >= int(op_multihot.shape[-1]):
                continue
            contrib = gate(m) * self.heads[m](feats[m])
            if op_multihot is not None and m in OP_INDEX:
                contrib = contrib * op_multihot[:, OP_INDEX[m]].reshape(bshape)
            W = W + self._rec(m, contrib)
        if self._on("bernoulli"):
            bern = gate("bernoulli") * self.bern_head(-0.5 * u * u)
            if op_multihot is not None:
                bern = bern * op_multihot[:, OP_INDEX["bernoulli"]].reshape(bshape)
            W = W + self._rec("bernoulli", bern)
        if self._on("kinematic"):
            W = W + self._rec("kinematic", gate("kinematic") * self.kin_head(self._kin_feat(u, K)))
        if self._on("vortex") and (K == 3 or self.cfg.vortex2d_legacy):
            W = W + self._rec("vortex", gate("vortex") * self.vs_head(self._vortex_feat(u, K)))
        if self._on("reaction_nl"):
            rnl = gate("reaction_nl") * self.reaction_nl_head(self.reaction_hidden(h))
            if op_multihot is not None:
                rnl = rnl * op_multihot[:, OP_INDEX["reaction_nl"]].reshape(bshape)
            W = W + self._rec("reaction_nl", rnl)
        if self._on("spectral"):
            W = W + self._rec("spectral", gate("spectral") * self.spec_head(self._spectral_feat(h, K)))
        if self._on("coefficient"):
            cf = self.coef_mlp(h)
            feat = torch.cat([cf * self.coef_lproj(feats["diffusion"]), cf], -1)
            W = W + self._rec("coefficient", gate("coefficient") * self.coef_head(feat))
        if self._on("compressible"):
            w = F.softplus(self.cmp_w(h))
            divu = 0.0
            for a in range(K):
                divu = divu + axis_grad(u[..., a:a + 1], a)
            cfeat = torch.cat([w * feats["gradvec"], w * feats["diffusion"], u * divu, w], -1)
            W = W + self._rec("compressible", gate("compressible") * self.cmp_head(cfeat))
        if self.gen_merge and self._on("generic"):
            gfeat = torch.cat([torch.cat([self.gen_S[i](h, K, roles) * self.gen_L[i](h), self._generic_spec(h, K, i)],
                                         -1) for i in range(self.n_generic)], -1)
            W = W + self._rec("generic", gate("generic") * self.gen_head[0](gfeat))
        for i, m in enumerate([] if self.gen_merge else self._gen):
            if not self._on(m):
                continue
            gfeat = torch.cat([self.gen_S[i](h, K, roles) * self.gen_L[i](h), self._generic_spec(h, K, i)], -1)
            W = W + self._rec(m, gate(m) * self.gen_head[i](gfeat))
        return W
