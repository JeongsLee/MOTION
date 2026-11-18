"""GPU timing of MOTION-NS inference and training: reference port vs the accelerated trunk (fast.py) (revision R1).

  python -m motion_torch.bench_fast --ckpt zsc_ckpt_160000.npz [--batch 8] [--fam incom_ns] [--n 128]

Inference = one full forecast (ten autoregressive frames at 128^2, all grid nodes), batch 8, after warm-up, as the
cost table of Supplementary Note 15 (excluding data preparation). Training = forward + backward of one ten-frame
rollout segment (finetune.rollout_loss_backward), batch 8. Each configuration is checked against the float32
reference on the same batch (relative difference of the forecast). FLOPs are counted once with
torch.utils.flop_counter (reference configuration).
"""
import argparse
import os
import copy
import json
import time

import numpy as np
import torch

from . import PRESETS, build_model
from .data.example import box_dims_for, make_example_ar, stack_batch, synthetic_sample
from .fast import accelerate
from .finetune import rollout_loss_backward
from .inference import predict, to_torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--fam", default="incom_ns")
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--out", default="bench_fast.json")
ap.add_argument("--device", default="cuda")
a = ap.parse_args()

dev = a.device
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
cfg = PRESETS["motion_ns"]
rng = np.random.default_rng(0)
exs = []
for i in range(a.batch):
    s = synthetic_sample(a.fam, rng, dims=(a.n, a.n), T=20)
    exs.append(make_example_ar(s, cfg.t_in, 10, 4, 0, a.n * a.n, rng, cfg, box_dims=box_dims_for(a.fam, cfg)))
b = to_torch(stack_batch(exs), dev)
base = build_model(cfg, a.ckpt, device=dev).eval()
fast = copy.deepcopy(base)
accelerate(fast)


def sync():
    if dev == "cuda":
        torch.cuda.synchronize()


def t_infer(model, bf16, reps):
    def run():
        with torch.no_grad(), torch.autocast(dev, torch.bfloat16, enabled=bf16):
            return predict(model, b, with_metrics=False)["pred"]
    for _ in range(3):
        out = run()
    sync()
    ts = []
    for _ in range(reps):
        sync(); t = time.perf_counter(); run(); sync(); ts.append(time.perf_counter() - t)
    return float(np.median(ts)), out.float()


def t_train(model, bf16, reps):
    model.train()
    def run():
        with torch.autocast(dev, torch.bfloat16, enabled=bf16):
            L = rollout_loss_backward(model, b, k_seg=10)
        model.zero_grad(set_to_none=True)
        return L
    for _ in range(3):
        run()
    sync()
    ts = []
    for _ in range(reps):
        sync(); t = time.perf_counter(); run(); sync(); ts.append(time.perf_counter() - t)
    model.eval()
    return float(np.median(ts))


from torch.utils.flop_counter import FlopCounterMode
fc = FlopCounterMode(display=False)
with torch.no_grad(), fc:
    predict(base, b, with_metrics=False)
flops = fc.get_total_flops()

res = {"gpu": torch.cuda.get_device_name() if dev == "cuda" else "cpu", "torch": torch.__version__, "batch": a.batch, "fam": a.fam, "n": a.n,
       "flops_per_forecast_batch": flops, "flops_per_trajectory": flops / a.batch, "cases": {}}
t_ref, p_ref = t_infer(base, False, a.reps)
cases = [("reference f32", base, False, False), ("reference bf16", base, True, False),
         ("fused f32", fast, False, False), ("fused bf16", fast, True, False)]
B6 = os.environ.get("BENCH6", "0") == "1"          # 11-06: bench6 = only the reference (inference) + the new cases
if B6 or os.environ.get("BENCH7", "0") == "1" or os.environ.get("BENCH9", "0") == "1":
    cases = []
    res["cases"]["reference f32"] = {"infer_s_per_traj": t_ref / a.batch, "train_s_per_step": None, "rel_diff_vs_ref_f32": 0.0,
                                     "tflops_per_s": flops / t_ref / 1e12}
for name, model, bf16, comp in cases:
    ti, p = (t_ref, p_ref) if name == "reference f32" else t_infer(model, bf16, a.reps)
    tt = t_train(model, bf16, max(3, a.reps // 2))
    rel = float((p - p_ref).norm() / p_ref.norm())
    res["cases"][name] = {"infer_s_per_traj": ti / a.batch, "train_s_per_step": tt, "rel_diff_vs_ref_f32": rel,
                          "tflops_per_s": flops / ti / 1e12}
    print(json.dumps({name: res["cases"][name]}), flush=True)
# ---------------- compiled variants: the whole AR step is compiled (as TF compiles it with XLA) ----------------
import torch._inductor.config as _icfg
if hasattr(_icfg, "size_asserts"):
    _icfg.size_asserts = False                                  # inductor stride assert on views (torch 2.4)
from .fast import _no_compile_fft
_no_compile_fft()


def compiled_copy(src, bf16=False, mode=None):
    m = copy.deepcopy(src)
    orig = m.step

    def _step(*args, **kw):
        args = [x.contiguous() if torch.is_tensor(x) else x for x in args]
        kw = {k: (v.contiguous() if torch.is_tensor(v) else v) for k, v in kw.items()}
        with torch.autocast(dev, torch.bfloat16, enabled=bf16):
            return orig(*args, **kw)
    m.step = torch.compile(_step, dynamic=False, mode=mode)
    return m


def run_case(name, model, bf16, train=True, tf32=False):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    try:
        ti, p = t_infer(model, bf16, a.reps)
        tt = t_train(model, bf16, max(3, a.reps // 2)) if train else None
        rel = float((p - p_ref).norm() / p_ref.norm())
        res["cases"][name] = {"infer_s_per_traj": ti / a.batch, "train_s_per_step": tt, "rel_diff_vs_ref_f32": rel,
                              "tflops_per_s": flops / ti / 1e12}
        print(json.dumps({name: res["cases"][name]}), flush=True)
    except Exception as ex:
        import traceback
        res["cases"][name] = {"error": repr(ex)[:400]}
        print("CASE_ERROR", name, traceback.format_exc()[-1500:], flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class CoreWrap(torch.nn.Module):
    """The FiLM-modulated axial core as a pure tensor function (h, e) -> h for one family layout (fixed roles)."""

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


def core_compiled_copy(src, dtype=torch.float32, mode=None):
    """Compile only the core (static shapes, no Python control flow); bf16 = a bf16 copy of the core weights."""
    m = copy.deepcopy(src)
    roles = list(b["roles"])
    cw = copy.deepcopy(CoreWrap(m, roles)).to(dtype)        # own copy: the rest of the model keeps float32 weights
    from .fast import freeze_fast_weights
    freeze_fast_weights(cw, roles, K=2, dtype=dtype)
    comp = torch.compile(cw, dynamic=False, fullgraph=True, mode=mode)

    def core(h, e, roles_):
        return comp(h.to(dtype), e.to(dtype)).to(h.dtype)
    m.core = core
    return m


def banks_compiled(m):
    """11-06 (bench6): additionally compile the mechanism bank (not fullgraph: the complex FFT of the spectral filter
    stays eager through _no_compile_fft)."""
    m.banks = torch.compile(m.banks, dynamic=False)
    return m


def freeze_kcache(m):
    """11-07 (bench7): the bank's k-space feature cache (dict of tensors) breaks Dynamo in torch 2.4
    (FakeRootModule ... _kcache). Fill it eagerly, then serve the entries as registered buffers."""
    bk = m.banks
    with torch.no_grad():
        predict(m, b, with_metrics=False)
    names = {}
    for i, (key, t) in enumerate(list(bk._kcache.items())):
        nm = f"_kc{i}"
        bk.register_buffer(nm, t.clone(), persistent=False)
        names[(key[0], key[1])] = nm

    def _kfeat(dims, comps, device):
        return getattr(bk, names[(tuple(dims), comps)]).to(bk.base.kernel.dtype)
    bk._kfeat = _kfeat
    return m


def banks7(m, dec=False):
    # 11-07 (bench8): the eager spectral filter breaks the graph inside _generic_spec, whose resumed frame is
    # specialized on the generic-head index i (one entry per head) -> more than the default 8 cache entries.
    import torch._dynamo as _dy
    _dy.config.cache_size_limit = 64
    if hasattr(_dy.config, "accumulated_cache_size_limit"):
        _dy.config.accumulated_cache_size_limit = 512
    freeze_kcache(m)
    m.banks = torch.compile(m.banks, dynamic=False)
    if dec:
        m.dec = torch.compile(m.dec, dynamic=False)
    return m



def bank9(m, bf16=False, encdec=False):
    """11-07 (bench9): bank compiled as ONE graph with the DFT spectral filter (fast.use_dft_spectral_filter);
    bf16=True runs the bank in bf16 weights/activations (output cast back to float32); encdec=True also compiles
    the bin encoder and the decoder."""
    import torch._dynamo as _dy
    _dy.config.cache_size_limit = 64
    from .fast import use_dft_spectral_filter
    use_dft_spectral_filter()
    freeze_kcache(m)
    bk = m.banks
    if bf16:
        bk.to(torch.bfloat16)
        inner = bk.forward

        def fwd(h, **kw):
            kw = {k: (v.to(torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in kw.items()}
            return inner(h.to(torch.bfloat16), **kw).float()
        bk.forward = fwd
    m.banks = torch.compile(bk, dynamic=False, fullgraph=True)
    if encdec:
        m.enc = torch.compile(m.enc, dynamic=False)
        m.dec = torch.compile(m.dec, dynamic=False)
    return m


B7 = os.environ.get("BENCH7", "0") == "1"
B9 = os.environ.get("BENCH9", "0") == "1"
if B9:
    from torch._dynamo.utils import counters as _ctr
    run_case("fused+core-compile bf16-weights", core_compiled_copy(fast, torch.bfloat16), False, train=False, tf32=True)
    run_case("+ bank fullgraph (DFT) f32", bank9(core_compiled_copy(fast, torch.bfloat16)), False, train=True, tf32=True)
    run_case("+ bank fullgraph (DFT) bf16", bank9(core_compiled_copy(fast, torch.bfloat16), bf16=True), False, train=False, tf32=True)
    run_case("+ bank (DFT) bf16 + enc/dec compile", bank9(core_compiled_copy(fast, torch.bfloat16), bf16=True, encdec=True), False, train=False, tf32=True)
    run_case("+ bank (DFT) f32 + enc/dec compile", bank9(core_compiled_copy(fast, torch.bfloat16), encdec=True), False, train=True, tf32=True)
    print("DYNAMO", {k: dict(v) for k, v in _ctr.items() if k in ("stats", "graph_break")}, flush=True)
elif B7:
    run_case("fused+core-compile bf16-weights", core_compiled_copy(fast, torch.bfloat16), False, train=False, tf32=True)
    run_case("+ bank compile (kcache frozen)", banks7(core_compiled_copy(fast, torch.bfloat16)), False, train=False, tf32=True)
    run_case("+ bank + decoder compile", banks7(core_compiled_copy(fast, torch.bfloat16), dec=True), False, train=True, tf32=True)
elif not B6:
    run_case("fused f32 + TF32", fast, False, tf32=True)
    run_case("fused+core-compile f32 + TF32", core_compiled_copy(fast), False, tf32=True)
    run_case("fused+core-compile bf16-weights", core_compiled_copy(fast, torch.bfloat16), False, tf32=True)
    run_case("fused+core-compile bf16-weights reduce-overhead", core_compiled_copy(fast, torch.bfloat16, "reduce-overhead"), False, train=False, tf32=True)
    run_case("fused+core-compile bf16-weights max-autotune", core_compiled_copy(fast, torch.bfloat16, "max-autotune-no-cudagraphs"), False, train=False, tf32=True)
else:
    run_case("fused+core-compile bf16-weights (bench5 best, rerun)", core_compiled_copy(fast, torch.bfloat16), False, tf32=True)
    run_case("fused+core-compile bf16-weights + bf16 autocast elsewhere", core_compiled_copy(fast, torch.bfloat16), True, tf32=True)
    run_case("fused+core+bank-compile bf16 + bf16 autocast elsewhere", banks_compiled(core_compiled_copy(fast, torch.bfloat16)), True, tf32=True)

# ---------------- profile: where the time goes (fused, bf16, eager and compiled) ----------------
from torch.profiler import ProfilerActivity, profile
PROF = ((("eager fused bf16", fast), ("core-compiled bf16-weights", core_compiled_copy(fast, torch.bfloat16))) if not (B6 or B7 or B9) else
        (("core+bank(DFT, bf16)+enc/dec-compiled steady", bank9(core_compiled_copy(fast, torch.bfloat16), bf16=True, encdec=True)),) if B9 else
        (("core+bank+dec-compiled bf16-weights", banks7(core_compiled_copy(fast, torch.bfloat16), dec=True)),) if B7 else
        (("core+bank-compiled bf16 + autocast", banks_compiled(core_compiled_copy(fast, torch.bfloat16))),))
if B9:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
for tag, mdl in PROF:
    try:
        with torch.no_grad(), torch.autocast(dev, torch.bfloat16, enabled=(tag.startswith("eager") or "autocast" in tag)):
            for _ in range(3):
                predict(mdl, b, with_metrics=False)
        sync()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with torch.no_grad(), torch.autocast(dev, torch.bfloat16, enabled=(tag.startswith("eager") or "autocast" in tag)):
                predict(mdl, b, with_metrics=False)
            sync()
        print(f"PROFILE {tag}\n" + prof.key_averages().table(sort_by="cuda_time_total", row_limit=25), flush=True)
    except Exception as ex:
        print("PROFILE_ERROR", tag, repr(ex)[:400], flush=True)
json.dump(res, open(a.out, "w"), indent=1)
print("BENCH_DONE", json.dumps(res))
