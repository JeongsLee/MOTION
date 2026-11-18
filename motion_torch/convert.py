"""TF weights-only npz (core/pretrain.save_ckpt) -> PyTorch state_dict.

Every torch parameter knows its TF variable name (V3Model.tf_name_map()); the npz is keyed by exactly those names.
The load replicates core/pretrain.load_ckpt rule by rule:
  * exact shape                          -> copied
  * axis-0 grew (e.g. dec_famW 26 -> 32 family rows, enc_f input rows)   -> old rows copied, new rows zero
  * banks_/steady_o last axis = 2x       -> duplicated halves (per-basis split; not used by v3 checkpoints)
  * last axis grew by a multiple of d_w  -> n_p panel zero-pad
  * absent from the npz                  -> keeps its init (zero-init gates/heads; reported)
  * DEAD-BANK REPAIR (only when the config has no_dead_repair=False, i.e. NO_DEAD_REPAIR unset): an all-zero
    banks_* tensor (not base, not gate) keeps its fresh init instead of the stored zeros. For the released
    checkpoints every such tensor is a bias or a gcond kernel whose fresh init is zero, so the repair is a no-op;
    any affected tensor with a random init is listed in the report (it cannot match TF's random draw).
Parameters stay in the TF layout (see model/layers.py), so a copy is the whole conversion.

  python -m motion_torch.convert --ckpt m1f_ckpt_best.npz --preset motion_s --out motion_s.pt
"""
from __future__ import annotations

import argparse
import dataclasses
import json

import numpy as np
import torch

from .config import PRESETS, MotionConfig
from .model.v3model import V3Model


def _try_np_pad(vs, w, d_w, nm):
    ws = tuple(int(x) for x in w.shape)
    if "gain_t" in nm and len(vs) == 1 and len(ws) == 1 and vs[0] > ws[0]:
        out = np.ones(vs, w.dtype); out[:ws[0]] = w; return out
    if d_w and len(vs) == len(ws) and vs[:-1] == ws[:-1] and vs[-1] > ws[-1] and vs[-1] % d_w == 0 \
            and ws[-1] % d_w == 0:
        np_new, np_old = vs[-1] // d_w, ws[-1] // d_w
        out = np.zeros((*vs[:-1], d_w, np_new), w.dtype)
        out[..., :np_old] = w.reshape(*ws[:-1], d_w, np_old)
        return out.reshape(vs)
    return None


def load_tf_npz(model: V3Model, path, strict=False, verbose=True):
    """Load a TF npz into `model` in place (load_ckpt semantics). Returns a report dict."""
    cfg = model.cfg
    ck = np.load(path)
    have = set(ck.files)
    params = dict(model.named_parameters())
    tmap = model.tf_name_map()
    rep = {"loaded": [], "padded": [], "new_at_init": [], "shape_mismatch": [], "dead_repair": [], "basis_dup": [],
           "unused_npz_keys": sorted(have - set(tmap.values()))}
    with torch.no_grad():
        for pn, nm in tmap.items():
            p = params[pn]
            vs = tuple(p.shape)
            if nm not in have:
                rep["new_at_init"].append(nm)
                continue
            w = ck[nm]
            if ("banks_" in nm and "banks_base" not in nm and "gate" not in nm and not np.any(w)
                    and not cfg.no_dead_repair):
                rep["dead_repair"].append((nm, bool(p.abs().max() > 0)))   # (name, fresh init non-zero?)
                continue
            if tuple(w.shape) == vs:
                p.copy_(torch.from_numpy(np.asarray(w, np.float32)).reshape(vs)); rep["loaded"].append(nm)
            elif (("banks_" in nm or "steady_o" in nm) and tuple(w.shape[:-1]) == vs[:-1]
                  and vs[-1] == 2 * int(w.shape[-1])):
                p.copy_(torch.from_numpy(np.concatenate([w, w], -1).astype(np.float32))); rep["basis_dup"].append(nm)
            elif len(vs) == w.ndim and vs[0] > w.shape[0] and vs[1:] == tuple(w.shape[1:]):
                pad = np.zeros(vs, np.float32); pad[:w.shape[0]] = w
                p.copy_(torch.from_numpy(pad)); rep["padded"].append(nm)
            else:
                pw = _try_np_pad(vs, w, cfg.d_w, nm)
                if pw is not None:
                    p.copy_(torch.from_numpy(pw.astype(np.float32))); rep["padded"].append(nm)
                else:
                    rep["shape_mismatch"].append((nm, vs, tuple(w.shape)))
    if verbose:
        print(f"[convert] {path}: loaded {len(rep['loaded'])}, padded {len(rep['padded'])} {rep['padded']}, "
              f"new-at-init {len(rep['new_at_init'])} {rep['new_at_init']}, shape-mismatch {rep['shape_mismatch']}, "
              f"dead-bank-repair {len(rep['dead_repair'])} (random-init among them: "
              f"{[n for n, r in rep['dead_repair'] if r]}), unused npz keys {rep['unused_npz_keys']}", flush=True)
    if strict and (rep["shape_mismatch"] or rep["unused_npz_keys"]):
        raise RuntimeError(f"strict load failed: {rep['shape_mismatch']} {rep['unused_npz_keys']}")
    return rep


def build_model(cfg: MotionConfig, ckpt=None, device="cpu", seed=0, strict=False):
    """V3Model for `cfg`; ckpt = TF .npz (converted on the fly) or a torch .pt written by this script."""
    torch.manual_seed(seed)
    if ckpt is not None and str(ckpt).endswith(".pt"):
        blob = torch.load(ckpt, map_location="cpu")
        cfg = MotionConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in blob["config"].items()})
        model = V3Model(cfg)
        model.load_state_dict(blob["state_dict"])
    else:
        model = V3Model(cfg)
        if ckpt is not None:
            load_tf_npz(model, ckpt, strict=strict)
    return model.to(device).eval()


def save_pt(model: V3Model, path, report=None):
    torch.save({"config": dataclasses.asdict(model.cfg), "state_dict": model.state_dict(),
                "tf_names": model.tf_name_map(), "convert_report": report}, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="TF weights-only npz")
    ap.add_argument("--preset", default="motion_s", choices=sorted(PRESETS) + ["env"],
                    help="released configuration (or 'env': read the MOTION_* environment like the TF code)")
    ap.add_argument("--d", type=int); ap.add_argument("--depth", type=int); ap.add_argument("--d_w", type=int)
    ap.add_argument("--n_p", type=int); ap.add_argument("--t_in", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strict", type=int, default=0)
    a = ap.parse_args()
    cfg = MotionConfig.from_env() if a.preset == "env" else PRESETS[a.preset]
    over = {k: getattr(a, k) for k in ("d", "depth", "d_w", "n_p", "t_in") if getattr(a, k) is not None}
    cfg = cfg.replace(**over)
    torch.manual_seed(0)
    model = V3Model(cfg)
    rep = load_tf_npz(model, a.ckpt, strict=bool(a.strict))
    n = sum(p.numel() for p in model.parameters())
    save_pt(model, a.out, report={k: (len(v) if isinstance(v, list) else v) for k, v in rep.items()})
    print(f"[convert] {cfg.name}: {n / 1e6:.2f}M params -> {a.out}")
    print(json.dumps({k: v for k, v in dataclasses.asdict(cfg).items()}, default=str))


if __name__ == "__main__":
    main()
