"""Port-only smoke test (no TensorFlow, no checkpoint needed): every preset builds, the TF name map is complete,
AR step / single-shot / rh run on synthetic samples, structural invariants of the reference hold, gradients flow.

  ~/anaconda3/envs/torch/bin/python tests/test_torch_smoke.py      (or pytest)"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch

from motion_torch import PRESETS, V3Model
from motion_torch.data.example import compute_global_norm, make_example_ar, stack_batch, synthetic_sample
from motion_torch.inference import predict, to_torch


def _batch(cfg, fam, rng, dims=(16, 16)):
    s = synthetic_sample(fam, rng, dims=dims if len(dims) == 2 else dims, T=14)
    gn = {fam: compute_global_norm([s])[:2]} if cfg.global_norm else None
    ex = make_example_ar(s, cfg.t_in, 4, 2, 64, 256, rng, cfg, box_dims=(8,) * len(dims), global_norm=gn)
    return to_torch(stack_batch([ex]), "cpu")


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    for name, base in PRESETS.items():
        cfg = base.replace(d=32, depth=2, d_w=4, n_p=8 if base.n_p == 32 else 4, d_cond=32)
        m = V3Model(cfg)
        tmap = m.tf_name_map()
        assert len(tmap) == len(dict(m.named_parameters()))
        with torch.no_grad():                                  # open every zero-init gate / head
            for p in m.parameters():
                p.add_(0.05 * torch.randn_like(p))
        for fam, dims in (("incom_ns", (16, 16)), ("ACE", (16, 16)),
                          ("pdebench3d_cns_Rand_M0.1_Eta1e-08_Zeta1e-08", (8, 8, 8)), ("Poisson-Gauss", (16, 16))):
            b = _batch(cfg, fam, rng, dims)
            o = predict(m, b)
            assert torch.isfinite(o["pred"]).all(), (name, fam)
            if b["mode"] == "ss":                              # lead subsets == the same leads of a full call
                full = o["pred"]
                sub = predict(m, b, leads=[1, 3], with_metrics=False)["pred"]
                assert torch.allclose(sub[:, 0], full[:, 0], atol=1e-5) and torch.allclose(sub[:, 1], full[:, 2],
                                                                                         atol=1e-5)
            print(f"[smoke] {name:13s} {fam:44s} mode {b['mode']:6s} pred {tuple(o['pred'].shape)} "
                  f"rel_phys {float(o['rel_phys'].mean()):.3f}")
        # hard IC: z = 0 -> the decoder is bias-free, u_next = u_prev on the physical slots (no transport)
        b = _batch(cfg, "ACE", rng)
        m.z_scale = 0.0
        with torch.no_grad():
            qall = torch.cat([b["coords_node"], b["coords_q"]], 1); gall = torch.cat([b["geom_node"], b["geom_q"]], 1)
            up = torch.cat([b["x_win"][:, -1], b["ic_q"]], 1)
            u1 = m.step(b["coords_node"], b["x_win"], b["fmask"], b["geom_node"], 2, b["roles"], b["cond"],
                        b["op_multihot"], qall, gall, up, fam_id=b["fam_id"], sdf_box=b["sdf_box"],
                        box_dims=b["box_dims"], fbmask=b["fbmask"])
        m.z_scale = None
        assert torch.allclose(u1[..., :12], up[..., :12], atol=1e-6)
        # gradients
        b = _batch(cfg, "incom_ns", rng)
        m.train()
        from motion_torch.finetune import rollout_loss_backward
        L = rollout_loss_backward(m, b, k_seg=2)
        n_g = sum(1 for p in m.parameters() if p.grad is not None and float(p.grad.abs().max()) > 0)
        assert np.isfinite(L) and n_g > 0.6 * len(list(m.parameters())), (L, n_g)
        print(f"[smoke] {name}: {len(tmap)} params named, rollout loss {L:.4f}, {n_g} params with gradient")
    assert "tensorflow" not in sys.modules
    print("[smoke] OK (tensorflow not imported)")


def test_torch_smoke():
    main()


if __name__ == "__main__":
    main()
