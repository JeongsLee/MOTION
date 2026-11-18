"""Timing of Poseidon-T / Poseidon-B with the released scOT code (revision R1, 2026-11-08), on the shapes of the
NS-PwC downstream task (4 channels, 128^2): inference = 20 forward calls per trajectory (one per lead of the
20-frame trajectory, direct or hop-1 AR alike), batch 8; fine-tuning update = forward + backward of a batch of 40
plus one AdamW step (the recipe of run_ft15_scot.sh, 3,200 updates x batch 40). Released defaults: float32
(Trainer fp16=False), no torch.compile; bf16 autocast reported alongside.

  python bench_scot_time.py /work/models/Poseidon-T /work/models/Poseidon-B
"""
import json
import sys
import time

import numpy as np
import torch
from scOT.model import ScOT

dev = "cuda"
res = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__}


def sync():
    torch.cuda.synchronize()


for path in sys.argv[1:]:
    name = path.rstrip("/").split("/")[-1]
    model = ScOT.from_pretrained(path).to(dev)
    C = model.config.num_channels
    pm = torch.zeros(C, dtype=torch.bool, device=dev)
    for prec in ("f32", "bf16-autocast"):
        ac = prec != "f32"
        model.eval()
        x = torch.randn(8, C, 128, 128, device=dev)
        t = torch.full((8,), 0.05, device=dev)

        def infer():
            with torch.no_grad(), torch.autocast(dev, torch.bfloat16, enabled=ac):
                y = x
                for _ in range(20):
                    y = model(pixel_values=y, time=t).output  # pixel_mask needs labels (scOT model.py l.1422)
                return y
        for _ in range(3):
            infer()
        sync(); ts = []
        for _ in range(10):
            sync(); t0 = time.perf_counter(); infer(); sync(); ts.append(time.perf_counter() - t0)
        ti = float(np.median(ts)) / 8
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-9)
        xb, yb = torch.randn(40, C, 128, 128, device=dev), torch.randn(40, C, 128, 128, device=dev)
        tb = torch.rand(40, device=dev)

        def upd():
            with torch.autocast(dev, torch.bfloat16, enabled=ac):
                out = model(pixel_values=xb, time=tb, pixel_mask=pm.expand(40, C), labels=yb)
            out.loss.backward()
            opt.step(); opt.zero_grad(set_to_none=True)
        for _ in range(3):
            upd()
        sync(); ts = []
        for _ in range(10):
            sync(); t0 = time.perf_counter(); upd(); sync(); ts.append(time.perf_counter() - t0)
        tu = float(np.median(ts))
        res[f"{name} {prec}"] = {"infer_s_per_traj_20_calls": ti, "update_s_batch40": tu,
                                 "params_M": sum(p.numel() for p in model.parameters()) / 1e6}
        print(json.dumps({f"{name} {prec}": res[f"{name} {prec}"]}), flush=True)
        del opt
    del model
    torch.cuda.empty_cache()
print("SCOT_TIME_DONE", json.dumps(res), flush=True)
