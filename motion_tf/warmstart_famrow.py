"""Warm-start a downstream family's OUTPUT HEAD row from a kin family's trained row.

Why this is needed: v3 is built with --fam_head 1, which sets pointio's `fam_only=True` -- the
per-family row of `dec_famW` IS the entire output head, not a delta on a shared projection
(core/pointio.py:115). The Fig-2 downstream families were appended LAST in the registry, so
NS-PwC lands at fam_id 26 while the pretrained table has rows 0..25; load_ckpt zero-pads the new
row, and a zero head decodes to exactly the anchor (measured: rms(pred-anchor) = 0.0 for NS-PwC
vs 0.11 for incom_ns, every other input to the model being identical). Zero-shot is therefore
persistence by construction, and a low-shot finetune has to learn the head from scratch -- which
is not the comparison we want against Poseidon's shared decoder.

Copying incom_ns's row (fam_id 3) is the natural transfer: same physics (incompressible NS),
same active slots (0 = vx, 1 = vy), same operator set (advection/diffusion/pressure_projection).
"""
import os
import sys

import numpy as np

SRC, DST = sys.argv[1], sys.argv[2]
FROM_ID = int(os.environ.get("FROM_ID", "3"))     # incom_ns
TO_ID = int(os.environ.get("TO_ID", "26"))        # NS-PwC
N_FAMS = int(os.environ.get("N_FAMS", "0"))       # 0 = keep current row count

z = dict(np.load(SRC))
key = [k for k in z if "dec_famW" in k]
assert len(key) == 1, f"expected one dec_famW, found {key}"
k = key[0]
W = z[k]
print(f"[warmstart] {k} {W.shape}  from row {FROM_ID} -> row {TO_ID}")
rows = max(N_FAMS, TO_ID + 1, W.shape[0])
if rows > W.shape[0]:
    W = np.concatenate([W, np.zeros((rows - W.shape[0], W.shape[1]), W.dtype)], 0)
    print(f"[warmstart] grew table to {W.shape}")
print(f"[warmstart] |row {FROM_ID}| = {np.linalg.norm(W[FROM_ID]):.4e}  "
      f"|row {TO_ID}| before = {np.linalg.norm(W[TO_ID]):.4e}")
W[TO_ID] = W[FROM_ID]
print(f"[warmstart] |row {TO_ID}| after  = {np.linalg.norm(W[TO_ID]):.4e}")
z[k] = W
np.savez(DST, **z)
print(f"[warmstart] wrote {DST}")
