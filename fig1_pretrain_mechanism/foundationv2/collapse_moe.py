"""Fold an E-expert MoE checkpoint down to a single expert, keeping the trained function.

The mixture never earned its cost. Measured on r10/r13: the 16 experts sit 2.1-2.4% from their own
mean (3.4-5.5% after the load-balance loss), the top-1 gate share is 0.502-0.64 so the router
AVERAGES its two picks rather than choosing between them, and running all 16 instead of 2 moves
the class average by 0.13pp. Meanwhile the mixture is 75.5M of the model's 93M parameters and 82%
of a core block's FLOPs (38.7 GMAC against 6.0 for the axial conv and 2.8 for the attention).
Every lever that has actually moved this model was about channel overlap, not weight count.

So each block's experts are averaged into the e0 slot and the run continues with
`--moe_experts 1 --moe_topk 1`, where a 1-way softmax gate is identically 1.0 — the same code path,
no naming change, ~22M parameters, and the MLP cost of a dense block.

This is not exactly function-preserving: a top-2 blend of two experts computes
0.5*(o_a(gelu(h_a(x))) + o_b(gelu(h_b(x)))) while the fold computes o_bar(gelu(h_bar(x))), and gelu
does not commute with averaging. With experts a few percent apart the gap should be small, but the
`collapsed` eval that would have measured it OOMed twice, so the first eval after this fold is
where the price shows up. Revert by warm-starting from the pre-fold checkpoint if it is too high.

  python collapse_moe.py <in.npz> <out.npz> [n_blocks] [n_experts]
"""
from __future__ import annotations

import re
import sys

import numpy as np


def main():
    src, dst = sys.argv[1], sys.argv[2]
    depth = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    E = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    z = np.load(src)
    out, folded, dropped = {}, [], 0

    # b{i}_mlp_e{e}_{h,o}/{kernel,bias}  ->  mean over e, written back into the e0 slot
    # tolerant of the naming difference between Keras versions: the container gives
    # "b0_mlp_e0_h/kernel:0", a newer Keras gives "b0_mlp/b0_mlp_e0_h/kernel"
    pat = re.compile(r"(?:^|/)b(\d+)_mlp_e(\d+)_([ho])/(kernel|bias)(:0)?$")
    groups = {}
    for k in z.files:
        m = pat.search(k)
        if m:
            groups.setdefault((int(m.group(1)), m.group(3), m.group(4), m.group(5) or ""),
                              {})[int(m.group(2))] = k

    for k in z.files:
        m = pat.search(k)
        if m is None:
            if "_mlp_router" in k:                 # an E-way router cannot be reused at E=1
                dropped += 1
                continue
            out[k] = z[k]
            continue
        if int(m.group(2)) != 0:                   # experts 1..E-1 disappear into the mean
            dropped += 1

    for (bi, ho, wb, suf), members in sorted(groups.items()):
        assert len(members) == E, f"block {bi} {ho}/{wb}: found {len(members)} experts, expected {E}"
        mu = np.mean([z[members[e]] for e in range(E)], 0)
        out[members[0]] = mu.astype(z[members[0]].dtype)
        sp = float(np.linalg.norm(np.stack([z[members[e]] for e in range(E)]) - mu)
                   / (np.linalg.norm(mu) * np.sqrt(E) + 1e-12))
        folded.append(f"b{bi}_mlp_{ho}/{wb}: {E} -> 1  (expert spread was {100*sp:.1f}%)")

    np.savez(dst, **out)
    p_in = sum(int(np.prod(z[k].shape)) for k in z.files)
    p_out = sum(int(np.prod(v.shape)) for v in out.values())
    print(f"folded {src} -> {dst}")
    for f in folded:
        print("  " + f)
    print(f"tensors: {len(z.files)} -> {len(out)}  (dropped {dropped})")
    print(f"params:  {p_in/1e6:.1f}M -> {p_out/1e6:.1f}M")
    assert len(folded) == depth * 4, f"expected {depth*4} folded tensors, got {len(folded)}"
    print("COLLAPSE_MOE_OK")


if __name__ == "__main__":
    main()
