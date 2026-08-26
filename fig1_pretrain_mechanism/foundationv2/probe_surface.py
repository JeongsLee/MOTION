"""Which mechanisms are actually ON? Every gate in this model is zero-init, so its norm IS its
activation.

Two questions this answers. (1) The tendency banks: 17 mechanisms, each a zero-init per-channel
gate times a head, so a gate still near zero means that mechanism never earned its way into the
tendency. (2) The surface path, which exists specifically for the co-dimension-1 families
(drivaernet, shapenet_car, airfrans, geofno_*) whose volumetric box is ~97% empty — the slice
read `sf_out`, the wall-law boundary-layer bank `bl_gate`, and the curvature x tangential-flow
term `sk_gate` are all zero-init too, and all three were added on the argument that the box
cannot represent a wall. If they are still near zero the argument was wrong, or the path is
starved; if they are large the path carries real signal and the remaining mesh error is elsewhere.

Read alongside the measured box floors (diag_resolution.py): airfrans already scores BELOW its
box floor, which can only happen through the surface path, so at least for that family the path
demonstrably works.

  python probe_surface.py <ckpt.npz> [<ckpt2.npz> ...]
"""
from __future__ import annotations

import sys

import numpy as np


def norms(z, sub):
    out = {}
    for k in z.files:
        if sub in k:
            out[k] = float(np.linalg.norm(z[k]))
    return out


def main():
    for p in sys.argv[1:]:
        try:
            z = np.load(p)
        except Exception as e:
            print(f"\n=== {p} ===\n  skip: {type(e).__name__}")
            continue
        print(f"\n=== {p} ===")

        base = [float(np.linalg.norm(z[k])) for k in z.files if "banks_base" in k]
        ref = base[0] if base else 1.0
        print(f"reference: |banks_base| = {ref:.4f}   (the always-on tendency head)")

        gates = {k.split("banks_gate_")[1].split(":")[0]: float(np.linalg.norm(z[k]))
                 for k in z.files if "banks_gate_" in k}
        print(f"\nTENDENCY BANK GATES ({len(gates)} mechanisms, zero-init)")
        print(f"{'mechanism':22s} {'|gate|':>9s} {'% of base':>10s}")
        for m, v in sorted(gates.items(), key=lambda kv: -kv[1]):
            print(f"{m:22s} {v:9.4f} {100*v/max(ref,1e-9):9.1f}%")
        dead = [m for m, v in gates.items() if v < 1e-3]
        print(f"still at zero (<1e-3): {dead if dead else 'none'}")

        print("\nSURFACE PATH (all zero-init; these bypass the volumetric box)")
        rows = []
        for tag, sub in (("sf_out  slice read -> tendency", "sf_out"),
                         ("bl_gate boundary-layer bank", "bl_gate"),
                         ("bl_head", "bl_head"),
                         ("sk_gate curvature x flow", "sk_gate"),
                         ("sk_head", "sk_head"),
                         ("sf_k/sf_v/sf_q attention", "sf_"),
                         ("sf_beta locality", "sf_beta"),
                         ("bl_tau profile scale", "bl_tau")):
            n = norms(z, sub)
            if not n:
                continue
            tot = float(np.sqrt(sum(v * v for v in n.values())))
            rows.append((tag, tot, len(n)))
        for tag, v, c in rows:
            print(f"{tag:34s} {v:9.4f}  ({c} tensor{'s' if c > 1 else ''})")

        for k in z.files:
            if "sf_beta" in k or "bl_tau" in k:
                print(f"  raw {k}: {z[k].ravel()[:4]}")
    print("\nPROBE_SURF_OK")


if __name__ == "__main__":
    main()
