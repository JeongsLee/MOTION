"""Does the PER-FAMILY output head actually learn, or is private capacity starved of data?

The decoder already has both kinds of output parameter: a shared projection `dec_o` (hidden, S)
that every family updates, and `dec_famW` (n_fams, hidden*S), a zero-init additive delta that only
its own family ever touches. Fully privatising the output — one head per family instead of
shared-plus-delta — would cost almost nothing in parameters (96 x 73 active channels = 7k, less
than dec_famW's 40k). The real question is the other burden: a private parameter sees 1/26 of the
gradient a shared one does, and exposure has been this model's binding constraint all along.

dec_famW is the experiment that has already been running. It starts at exactly zero, so its norm
relative to the shared projection measures how much signal per-family parameters actually get. If
the deltas are still tiny after this many runs, privatising further puts capacity where the data
is not, which is the same way the MoE failed.

  python probe_famhead.py <ckpt.npz> [<ckpt2.npz> ...]
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/tmp/fv2")

from data.registry import FAMILIES, pretrain_families    # noqa: E402


def main():
    fams = {f.name for f in pretrain_families()}
    allf = list(FAMILIES.keys())
    for p in sys.argv[1:]:
        try:
            z = np.load(p)
        except Exception as e:                            # a run may not have checkpointed yet
            print(f"\n=== {p} ===\n  skip: {type(e).__name__}: {e}")
            continue
        ko = [k for k in z.files if "dec_o/kernel" in k]
        kf = [k for k in z.files if "dec_famW" in k]
        if not ko or not kf:
            print(f"\n=== {p} ===\n  no decoder head found")
            continue
        O, F = z[ko[0]], z[kf[0]]
        hid = O.shape[0]
        Fr = F.reshape(F.shape[0], hid, -1)
        nO = float(np.linalg.norm(O))
        rows = []
        for i, fn in enumerate(allf):
            if fn not in fams:
                continue
            n = float(np.linalg.norm(Fr[i]))
            rows.append((fn, n, 100.0 * n / (nO + 1e-12)))
        rows.sort(key=lambda r: -r[1])
        nz = sum(1 for _, n, _ in rows if n > 1e-6)
        print(f"\n=== {p} ===")
        print(f"shared dec_o {tuple(O.shape)}   |dec_o| = {nO:.4f}")
        print(f"per-family dec_famW {tuple(F.shape)} -> ({F.shape[0]},{hid},{Fr.shape[-1]})")
        print(f"family rows that moved off zero: {nz}/{len(rows)}")
        print(f"{'family':42s} {'|delta|':>9s} {'% of shared':>12s}")
        for fn, n, pc in rows[:8]:
            print(f"{fn:42s} {n:9.4f} {pc:11.1f}%")
        if len(rows) > 12:
            print(f"{'  ...':42s}")
            for fn, n, pc in rows[-4:]:
                print(f"{fn:42s} {n:9.4f} {pc:11.1f}%")
        print(f"MEAN per-family delta = {np.mean([r[2] for r in rows]):.1f}% of the shared "
              f"projection   (median {np.median([r[2] for r in rows]):.1f}%)")
    print("\nPROBE_OK")


if __name__ == "__main__":
    main()
