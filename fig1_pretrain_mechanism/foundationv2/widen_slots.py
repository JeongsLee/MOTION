"""Re-shape a trained checkpoint from S_old output slots to S_new, losing nothing.

NUM_SLOTS went 9 -> 12 to give car-surface pressure, wave displacement and the steady
non-pressure scalars their own output channels. Four tensors carry the slot count and each needs
a DIFFERENT treatment — a generic "pad the end with zeros" would silently scramble the encoder,
which is exactly the failure mode that corrupted r8 once already:

  enc/enc_f/kernel   (t_in*S + t_in + GEOM, d)   rows are time-major per node, so row t*S+s must
                                                 move to t*S_new+s and the fmask/geom rows shift
  dec/dec_o/kernel   (hidden, S)                 append zero columns
  tgate/kernel       (hidden_e, S)               append zero columns
  dec/dec_famW       (n_fams, hidden*S)          reshape (n_fams, hidden, S), pad, flatten

Everything else is copied byte-for-byte. The new gate-specialisation variables are absent on
purpose: gcond and gmix are zero-init, so the model's function at load is unchanged and the
loader reports them as new-at-init.

New slots INHERIT the head of the slot they were split off, rather than starting at zero: slot 9
(car-surface pressure) copies slot 4's weights, slot 10 (wave) copies slot 6, slot 11 copies 4.
That makes the split free at t=0 in both directions — a remapped family's field now enters slot 9
and is read by a copy of the head that was reading it on slot 4, so its prediction is unchanged,
while the families that stayed on slot 4 see zeros in slot 9 and are equally unaffected. The two
heads only diverge once training pulls them apart, which is the entire point of splitting.

  python widen_slots.py <in.npz> <out.npz> [S_old] [S_new] [t_in] [geom_w]
"""
from __future__ import annotations

import sys

import numpy as np


def main():
    src, dst = sys.argv[1], sys.argv[2]
    So = int(sys.argv[3]) if len(sys.argv) > 3 else 9
    Sn = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    t_in = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    geom = int(sys.argv[6]) if len(sys.argv) > 6 else 8
    INHERIT = {9: 4, 10: 6, 11: 4}          # new slot -> the slot it was split off
    z = np.load(src)
    out, touched = {}, []

    for k in z.files:
        v = z[k]
        if k.endswith("enc_f/kernel:0") or k.endswith("enc_f/kernel"):
            exp = t_in * So + t_in + geom
            assert v.shape[0] == exp, f"{k}: expected {exp} rows, got {v.shape[0]}"
            w = np.zeros((t_in * Sn + t_in + geom, v.shape[1]), v.dtype)
            for t in range(t_in):                       # time-major frame block, per slot
                w[t * Sn:t * Sn + So] = v[t * So:t * So + So]
                for nsl, osl in INHERIT.items():        # new slot reads like its parent slot
                    w[t * Sn + nsl] = v[t * So + osl]
            w[t_in * Sn:] = v[t_in * So:]               # frame mask + geometry rows, shifted
            out[k], touched = w, touched + [f"{k} {v.shape}->{w.shape} (row remap)"]
        elif k.endswith("dec_o/kernel:0") or k.endswith("dec_o/kernel") \
                or k.endswith("tgate/kernel:0") or k.endswith("tgate/kernel"):
            assert v.shape[-1] == So, f"{k}: expected {So} columns, got {v.shape[-1]}"
            w = np.zeros(v.shape[:-1] + (Sn,), v.dtype)
            w[..., :So] = v
            for nsl, osl in INHERIT.items():
                w[..., nsl] = v[..., osl]
            out[k], touched = w, touched + [f"{k} {v.shape}->{w.shape} (zero columns)"]
        elif k.endswith("dec_famW:0") or k.endswith("dec_famW"):
            assert v.shape[1] % So == 0, f"{k}: {v.shape[1]} not divisible by {So}"
            hid = v.shape[1] // So
            w = np.zeros((v.shape[0], hid, Sn), v.dtype)
            vv = v.reshape(v.shape[0], hid, So)
            w[:, :, :So] = vv
            for nsl, osl in INHERIT.items():
                w[:, :, nsl] = vv[:, :, osl]
            w = w.reshape(v.shape[0], hid * Sn)
            out[k], touched = w, touched + [f"{k} {v.shape}->{w.shape} (per-family head)"]
        else:
            out[k] = v

    np.savez(dst, **out)
    print(f"widened {src} -> {dst}   S {So} -> {Sn}")
    for t in touched:
        print("  " + t)
    assert len(touched) == 4, f"expected 4 slot-carrying tensors, remapped {len(touched)}"
    print(f"copied unchanged: {len(out) - len(touched)}")
    print("WIDEN_OK")


if __name__ == "__main__":
    main()
