"""REVISION-R1 10-24: widen a t_in-1 checkpoint to a t_in-T window without changing its output on a 1-frame input.
The encoder input is [frames time-major per node (T*S) | frame mask (T) | geom (G)] (V3Model.feats_from_win), and the
window is right-aligned (latest frame last). The old rows go to the LATEST frame slot, its mask row and the geometry
rows; the rows of the T-1 older frames and their masks start at zero, so with only the latest frame observed (the
others zero, mask 0) the encoder sees exactly the old features. Usage: python pad_tin.py IN.npz OUT.npz T [S=16] [G=8]"""
import sys, numpy as np
src, dst, T = sys.argv[1], sys.argv[2], int(sys.argv[3])
S = int(sys.argv[4]) if len(sys.argv) > 4 else 16; G = int(sys.argv[5]) if len(sys.argv) > 5 else 8
z = dict(np.load(src, allow_pickle=True)); k = "enc/enc_f/kernel:0"; W = z[k]
if W.shape[0] == T * S + T + G:                # already a t_in-T checkpoint (e.g. continuing stage 3): copy unchanged
    np.savez(dst, **z); print(f"[pad_tin] {k}: {W.shape} already t_in {T}; copied -> {dst}", flush=True); sys.exit(0)
assert W.shape[0] == S + 1 + G, (W.shape, S, G)
Wn = np.zeros((T * S + T + G, W.shape[1]), W.dtype)
Wn[(T - 1) * S:T * S] = W[:S]                 # latest frame
Wn[T * S + T - 1] = W[S]                      # its frame-mask row
Wn[T * S + T:] = W[S + 1:]                    # geometry
z[k] = Wn
np.savez(dst, **z)
print(f"[pad_tin] {k}: {W.shape} -> {Wn.shape}; {len(z)} arrays -> {dst}", flush=True)
