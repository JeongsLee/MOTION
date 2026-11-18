"""REVISION-R1 11-03 (R1 comment 4): adds MOTION-NS to the one-trajectory SWE npz of swe_onetraj.py, from the
eval_equiv.py run with EQUIV_DUMP_MEMBERS=fr,d1,id_s8 (job nss-equiv-onetraj; first 16 test trajectories, per-trajectory
errors match nss_equiv_zsc/run4). Dumps are inverse-transformed by eval_equiv.py; h is slot 8; last frame. Titles use
the float32 per-trajectory errors of the run (the dumps are float16).
  python3 -I swe_onetraj_ns.py <swe_onetraj.npz> <nss equiv.npz> <out.npz>"""
import sys
import numpy as np
Z, A, OUT = np.load(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3]
k, S, f = int(Z["traj"]), 8, "shallow_water"
out = {x: Z[x] for x in Z.files}
gt = A[f"{f}|dump_gt"][k, ..., S].astype(np.float32)
assert np.abs(gt - Z["gt"]).max() < 1e-3, "trajectory mismatch"
m = "MOTION-NS"; out[f"{m}|id"] = A[f"{f}|dump_pred"][k, ..., S].astype(np.float32)
for g, d in (("fr", "fr"), ("d1", "d1"), ("sh8", "id_s8")):
    out[f"{m}|{g}"] = A[f"{f}|dumpm_{d}"][k, ..., S].astype(np.float32)
    out[f"{m}|err|{g}"] = float(A[f"{f}|{d}|err"][k, -1])
out[f"{m}|err|id"] = float(A[f"{f}|id|err"][k, -1])
np.savez_compressed(OUT, **out)
print({x: round(100 * out[x], 3) for x in out if x.startswith(m + "|err")})
