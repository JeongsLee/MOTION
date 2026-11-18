#!/bin/bash
# Local GPU chain 18 (m1f): NS-Sines diffusion knob on test trajectory 4 (= the shear/vortex trajectory of Fig. 2c), with the
# amplitude-calibrated factor f = 2.01 (nu x2; elasticity from calibration trajectories 0-3) and the uncalibrated x2.
L=${MOTION_LOCAL:-motion_local}; O=$L/out_knob
exec 9>$L/chain.lock; flock 9
cd ${MOTION_REPO:-.}/motion_tf
export TF_FORCE_GPU_ALLOW_GROWTH=true MOTION_LEGACY_VOCAB=1 MOTION_VORTEX2D_LEGACY=1 NO_DEAD_REPAIR=1 TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1 CORPUS_ROOT=/tmp
PY=python
CAP="systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0"
# store the velocity response of the Fig. 2c diffusion command (Laplacian heads x1.217, traj 4) for a velocity r vs the solver
LOCAL_UNITS=$L/units.pkl $CAP $PY -u latent_knob.py --ckpt $L/m1f_ckpt_best.npz --fam NS-Sines --n 5 --only_traj 4 --factors 1.217 --calib 0 \
  --only_knob diff --lap_steer_heads diffusion,coefficient,compressible --out $O/group_t4_uv.npz > $O/group_t4_uv.log 2>&1
grep -E "Traceback|KNOB_OK" $O/group_t4_uv.log; echo CHAIN26_DONE
