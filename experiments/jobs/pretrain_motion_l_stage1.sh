#!/bin/bash
# MOTION-L, first stage: 4 x H100, 102.6 h; stopped at 106,250 updates.
# Command of the original cluster job (2026-08-06, image tensorflow/tensorflow:2.15.0-gpu), as submitted.
# It ran the code of August 2026; the present motion_tf/ contains the second-stage options (--fluct_loss) directly,
# so the overlay copy below is no longer needed. Released checkpoints need MOTION_LEGACY_VOCAB=1
# MOTION_VORTEX2D_LEGACY=1 with the present code (see WEIGHTS.md).
export PYTHONUNBUFFERED=1 CORPUS_ROOT=/corpus TF_CPP_MIN_LOG_LEVEL=2
cp -r /work/MOTION/motion_tf /tmp/fv2; cd /tmp/fv2
pip install -q "numpy<2" h5py scipy || true
mkdir -p /corpus/results/motion_m2
python -m v3.train --families motion --steps 160000 --batch 2 --batch3d 1 --d 896 --depth 12 --d_w 48 --n_p 16 --box_table 1 --gate_cond 1 --fam_head 1 --dict_decode 0 --t_in 10 --k_fut 10 --k_relax 4 --enc_n 8192 --n_colloc 2048 --reuse3d 8 --xla 1 --eval_every 2000 --eval_n 8 --ckpt_every 2000 --save /corpus/results/motion_m2 2>&1 | tee -a /corpus/results/motion_m2/train.log
echo "rc=$?"
