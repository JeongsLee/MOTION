#!/bin/bash
# MOTION-S, second stage: 1 x H100, 9.5 h; stopped at 25,150 updates; checkpoint at 20,000 = released weights.
# Command of the original cluster job (2026-08-08, image tensorflow/tensorflow:2.15.0-gpu), as submitted.
# It ran the code of August 2026; the present motion_tf/ contains the second-stage options (--fluct_loss) directly,
# so the overlay copy below is no longer needed. Released checkpoints need MOTION_LEGACY_VOCAB=1
# MOTION_VORTEX2D_LEGACY=1 with the present code (see WEIGHTS.md).
export PYTHONUNBUFFERED=1 CORPUS_ROOT=/corpus TF_CPP_MIN_LOG_LEVEL=2
cp -r /work/MOTION/motion_tf /tmp/fv2; cd /tmp/fv2
cp /work/MOTION/motion_tf_stage2_overlay/train.py /work/MOTION/motion_tf_stage2_overlay/model.py /work/MOTION/motion_tf_stage2_overlay/data_ar.py v3/
grep -c 'fluct_loss' v3/train.py
pip install -q "numpy<2" h5py scipy || true
mkdir -p /corpus/results/motion_m1f
if [ ! -f /corpus/results/motion_m1f/init_stage1.npz ]; then
  cp /corpus/results/motion_m1/ckpt_best.npz /corpus/results/motion_m1f/init_stage1.npz
fi
echo "stage-2 init = snapshot of motion_m1/ckpt_best (10.15%@110k lineage)"
python -m v3.train --families motion --steps 30 --batch 4 --batch3d 1 \
  --d 384 --depth 8 --d_w 16 --n_p 16 --box_table 1 --gate_cond 1 --fam_head 1 \
  --dict_decode 0 --t_in 10 --k_fut 10 --k_relax 4 --enc_n 8192 --n_colloc 2048 \
  --reuse3d 8 --xla 1 --lr 5e-5 --warmup 200 --eval_every 100000 --fluct_loss 1 \
  --init_ckpt /corpus/results/motion_m1f/init_stage1.npz --save /tmp/gate > /tmp/gate.log 2>&1 \
  || { tail -20 /tmp/gate.log; echo GATE_FAIL; exit 1; }
tail -3 /tmp/gate.log; echo GATE_OK
python -m v3.train --families motion --steps 40000 --batch 4 --batch3d 1 \
  --d 384 --depth 8 --d_w 16 --n_p 16 --box_table 1 --gate_cond 1 --fam_head 1 \
  --dict_decode 0 --t_in 10 --k_fut 10 --k_relax 4 --enc_n 8192 --n_colloc 2048 \
  --reuse3d 8 --xla 1 --lr 5e-5 --warmup 1000 --eval_every 2000 --eval_n 8 --ckpt_every 2000 \
  --fluct_loss 1 --init_ckpt /corpus/results/motion_m1f/init_stage1.npz \
  --save /corpus/results/motion_m1f 2>&1 | tee -a /corpus/results/motion_m1f/train.log
echo "rc=$?"
