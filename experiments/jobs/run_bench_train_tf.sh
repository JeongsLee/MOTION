#!/bin/bash
# REVISION-R1 11-18: TensorFlow/XLA training-step time of the MOTION-NS configuration on ONE H100 (batch 8, bf16),
# synthetic samples (v3/train.py MOTION_BENCH_STEPS hook). Setting 1: training example sizes (8,192 encoder points,
# 2,048 collocation points); setting 2: the sizes of the PyTorch bench11 step (16,384 points, no collocation).
set -x
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
OUT=/work/runs/bench_train_tf; mkdir -p $OUT
cp -r /work/MOTION/motion_tf /tmp/fv2; cp /work/MOTION/motion_tf_bench/train.py /tmp/fv2/v3/train.py
md5sum /tmp/fv2/v3/train.py /tmp/fv2/v3/model.py /tmp/fv2/v3/data_ar.py | tee $OUT/md5.txt
cd /tmp/fv2
pip install -q "numpy<2" h5py scipy netCDF4 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv | tee $OUT/gpu.txt
python -c "import tensorflow as tf; print('TF', tf.__version__)" 2>&1 | tail -1 | tee -a $OUT/gpu.txt
export MOTION_WARP_LIVE=1 MOTION_BANK_NGENERIC=8 MOTION_BANK_GENERIC_MERGE=1 MOTION_SPLIT_MTF=1 MOTION_INPUT_ZSCORE=1 MOTION_COND_STATS=1
export MOTION_BENCH_STEPS=20 MOTION_BENCH_ENC=16384 MOTION_BENCH_COLLOC=0
python -m v3.train --families shallow_water,com_ns,incom_ns,pdearena_ns,cfdbench,pdearena_uncond --steps 10 \
  --batch 8 --batch3d 1 --d 384 --depth 8 --d_w 16 --n_p 16 --box_table 0 --dims2 64 --gate_cond 1 --fam_head 1 --dict_decode 0 \
  --t_in 10 --k_fut 10 --k_relax 4 --enc_n 8192 --n_colloc 2048 --xla 1 \
  --lr 1e-4 --warmup 4000 --lr_sched cosine --lr_alpha 0.05 --fluct_loss 0 --bf16 1 --seed 0 \
  --save $OUT 2>&1 | tee $OUT/bench_raw.log | grep -E '^\[bench\]|BENCH_TRAIN|params|replicas|Error|Traceback|OOM'
nvidia-smi --query-gpu=memory.used --format=csv | tee -a $OUT/gpu.txt
echo JOB_DONE
