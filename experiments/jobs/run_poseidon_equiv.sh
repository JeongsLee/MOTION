#!/bin/bash
# REVISION-R1 (10-22): Poseidon-T equivariance (D4 x shifts) on the six pretraining families (poseidon_equiv.py).
set -e -o pipefail
export PYTHONUNBUFFERED=1
OUT=/work/runs/poseidon_equiv; mkdir -p $OUT
mkdir -p /tmp/corpus/cache /tmp/corpus/raw/poseidon
ln -s /corpus/meta /tmp/corpus/meta; ln -s /eu/data/poseidon/_assembled /tmp/corpus/raw/poseidon_assembled
for p in /work/data/poseidon/*; do b=$(basename $p); [ "$b" = "NS-Gauss" ] || ln -s $p /tmp/corpus/raw/poseidon/$b; done
ln -s /eu/data/poseidon_raw/NS-Gauss /tmp/corpus/raw/poseidon/NS-Gauss
export CORPUS_ROOT=/tmp/corpus
cp -r /work/MOTION/motion_tf /tmp/fv2; cd /tmp/fv2
pip install -q "numpy<2" h5py netCDF4 scipy 2>&1 | tail -1
pip install -q --no-deps https://github.com/camlab-ethz/poseidon/archive/refs/heads/main.zip 2>&1 | tail -2
pip install -q "transformers==4.29.2" "accelerate==0.31.0" safetensors huggingface_hub 2>&1 | tail -2
python -c "import torch;print(torch.__version__,torch.cuda.get_arch_list());torch.zeros(1).cuda()+1"
export POSEIDON_VA1=CE-RP:2,CE-CRP:2,CE-KH:2,CE-Gauss:2   # CE adapter layout (CE-RP signature 1.03 with ch2 vs 14 with ch1); continuity test not decisive at dt=1/20
export SPLIT=${SPLIT:-val}; S_=$( [ $SPLIT = val ] && echo "" || echo "_$SPLIT" )${NEV:+_n$NEV}   # 10-22: test-split run keeps the val files
PM=${PMODEL:-Poseidon-T}   # 10-22: PMODEL env (Poseidon-T default, Poseidon-B)
python -u poseidon_equiv.py camlab-ethz/$PM $OUT/${PM}_equiv$S_.json 2>&1 | tee $OUT/${PM}_equiv$S_.log
# REVISION-R1 11-02: the heldout JSONs written ~3 s before exit were lost in the object-volume sync -> second copy on
# the cluster volume, flush, and check that the JSON is listed before the job ends
mkdir -p /corpus/results/poseidon_equiv; cp $OUT/${PM}_equiv$S_.json /corpus/results/poseidon_equiv/
sync; sleep 60; ls -la $OUT/${PM}_equiv$S_.json /corpus/results/poseidon_equiv/${PM}_equiv$S_.json
echo POSEIDON_EQUIV_ALL_DONE | tee $OUT/done.txt
sync; sleep 30
