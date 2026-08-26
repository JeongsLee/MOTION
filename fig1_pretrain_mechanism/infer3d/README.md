# 3D inference-only ensemble — where the code actually lives

The canonical 3D ensemble numbers (rot24, flips, the Walrus head-to-head) do **not** come
from the training tree. They come from a separate in-graph entry point that is embedded,
base64-encoded, inside the VESSL job command. It is not checked into `foundationv2/`, which
is why it keeps getting lost.

## The file

`infer3d_graph.py` (08-14) — recovered from `job-wbe42wqzpzfb` ("m2ft-canon") and copied here.

Everything except the example *builder* is the model's own code (trunk, decoder, readout are
imported, not reimplemented). The point draw, the gather, the normalization **and the group
action** all happen inside the graph, so every replica draws its own points while the trunk
sees one fused batch.

Measured on m2s2, 24 replicas, 32³ box, H100 — the reason this file exists:

| variant | host builds | time | per replica |
|---|---|---|---|
| `IT=0` volume rotated per element | 24 | 190.4 s | 7.93 s |
| `IT=2` no volume rotation | 24 | 57.5 s | 2.40 s (XLA) |
| `IT=1` no volume rotation | 1 | 37.4 s | 1.56 s (XLA) |

`IT=1` is fastest but shares ONE point set across replicas, which costs accuracy
(s0 rel-L2 8.72 % vs 7.03 % when each replica draws its own points). `infer3d_graph.py`
removes the host build without removing that diversity.

## How to recover it if this copy is lost again

```bash
vesslctl job export job-wbe42wqzpzfb > j.json
# the command contains: echo <b64> | base64 -d > /tmp/j.sh; bash /tmp/j.sh
# and j.sh in turn contains: echo <b64> | base64 -d > /tmp/fv2/infer3d_graph.py
```

## The canonical invocation (m2, Mach 1.0, 8 held-out trajectories)

```bash
cp -r /code-vol/motion_fv2 /tmp/fv2; cd /tmp/fv2
cp /code-vol/motion_fv2_phys/{train.py,model.py,data_ar.py} v3/
cp /code-vol/motion_fv2_phys/registry.py data/
cp /code-vol/motion_fv2_phys/pretrain.py core/

FAM=pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08 \
ARCH_D=896 ARCH_DEPTH=12 ARCH_DW=48 \
ENS=rot24 XLA=1 BF16=1 CHUNK_R=24 QGRID=32 BOX=32,32,32 \
SAVE_PRED=1 N_SAMP=8 SKIP=0 \
UIDS=0:90,0:91,0:92,0:93,0:94,0:95,0:96,0:97 \
OUT_TAG=m2ft_canon \
python3 -u infer3d_graph.py /corpus/results/ft3d_m2_data/ckpt_best.npz /corpus/results/_infer3d
```

`ARCH_*` are the model dims: **m2 = 896 / 12 / 48**, **m1 = 384 / 8 / 16** (the defaults in
`fig1_ens3d_v3.py` are m1's, which is a standing trap).

## Related harnesses, and what each is for

| file | role |
|---|---|
| `infer3d_graph.py` | in-graph ensemble, the fast canonical path (this directory) |
| `foundationv2/fig1_ens3d_v3.py` | host-build batched ensemble; the older, slower path |
| `foundationv2/v3/wens.py` | the *convention*: subsample axis averages in W, symmetry axis in field space |
| `group_size_curve.py`, `eps_equiv_motion.py` | CPU-only post-hoc: read `SAVE_PERG` per-transform predictions, any subgroup average is a mean over stored keys |

## Two facts that are easy to lose

1. **The symmetry axis has never been measured in W-space.** `wens.py` states
   "latents are NOT transform-covariant" and defers rotations ("role re-wiring") as a
   *premise*, not a result. The manuscript repeats it as a causal claim.
2. **A W-space symmetry average cannot be recovered from stored predictions.** Averaging W
   changes the autoregressive feedback, so the trajectory diverges from every per-element
   rollout; it needs a fresh rollout. The `SAVE_PERG` trick only accelerates *field-space*
   subgroup averages.
