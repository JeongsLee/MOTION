"""Universal pretraining driver (foundationv2 P3/P4).

Streams the unified corpus -> bucketed batches -> UniversalTPADA collocation loss ->
Adam. Multi-family: families are interleaved so buckets fill across the corpus. Single
weight set over 2D+3D, grid+mesh. mGPU via MirroredStrategy when >1 GPU is visible
(per-replica batches drawn from the bucket stream). Checkpoints weights-only npz.

    python -m core.pretrain --families pretrain --steps 20000 --batch 8 --lr 3e-4
                            --save /corpus/results/univ_p4  [--dims2 128 --dims3 48]

`--families pretrain` = registry.pretrain_families(); or a comma list. Steady/mesh
downstream families are excluded unless named. Eval: periodic held-out val rel-L2.
"""
from __future__ import annotations

import argparse
import functools
import glob
import os
import re
import time

import numpy as np
import tensorflow as tf

from core.batching import bucketed_batches
from core.model import BasisTowerEnsemble, UniversalTPADA
from core.train_step import GEOM_MAX, batched_loss
from data import loader
from data.registry import FAMILIES, NUM_SLOTS, pretrain_families


def _interleave(families, split, t_in, n_colloc, seed, max_per_fam):
    """Round-robin across families, CYCLING each family's stream forever (re-shuffled per epoch via
    the seed) so NO family is ever dropped. The old version removed a family once its units ran out,
    which STARVED the data-poor families: airfrans (~800 units) exhausted around step 10k and was
    then never trained again through the long tail dominated by the big families (Poisson 20k,
    drivaernet 6.5k) — so it (and geofno/shapenet) progressively forgot and DIVERGED, with the
    divergence onset per family matching its exhaustion step. Cycling keeps every family trained for
    all steps and equalizes their gradient share. Infinite generator: the caller breaks on step."""
    def mk(i, ep):
        return loader.stream([families[i]], split, t_in=t_in, n_colloc=n_colloc,
                             seed=seed + i + 9973 * ep, max_per_fam=max_per_fam)
    gens = [mk(i, 0) for i in range(len(families))]
    ep = [0] * len(families)
    while True:
        for i in range(len(gens)):
            try:
                yield next(gens[i])
            except StopIteration:                                  # family exhausted -> reshuffle & continue
                ep[i] += 1
                gens[i] = mk(i, ep[i])
                try:
                    yield next(gens[i])
                except StopIteration:
                    pass                                           # empty family (no units) -> skip


class _WarmupConst(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear LR warmup over `warmup` steps to `lr`, then constant. Stabilises from-scratch
    transformer training (constant high LR from step 0 diverges/oscillates)."""

    def __init__(self, lr, warmup):
        self.lr = float(lr); self.warmup = max(1, int(warmup))

    def __call__(self, step):
        return self.lr * tf.minimum(1.0, tf.cast(step, tf.float32) / float(self.warmup))

    def get_config(self):
        return {"lr": self.lr, "warmup": self.warmup}


def save_ckpt(model, tvars, path, names=None):
    """names: optional parallel key list (ensemble towers share layer names in eager mode,
    so name-keyed saving would silently collide — tower-prefixed positional keys instead)."""
    keys = names if names else [v.name for v in tvars]
    np.savez_compressed(path, **{k: v.numpy() for k, v in zip(keys, tvars)})


def _try_np_pad(v, w, d_w, nm=None):
    """Warm-start across an INCREASE in the ADA time-panel count n_p. n_p is a weight dim (banks
    output width = d_w*n_p; synth gain_t = n_p), so a plain name-keyed load shape-mismatches when
    n_p grows. Pad instead: the banks outputs (…, d_w*n_p) reshape to (…, d_w, n_p) and the NEW
    panels get 0 (zero tendency coeff -> those basis functions contribute 0 -> ADA trajectory
    UNCHANGED at load); gain_t gets 1 (identity, irrelevant since the new-panel coeffs are 0).
    Returns the padded array or None if the mismatch is not an n_p growth."""
    vs, ws = tuple(int(x) for x in v.shape), tuple(int(x) for x in w.shape)
    nm = nm or v.name
    if "gain_t" in nm and len(vs) == 1 and len(ws) == 1 and vs[0] > ws[0]:
        out = np.ones(vs, w.dtype); out[:ws[0]] = w; return out          # time gain -> identity pad
    if d_w and len(vs) == len(ws) and vs[:-1] == ws[:-1] and vs[-1] > ws[-1] \
            and vs[-1] % d_w == 0 and ws[-1] % d_w == 0:
        np_new, np_old = vs[-1] // d_w, ws[-1] // d_w                    # out = d_w*n_p, panel-major last
        out = np.zeros((*vs[:-1], d_w, np_new), w.dtype)
        out[..., :np_old] = w.reshape(*ws[:-1], d_w, np_old)             # new panels stay 0 (inert)
        return out.reshape(vs)
    return None


def load_ckpt(tvars, path, d_w=None, names=None, strip_enum=False):
    """NAME-KEYED partial load: restore variables whose name matches the checkpoint; leave any
    NEW variable (e.g. a physics bank added after this ckpt was saved) at its init. Because new
    banks are zero-init gated, the model output is UNCHANGED at load and the new bank learns from
    zero — so training can continue from an earlier ckpt after inserting more banks. With d_w given,
    also PAD-loads across an n_p (ADA time-panel) increase (new panels 0 -> output unchanged).
    strip_enum: the ckpt was saved with enumerated positional keys ('m{i}|', 'tw{i}|', 'p{i}|') but
    is being loaded into a model whose var ORDER differs (e.g. warm-start a MoE+dual model from a
    MoE-only ckpt: core2/banks2 insert vars and shift every index). Remap the ckpt to a plain-name
    dict (strip the prefix) so matching is by v.name, not position. Source names are unique after
    strip; in the TARGET, core2 shares core's name -> both load the source core weights (h2=h1 boot)."""
    ck = np.load(path)
    if strip_enum:
        import re as _re
        src = {}
        for k in ck.files:
            src[_re.sub(r"^(m|tw|p)\d+\|", "", k)] = ck[k]
    else:
        src = ck
    loaded, padded, skipped_new, shape_mismatch, skipped_dead, basis_dup = 0, 0, 0, 0, 0, 0
    have = set(src.keys()) if strip_enum else set(ck.files)
    for j, v in enumerate(tvars):
        nm = names[j] if names else v.name
        if nm in have:
            w = src[nm]
            # DEAD-BANK REPAIR: mechanism head weights saved by the old zero-init x zero-gate
            # code are identically 0 (never trained). Loading them would overwrite the fresh
            # small-random init and re-create the gradient-dead saddle — keep the fresh init.
            # NO_DEAD_REPAIR=1 restores the checkpoint EXACTLY. The repair below is a training
            # affordance (a head saved as identically 0 keeps its fresh random init so it can
            # start learning); at EVAL it substitutes random values for 4 heads whose trained
            # gates are loaded as-is, which perturbs the very state we are trying to score.
            if ("banks_" in nm and "banks_base" not in nm and "gate" not in nm
                    and not np.any(w) and not os.environ.get("NO_DEAD_REPAIR")):
                skipped_dead += 1
                continue
            if tuple(w.shape) == tuple(v.shape):
                v.assign(w); loaded += 1
            elif (("banks_" in nm or "steady_o" in nm)
                  and tuple(w.shape[:-1]) == tuple(v.shape[:-1])
                  and int(v.shape[-1]) == 2 * int(w.shape[-1])):
                # PER-BASIS W split: old shared-W ckpt -> new 2x width [Fourier | Legendre].
                # Duplicate into both halves so each basis starts as the old shared content
                # (output exactly unchanged at load). MUST run before _try_np_pad, which would
                # misread the 2x width as an n_p growth and zero-pad (output-changing).
                v.assign(np.concatenate([w, w], axis=-1)); basis_dup += 1
            elif (len(v.shape) == len(w.shape) and int(v.shape[0]) > int(w.shape[0])
                  and tuple(v.shape[1:]) == tuple(w.shape[1:])):
                # INPUT-DIM GREW along axis 0 (e.g. GEOM_MAX widened 4->8: encoder fproj/geom_mlp
                # kernels gain new trailing input rows for the appended geom channels). Load the old
                # rows, ZERO the new ones -> the new geom channels contribute 0 at load (warm-safe;
                # the geom_gate/geom_mlp already-trained behavior on old channels is preserved).
                pad = np.zeros(v.shape, w.dtype)
                pad[:int(w.shape[0])] = w
                v.assign(pad); padded += 1
            else:
                pw = _try_np_pad(v, w, d_w, nm)
                if pw is not None:
                    v.assign(pw); padded += 1                            # n_p grew -> zero-pad new panels
                else:
                    shape_mismatch += 1
        else:
            skipped_new += 1
    print(f"[init_ckpt] loaded {loaded}, np-panel-padded {padded}, new-at-init {skipped_new}, "
          f"shape-mismatch {shape_mismatch}, dead-bank-reinit {skipped_dead}, "
          f"basis-dup {basis_dup} (zero gates + duplicate halves keep output unchanged at load)",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="pretrain")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--d_w", type=int, default=16)
    ap.add_argument("--n_p", type=int, default=32)
    ap.add_argument("--dims2", type=int, default=64)      # 2D latent box side
    ap.add_argument("--dims3", type=int, default=32)      # 3D latent box side
    ap.add_argument("--t_pad", type=float, default=1.1, help="ADA time-domain endpoint > 1: the last "
                    "supervised frame (t=1) maps INSIDE the basis interval (xi~0.82 not the high-"
                    "leverage endpoint xi=1), and (1, t_pad] is an unsupervised pad away from the "
                    "extrapolation edge. Hard IC (t=0 -> xi=-1) preserved.")
    ap.add_argument("--t_in", type=int, default=10)
    ap.add_argument("--n_colloc", type=int, default=2048)
    ap.add_argument("--enc_n", type=int, default=8192)
    ap.add_argument("--max_per_fam", type=int, default=0)
    ap.add_argument("--save", default="/corpus/results/univ_p4")
    ap.add_argument("--ckpt_every", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init_ckpt", default="", help="warm-start (name-keyed partial load; "
                    "new zero-init banks resume from zero -> output unchanged)")
    ap.add_argument("--xla", type=int, default=0, help="jit_compile the grad step (XLA fusion: "
                    "faster + better buffer planning)")
    ap.add_argument("--encoder", default="attn", choices=["attn", "bin"],
                    help="bin = GINO cell-binning (O(N), scalable, preserves fine mesh geometry)")
    ap.add_argument("--rec_steps", type=int, default=1, help="unsteady recursion depth "
                    "(combo-rec state feedback; 1 = single-shot)")
    ap.add_argument("--towers", default="", help="basis-tower ensemble: comma list of variants — "
                    "'spec' (V-spec Fourier), 'ada' (V-int Fourier-ADA), 'ftime' (Fourier-ADA "
                    "TIME, V-spec space), e.g. 'spec,ada,ftime'. Empty = single model "
                    "(variant from --spatial_ada/--time_basis).")
    ap.add_argument("--tower_ckpts", default="", help="comma list of per-tower warm-start ckpts "
                    "(parallel to --towers; single-model ckpts, loaded positionally per tower)")
    ap.add_argument("--spatial_ada", type=int, default=1, help="single-model spatial Fourier variant")
    ap.add_argument("--time_basis", default="legendre", choices=["legendre", "fourier"],
                    help="single-model time-axis basis (fourier = oscillatory-time ADA dictionary)")
    ap.add_argument("--spatial_basis", default="global", choices=["global", "local", "localdec"],
                    help="single-model spatial basis (local = patch-Legendre; localdec = Gaussian-kernel decoder)")
    ap.add_argument("--n_patches", type=int, default=4, help="local basis: patches per axis (must divide dims)")
    ap.add_argument("--patch_order", type=int, default=3, help="local basis: Legendre order per patch")
    ap.add_argument("--arch", default="tower", choices=["tower", "particle"],
                    help="particle = Perceiver latent-particle encoder + Transformer decoder (from scratch)")
    ap.add_argument("--n_particles", type=int, default=256, help="particle arch: number of latent particles")
    ap.add_argument("--warmup", type=int, default=0, help="linear LR warmup steps (0=off); needed for from-scratch transformer")
    ap.add_argument("--active_banks", default="", help="comma list of mechanism banks to KEEP active "
                    "(others hard-off); empty=all. Steady finetune: e.g. 'diffusion,shear'")
    ap.add_argument("--init_prefix", default="", help="prefix prepended to var names when matching the "
                    "init ckpt (e.g. 'tw0|' to load one tower of an ensemble ckpt into a single model)")
    ap.add_argument("--moe_experts", type=int, default=0, help="core-MLP MoE experts (0=dense). LLM-style capacity")
    ap.add_argument("--moe_topk", type=int, default=2, help="MoE top-k experts active per cell")
    ap.add_argument("--moe_aux_w", type=float, default=1e-2, help="load-balance aux-loss weight")
    ap.add_argument("--moe_upcycle", type=int, default=0, help="1 = sparse-upcycle: copy dense core-MLP "
                    "into every expert (+noise) from init_ckpt (name-remap _e{e}); router new-at-init")
    ap.add_argument("--dual_core", type=int, default=0, help="1 = parallel orthogonal second core h2 "
                    "feeding the banks via zero-init ungated heads (warm-safe decomposed hidden bank)")
    ap.add_argument("--ortho_w", type=float, default=1e-2, help="decorrelation aux weight (h1 _|_ h2)")
    ap.add_argument("--sl_steps", type=int, default=0, help="semi-Lagrangian latent sub-steps: split "
                    "[0,1] into N sub-windows, re-evaluate the banks (advection) at each evolved state "
                    "so the operator follows the pathline (targets NS/turbulence). 0/1 = single-window")
    ap.add_argument("--init_strip_enum", type=int, default=0, help="warm-start across a structure change: "
                    "strip enumerated ckpt prefixes (m{i}|/tw{i}|/p{i}|) and match by plain name (e.g. "
                    "MoE+dual model from a MoE-only ckpt, where core2/banks2 shift positional indices)")
    ap.add_argument("--steady_mode", default="ada", choices=["ada", "field"],
                    help="ada = steady solved as a t->inf ADA RELAXATION trajectory (QoI 0 -> solution "
                    "at t=1, source/geometry as fed forcing; unifies steady+unsteady under ONE operator "
                    "and mirrors pseudo-transient continuation, the real steady-RANS method). "
                    "field = separate SteadyField elliptic head (bypasses the ADA time integration).")
    a = ap.parse_args()

    fams = (pretrain_families() if a.families == "pretrain"
            else [FAMILIES[n] for n in a.families.split(",")])
    fams = [f.name for f in fams] if hasattr(fams[0], "name") else fams
    # skip families with no split manifest yet (data not staged / adapter deferred, e.g. geofno_pipe)
    split_dir = os.path.join(os.environ.get("CORPUS_ROOT", "/corpus"), "meta", "splits")
    keep = [f for f in fams if os.path.exists(os.path.join(split_dir, f"{f}.json"))]
    dropped = [f for f in fams if f not in keep]
    if dropped:
        print(f"SKIP (no manifest): {dropped}", flush=True)
    fams = keep
    os.makedirs(a.save, exist_ok=True)
    box = {2: (a.dims2, a.dims2), 3: (a.dims3, a.dims3, a.dims3)}
    print(f"pretrain: {len(fams)} families, box {box}, batch {a.batch}, "
          f"d={a.d} depth={a.depth} d_w={a.d_w} n_p={a.n_p} -> {a.save}", flush=True)

    # 2-GPU (or N-GPU) DATA-PARALLEL: MirroredStrategy replicates the model across visible GPUs;
    # each replica runs the graph on its own sample slice and gradients all-reduce in apply_gradients.
    # n_rep==1 (single GPU) is a no-op passthrough (identical to before).
    # ReductionToOneDevice avoids NCCL (unreliable in this container: "unhandled cuda error" on
    # CollectiveReduce); it copies grads to one GPU, reduces, broadcasts — robust for 2-8 GPU.
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.ReductionToOneDevice())
    n_rep = strategy.num_replicas_in_sync
    print(f'[dist] {n_rep} replica(s)', flush=True)

    # tower-variant -> (spatial_ada, time_basis, spatial_basis)
    VARIANTS = {"spec":     (False, "legendre", "global"),   # V-spec Fourier + Legendre time
                "ada":      (True,  "legendre", "global"),   # V-int Fourier-ADA + Legendre time
                "ftime":    (False, "fourier",  "global"),   # V-spec Fourier + Fourier-ADA time
                "local":    (False, "legendre", "local"),    # patch-Legendre compact-support
                "localdec": (False, "legendre", "localdec")} # local Gaussian-kernel decoder (v56aq-style)

    active_banks = [s.strip() for s in a.active_banks.split(",")] if a.active_banks else None

    def _mk_tower(spatial_ada, time_basis="legendre", spatial_basis="global"):
        return UniversalTPADA(box_cfgs=box, d=a.d, depth=a.depth, d_w=a.d_w, n_p=a.n_p,
                              c_out=NUM_SLOTS, t_final=a.t_pad, encoder=a.encoder,
                              rec_steps=a.rec_steps, spatial_ada=spatial_ada, time_basis=time_basis,
                              spatial_basis=spatial_basis, n_patches=a.n_patches,
                              patch_order=a.patch_order, active_banks=active_banks,
                              moe_experts=a.moe_experts, moe_topk=a.moe_topk,
                              dual_core=bool(a.dual_core), sl_steps=a.sl_steps)

    with strategy.scope():
        if a.arch == "particle":
            from core.particle_model import ParticleTPADA
            model = ParticleTPADA(box_cfgs=box, d=a.d, n_particles=a.n_particles, d_w=a.d_w,
                                  n_p=a.n_p, t_final=a.t_pad, c_out=NUM_SLOTS, depth=a.depth)
            print(f"arch=particle: N={a.n_particles} d={a.d} depth={a.depth}", flush=True)
        elif a.towers:
            variants = [v.strip() for v in a.towers.split(",")]
            model = BasisTowerEnsemble([_mk_tower(*VARIANTS[v]) for v in variants])
            print(f"ensemble: {len(variants)} towers {variants} + family gate", flush=True)
        else:
            model = _mk_tower(bool(a.spatial_ada), a.time_basis, a.spatial_basis)
        lr_sched = _WarmupConst(a.lr, a.warmup) if a.warmup > 0 else a.lr
        opt = tf.keras.optimizers.Adam(lr_sched, clipnorm=1.0)

        def tower_tvars(m):
            vs = (m.core.trainable_weights
                  + (m.core2.trainable_weights if getattr(m, "dual", False) else [])
                  + m.penc.trainable_weights
                  + m.grid_proj.trainable_weights + m.banks.trainable_weights
                  + m.steady.trainable_weights + m.dec.trainable_weights
                  + m.rec_proj.trainable_weights                     # empty until rec_steps>1
                  + [g for s in m.synth.values() for g in (s.gains or [])]
                  + [g for s in m.synth.values() for g in (s.gains_leg or [])]      # dual-basis
                  + [v for s in m.synth.values() for v in getattr(s, "kern_vars", [])])  # localdec sigma
            if getattr(m, "local_heads", False):                     # per-patch independent W heads
                vs = vs + [w for ph in m.pheads for w in ph.trainable_weights] + [m.g_extra]
            return vs

        def tvars():
            if a.arch == "particle":
                return model.trainable_weights_flat()
            if isinstance(model, BasisTowerEnsemble):
                vs = [v for t in model.towers for v in tower_tvars(t)]
                return vs + model.gate.trainable_weights
            return tower_tvars(model)

        def ckpt_names():
            """Save/load keys. Ensemble towers share eager layer names -> tower-prefixed positional
            keys ('tw{i}|<var name>'). Per-tower var lists (towers may differ, e.g. a local tower has
            extra per-patch heads), so iterate each tower's OWN tvars. Gate keys unprefixed."""
            if isinstance(model, BasisTowerEnsemble):
                keys = [f"tw{i}|{v.name}" for i, t in enumerate(model.towers)
                        for v in tower_tvars(t)]
                return keys + [v.name for v in model.gate.trainable_weights]
            if a.arch == "particle":
                # particle var names can collide (repeated Keras sublayers) -> enumerate positionally
                return [f"p{i}|{v.name}" for i, v in enumerate(tvars())]
            if a.moe_experts > 0 or a.dual_core:     # MoE / dual-core add sublayers whose names can collide
                return [f"m{i}|{v.name}" for i, v in enumerate(tvars())]
            return None

        def batches(split, seed):
            it = _interleave(fams, split, a.t_in, a.n_colloc, seed, a.max_per_fam or None)
            bs = n_rep if n_rep > 1 else a.batch      # data-parallel: exactly one sample per replica
            for b in bucketed_batches(it, bs, NUM_SLOTS, enc_n=a.enc_n, seed=seed):
                if n_rep > 1 and int(np.asarray(b["coords_q"]).shape[0]) != n_rep:
                    continue                          # drop partial batch (split needs exactly n_rep)
                yield b

        # build variables with one eager forward, then wrap the step in tf.function (GRAPH MODE):
        # graph execution reuses buffers across steps, eliminating the eager BFC fragmentation that
        # killed every earlier run at a deterministic step (~450/900). K/steady/roles are python args
        # so tf.function traces once per (K, steady) mode — a handful of stable graphs.
        # instantiate EVERY variable (both K, both steady/unsteady paths) BEFORE tf.function —
        # tf.function forbids creating variables mid-trace, and SteadyField/K-specific vars won't
        # exist unless their path is exercised once eagerly.
        F = 10 * NUM_SLOTS + 10 + GEOM_MAX                             # build_feats fixed width (GEOM_MAX geom cols)
        for K in sorted(model.synth):
            cn = tf.zeros([1, a.enc_n, K]); ft = tf.zeros([1, a.enc_n, F])
            cq = tf.zeros([1, 8, K])
            op1 = tf.ones([1, 13])
            h = model.encode(cn, ft, K, roles=list(range(K)))
            _ = model.steady_field_at(h, K, cq, op_multihot=op1)       # SteadyField (+gate) vars
            A, A_leg, _ = model.from_points(cn, ft, K, roles=list(range(K)),
                                            op_multihot=op1)           # banks + synth[K] gains(+leg)
            gqz = tf.zeros([1, 8, GEOM_MAX])                           # per-query geom (build geom-FiLM vars)
            _ = model.point_traj_perquery_g(A, A_leg, K, cq, tf.zeros([1, 8, model.n_p]),
                                            tf.zeros([1, 8, NUM_SLOTS]), op_multihot=op1, geom_q=gqz)
            if a.sl_steps > 1:                                          # build rec_proj + SL path vars
                Al, Aleg, _ = model.from_points_sl(cn, ft, K, roles=list(range(K)), op_multihot=op1)
                gqs = tf.zeros([1, 8, a.sl_steps, model.n_p])
                _ = model.point_traj_sl_g(Al, Aleg, K, cq, gqs, tf.zeros([1, 8, model.n_p]),
                                          tf.zeros([1, 8, NUM_SLOTS]), geom_q=gqz)
        V = tvars()
        NAMES = ckpt_names()
        if a.tower_ckpts and isinstance(model, BasisTowerEnsemble):
            # sparse-upcycling init: each tower loads a SINGLE-MODEL ckpt positionally via ITS OWN
            # variable names (a local tower has extra per-patch heads absent from the ckpt -> those
            # stay new-at-init; g_extra zero-init keeps the branch OFF = warm-safe). Gate uniform 1/N.
            for i, (t, p) in enumerate(zip(model.towers, a.tower_ckpts.split(","))):
                if p and os.path.exists(p):
                    print(f"[tower {i}] <- {p}", flush=True)
                    load_ckpt(tower_tvars(t), p, d_w=a.d_w, names=[v.name for v in tower_tvars(t)])
        elif a.init_ckpt and os.path.exists(a.init_ckpt) and a.moe_upcycle:
            # SPARSE-UPCYCLE: dense core-MLP -> every expert. Expert var 'lbcore_ml{i}_e{e}_h/..' loads
            # the dense 'lbcore_ml{i}_h/..' (strip '_e{e}') + tiny noise (symmetry break); router is new.
            import re as _re
            ck = np.load(a.init_ckpt); have = set(ck.files); ld = up = nw = 0
            for v in V:
                src = _re.sub(r"_e\d+", "", v.name)                   # expert -> dense source name
                key = a.init_prefix + src
                if key in have and tuple(ck[key].shape) == tuple(v.shape):
                    w = ck[key]
                    if src != v.name:                                 # an expert copy -> add symmetry noise
                        w = w + np.random.normal(0, 1e-3, w.shape).astype(w.dtype); up += 1
                    else:
                        ld += 1
                    v.assign(w)
                else:
                    nw += 1
            print(f"[upcycle] dense-loaded {ld}, expert-copied {up}, new (router/etc) {nw}", flush=True)
        elif a.init_ckpt and os.path.exists(a.init_ckpt):
            # --init_prefix: load one tower of an ensemble ckpt into a single model (keys = prefix+name)
            if a.init_prefix:
                names = [a.init_prefix + v.name for v in V]
            elif a.dual_core:
                # dual-core load. core & core2 share var NAMES (LatentBoxCore has a fixed name), so:
                #  * warm-start from a PRE-DUAL plain-name ckpt (e.g. v28): match by plain v.name -> core
                #    loads, and core2 (same name) ALSO loads v28's core weights = free h2=h1 bootstrap;
                #    banks2_* zero-init heads are absent -> new-at-init -> h2 contributes 0 (warm-safe).
                #  * resume from a DUAL ckpt (saved with enumerated m{i}| keys): use those positional keys.
                _have = set(np.load(a.init_ckpt).files)
                names = NAMES if (NAMES and NAMES[0] in _have) else [v.name for v in V]
            else:
                names = NAMES
            # strip_enum: warm-start across a structure change (MoE+dual from a MoE-only ckpt) -> match by
            # plain v.name against the enum-stripped ckpt (core2 shares core's name -> h2=h1 bootstrap).
            load_ckpt(V, a.init_ckpt, d_w=a.d_w,
                      names=([v.name for v in V] if a.init_strip_enum else names),
                      strip_enum=bool(a.init_strip_enum))
            if a.sl_steps > 1:
                # SEMI-LAGRANGIAN warm-safety: the telescoping recovers the single-window trajectory ONLY
                # if rec_proj = 0 at load (every sub-segment identical). A warm-start ckpt trained with
                # rec_steps>1 has a NONZERO rec_proj -> reset it to zero so the SL model starts EXACTLY at
                # the loaded single-window behavior, then learns the sub-step feedback from zero.
                nz = 0
                for v in V:
                    if "rec_fb" in v.name and np.any(v.numpy()):
                        v.assign(np.zeros(v.shape, v.dtype.as_numpy_dtype)); nz += 1
                print(f"[sl] rec_proj zeroed ({nz} tensors) -> telescoping warm-safe at load", flush=True)

    @tf.function(reduce_retracing=True, jit_compile=bool(a.xla))
    def _grad_step(coords_node, feats, coords_q, geom_q, ic_q, Gq, Gq_seg, y_q, cmask, op_multihot, K, steady, roles):
        with tf.GradientTape() as tape:
            if steady and a.steady_mode == "field":
                h = model.encode(coords_node, feats, K, roles=roles)
                pred = model.steady_field_at(h, K, coords_q, op_multihot=op_multihot)
            elif a.sl_steps > 1 and not steady:                   # SEMI-LAGRANGIAN: transient only
                A_list, A_leg, _ = model.from_points_sl(coords_node, feats, K, roles=roles,
                                                        op_multihot=op_multihot)
                pred = model.point_traj_sl_g(A_list, A_leg, K, coords_q, Gq_seg, Gq, ic_q, geom_q=geom_q)
            else:                                                  # ADA relaxation (steady=drop transient banks)
                A, A_leg, _ = model.from_points(coords_node, feats, K, roles=roles,
                                                op_multihot=op_multihot, steady=steady)
                pred = model.point_traj_perquery_g(A, A_leg, K, coords_q, Gq, ic_q,
                                                   op_multihot=op_multihot, geom_q=geom_q)  # ic=0,t=1 steady
            w = cmask[:, None]
            num = tf.sqrt(tf.reduce_sum(w * tf.square(pred - y_q), axis=[1, 2]) + 1e-12)
            den = tf.sqrt(tf.reduce_sum(w * tf.square(y_q), axis=[1, 2]) + 1e-12)
            loss = tf.reduce_mean(num / den)
            if a.moe_experts > 0:                              # MoE load-balance aux (router anti-collapse)
                loss = loss + a.moe_aux_w * model.moe_aux()
            if a.dual_core:                                    # decorrelate the two hidden boxes (h1 _|_ h2)
                loss = loss + a.ortho_w * model.ortho_aux()
        grads = tape.gradient(loss, V)
        grads = [g if g is not None else tf.zeros_like(v) for g, v in zip(grads, V)]
        opt.apply_gradients(zip(grads, V))
        return loss

    @tf.function(reduce_retracing=True, jit_compile=bool(a.xla))
    def _fwd_loss(coords_node, feats, coords_q, geom_q, ic_q, Gq, Gq_seg, y_q, cmask, op_multihot, K, steady, roles):
        if steady and a.steady_mode == "field":
            h = model.encode(coords_node, feats, K, roles=roles)
            pred = model.steady_field_at(h, K, coords_q, op_multihot=op_multihot)
        elif a.sl_steps > 1 and not steady:
            A_list, A_leg, _ = model.from_points_sl(coords_node, feats, K, roles=roles,
                                                    op_multihot=op_multihot)
            pred = model.point_traj_sl_g(A_list, A_leg, K, coords_q, Gq_seg, Gq, ic_q, geom_q=geom_q)
        else:                                                      # ADA relaxation (steady=drop transient banks)
            A, A_leg, _ = model.from_points(coords_node, feats, K, roles=roles,
                                            op_multihot=op_multihot, steady=steady)
            pred = model.point_traj_perquery_g(A, A_leg, K, coords_q, Gq, ic_q,
                                               op_multihot=op_multihot, geom_q=geom_q)
        w = cmask[:, None]
        num = tf.sqrt(tf.reduce_sum(w * tf.square(pred - y_q), axis=[1, 2]) + 1e-12)
        den = tf.sqrt(tf.reduce_sum(w * tf.square(y_q), axis=[1, 2]) + 1e-12)
        return tf.reduce_mean(num / den)

    def _args(b):
        K = int(b["K"]); steady = bool(b.get("steady")); roles = tuple(int(r) for r in b["roles"])
        use_ada = (not steady) or (a.steady_mode == "ada")         # steady-as-relaxation also needs Gq(t=1)
        Gq = (model.synth[K].query_time_map(b["t_q"]) if use_ada
              else tf.zeros([b["coords_q"].shape[0], b["coords_q"].shape[1], 1], tf.float32))
        if a.sl_steps > 1 and use_ada:                             # per-segment time-map differences
            tq = np.asarray(b["t_q"], np.float64)                  # (B,Q)
            ts = np.linspace(0.0, 1.0, a.sl_steps + 1)
            segs = [model.synth[K].query_time_map(np.minimum(tq, ts[k + 1]))
                    - model.synth[K].query_time_map(np.minimum(tq, ts[k])) for k in range(a.sl_steps)]
            Gq_seg = tf.stack(segs, axis=2)                        # (B,Q,N,Np)
        else:
            Gq_seg = Gq[:, :, None, :]                             # dummy (unused when sl_steps<=1)
        return (tf.constant(b["coords_node"]), tf.constant(b["feats"]), tf.constant(b["coords_q"]),
                tf.constant(b["geom_q"]), tf.constant(b["ic_q"]), Gq, Gq_seg, tf.constant(b["y_q"]),
                tf.constant(b["cmask"]), tf.constant(b["op_multihot"]), K, steady, roles)

    def _split(t, ctx):                                            # slice a batch tensor for one replica
        per = t.shape[0] // n_rep
        i = ctx.replica_id_in_sync_group
        return t[i * per:(i + 1) * per]

    @tf.function(reduce_retracing=True)
    def _dist_step(dcn, dft, dcq, dgq, dic, dGq, dGqs, dyq, dcm, dop, K, steady, roles):
        pr = strategy.run(_grad_step, args=(dcn, dft, dcq, dgq, dic, dGq, dGqs, dyq, dcm, dop,
                                            K, steady, roles))
        return strategy.reduce(tf.distribute.ReduceOp.MEAN, pr, axis=None)

    def run_batch(b):
        args = _args(b)
        if n_rep <= 1:
            return _grad_step(*args)                               # single GPU: direct
        *targs, K, steady, roles = args                            # split the batch across replicas
        dist = tuple(strategy.experimental_distribute_values_from_function(functools.partial(_split, t))
                     for t in targs)
        return _dist_step(*dist, K, steady, roles)

    globals()["_EVAL_FWD"] = lambda b: float(_fwd_loss(*_args(b)))     # graph eval (single-device)

    step = 0
    t0 = time.time()
    ema = None
    best_ca = float("inf")
    while step < a.steps:
        for b in batches("train", a.seed + step):
            L = float(run_batch(b))
            ema = L if ema is None else 0.98 * ema + 0.02 * L
            step += 1
            if step % 50 == 0:
                print(f"step {step} loss {L:.4f} ema {ema:.4f} "
                      f"({step/(time.time()-t0):.2f} it/s)", flush=True)
            if step % a.eval_every == 0:
                ca = _evaluate(model, fams, a)
                if ca == ca and ca < best_ca:                      # not-nan and improved -> best ckpt
                    best_ca = ca
                    save_ckpt(model, V, os.path.join(a.save, "ckpt_best.npz"), names=NAMES)
                    print(f"  [ckpt] new best class-avg {ca*100:.2f}% -> ckpt_best.npz", flush=True)
            if step % a.ckpt_every == 0:
                p = os.path.join(a.save, f"ckpt_{step}.npz")
                save_ckpt(model, V, p, names=NAMES)
                # keep the newest 3 NUMBERED ckpts (crash-resilience) — sort by STEP NUMBER, not
                # lexicographically (ckpt_8000 would outrank ckpt_26000). ckpt_best.npz is preserved
                # separately (excluded from the glob via the \d+ match) since late training can OVERFIT
                # the data-poor mesh families and the last ckpt is not the best one.
                cks = [q for q in glob.glob(os.path.join(a.save, "ckpt_*.npz"))
                       if re.search(r"ckpt_(\d+)\.npz$", q)]
                cks = sorted(cks, key=lambda q: int(re.search(r"ckpt_(\d+)\.npz$", q).group(1)))
                for old in cks[:-3]:
                    os.remove(old)
                print(f"[ckpt] {p}", flush=True)
            if step >= a.steps:
                break
    save_ckpt(model, tvars(), os.path.join(a.save, "ckpt_final.npz"), names=NAMES)
    print("PRETRAIN_DONE", flush=True)


def _evaluate(model, fams, a, n=16):
    """Held-out val class-avg rel-L2. Uses the SAME bounded-node batching (size 1) as training
    so 3D-grid val examples don't blow memory."""
    fwd = globals().get("_EVAL_FWD")
    per = {}
    for f in fams:
        it = loader.stream([f], "val", t_in=a.t_in, n_colloc=a.n_colloc, seed=7, max_per_fam=n)
        errs = [(fwd(b) if fwd else float(batched_loss(model, b)))    # graph eval avoids eager OOM
                for b in bucketed_batches(it, 1, NUM_SLOTS, enc_n=a.enc_n, seed=7)]
        if errs:
            per[f] = float(np.mean(errs))
    ca = float(np.mean(list(per.values()))) if per else float("nan")
    print(f"  [EVAL] class-avg rel-L2 {ca*100:.2f}%  | "
          + "  ".join(f"{k}={v*100:.1f}" for k, v in per.items()), flush=True)
    return ca


if __name__ == "__main__":
    main()
