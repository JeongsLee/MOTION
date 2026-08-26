"""Tier-0 foundation sanity: pretrain on several synthetic families, then few-shot transfer
to a held-out family. Compares ours-pretrained vs ours-from-scratch over a data-budget sweep.

    python -m motion_tf.train.train_foundation motion_tf.train.configs.foundation_synth

Success = at low N, the pretrained (finetuned) model beats from-scratch → pretraining transfers
in our architecture (the foundation premise). FNO-scratch is table-stakes and omitted here;
the meaningful Tier-0 ablation is ours-pretrained vs ours-scratch.
"""
from __future__ import annotations
import importlib
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf

for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)

from ..model import PhysicsOperatorMixture
from ..baselines.fno2d import FNO2D
from ..data import families as fam
from ..utils import ckpt


def rel_l2(pred, ref, eps=1e-6):
    num = tf.reduce_sum((pred - ref) ** 2, axis=[2, 3])
    den = tf.reduce_sum(ref ** 2, axis=[2, 3]) + eps
    return tf.reduce_mean(num / den)


def make_step(model, opt, lam, jit, kind="ours"):
    @tf.function(jit_compile=jit)
    def step(u, desc, coef):
        with tf.GradientTape() as tape:
            if kind == "fno":
                loss = rel_l2(model.call(u[:, 0], desc, coef), u)
            else:
                pred, alpha = model.call_with_gate(u[:, 0], desc, coef)
                loss = rel_l2(pred, u) + lam * tf.reduce_mean(tf.reduce_sum(alpha, -1))
        g = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(g, model.trainable_variables))
        return loss
    return step


def eval_test(model, u_te, coef_te):
    desc = fam.make_descriptor(coef_te)
    errs, n = [], u_te.shape[0]
    for i in range(0, n, 32):
        ub = u_te[i:i + 32]
        p = model.call(tf.constant(ub[:, 0]),
                       tf.constant(desc[i:i + 32]), tf.constant(coef_te[i:i + 32]))
        errs.append(float(rel_l2(p, tf.constant(ub))) * ub.shape[0])
    return sum(errs) / n


def train_on(cfg, u_tr, c_tr, steps, jit, init_ckpt=None, lr=None, kind="ours"):
    if kind == "fno":
        model = FNO2D(cfg)
    else:
        model = PhysicsOperatorMixture(cfg)
        if init_ckpt:
            ckpt.load(model, init_ckpt)
    opt = tf.keras.optimizers.Adam(lr if lr is not None else cfg.lr)
    step = make_step(model, opt, float(getattr(cfg, "lambda_gate", 0.0)), jit, kind=kind)
    N = u_tr.shape[0]
    if steps > 0 and N > 0:
        d_tr = tf.constant(fam.make_descriptor(c_tr)); c = tf.constant(c_tr); u = tf.constant(u_tr)
        bs = min(cfg.batch, N)
        for s in range(steps):
            idx = np.random.choice(N, bs, replace=False)
            step(tf.gather(u, idx), tf.gather(d_tr, idx), tf.gather(c, idx))
    return model


def main(cfg_path):
    cfg = importlib.import_module(cfg_path).Cfg
    jit = bool(getattr(cfg, "jit_compile", True))
    print(f"=== Tier-0 foundation | pretrain={cfg.pretrain_families} → transfer={cfg.heldout_family} "
          f"| Nx={cfg.Nx} | GPU={tf.config.list_physical_devices('GPU')} ===", flush=True)

    # ---- pretrain on several families (reuse existing ckpt if present) ----
    if getattr(cfg, "reuse_ckpt", False) and os.path.exists(cfg.ckpt):
        print(f"reusing existing pretrain ckpt → {cfg.ckpt}", flush=True)
    else:
        pre = fam.generate_multi(cfg.pretrain_families, cfg.n_per_pretrain,
                                 Nx=cfg.Nx, Nt=cfg.Nt, T=cfg.T_final, seed=1)
        print(f"pretrain data: {pre['u'].shape} ({cfg.pretrain_families})", flush=True)
        pre_model = train_on(cfg, pre["u"], pre["coeffs"], cfg.pretrain_steps, jit)
        ckpt.save(pre_model, cfg.ckpt)
        npar = int(sum(np.prod(v.shape) for v in pre_model.trainable_variables))
        print(f"pretrained ({npar:,} params) saved → {cfg.ckpt}", flush=True)
        del pre_model

    # ---- held-out family data ----
    ho = fam.generate(cfg.heldout_family, cfg.n_pool + cfg.n_test,
                      Nx=cfg.Nx, Nt=cfg.Nt, T=cfg.T_final, seed=42)
    u_pool, c_pool = ho["u"][:cfg.n_pool], ho["coeffs"][:cfg.n_pool]
    u_te, c_te = ho["u"][cfg.n_pool:], ho["coeffs"][cfg.n_pool:]
    print(f"held-out {cfg.heldout_family}: pool {u_pool.shape}  test {u_te.shape}\n", flush=True)

    # ---- ours-pretrained vs ours-scratch vs FNO-scratch over N ----
    print(f"{'N':>5s} | {'ours-pre':>11s} | {'ours-scr':>11s} | {'FNO-scr':>11s} | "
          f"{'pre/FNO':>8s}", flush=True)
    for N in cfg.N_list:
        u_tr, c_tr = u_pool[:N], c_pool[:N]
        ft_lr = float(getattr(cfg, "finetune_lr", cfg.lr))
        ft_steps = 0 if N == 0 else cfg.finetune_steps     # N=0 → zero-shot (no finetune)
        sc_steps = 0 if N == 0 else cfg.scratch_steps       # N=0 → random-init floor
        m_pre = train_on(cfg, u_tr, c_tr, ft_steps, jit, init_ckpt=cfg.ckpt, lr=ft_lr)
        e_pre = eval_test(m_pre, u_te, c_te); del m_pre
        m_scr = train_on(cfg, u_tr, c_tr, sc_steps, jit, lr=cfg.lr)
        e_scr = eval_test(m_scr, u_te, c_te); del m_scr
        m_fno = train_on(cfg, u_tr, c_tr, sc_steps, jit, lr=cfg.lr, kind="fno")
        e_fno = eval_test(m_fno, u_te, c_te); del m_fno
        gfno = e_fno / e_pre if e_pre > 0 else float("nan")
        tag = " (zero-shot)" if N == 0 else ""
        print(f"{N:5d} | {e_pre:11.4e} | {e_scr:11.4e} | {e_fno:11.4e} | {gfno:7.2f}×{tag}",
              flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "motion_tf.train.configs.foundation_synth")
