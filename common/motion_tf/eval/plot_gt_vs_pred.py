"""Qualitative GT-vs-prediction figure: one test example per family, recursive model (combo-rec).
6 rows (families) x 3 cols (GT | pred | |err|) at the last valid future frame, primary active channel.
Writes PNG to $OUT_DIR (default /out). Based on eval_prose_exact.py (physical-space, geom_mask, EVAL_EB).
"""
from __future__ import annotations
import importlib, os, sys
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..model import PhysicsOperatorMixture
from ..utils import ckpt as ckptlib
from ..data import stream as streammod


def main(cfg_path):
    cfg = importlib.import_module(cfg_path).Cfg
    if bool(getattr(cfg, "bf16", False)):
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
    cache_dir = os.environ.get("PREBUILT_DIR") or os.path.join(
        os.environ.get("PROSE_DATA_DIR", "/code-vol/data/prose"), "prebuilt")
    _, test = streammod.make_train_stream(cfg, 8, cache_dir)
    model = PhysicsOperatorMixture(cfg)
    root = os.environ.get("EVAL_CKPT_ROOT", "/eu"); step = os.environ.get("EVAL_CKPT_STEP", "31250")
    ck = os.environ.get("EVAL_CKPT_PATH") or os.path.join(root, cfg.save_dir, f"ckpt_{step}.npz")
    if not os.path.exists(ck):
        print("no ckpt", ck); return
    ckptlib.load(model, ck)
    print(f"loaded {ck}", flush=True)

    @tf.function
    def ev(u0, dsc, cf, geom):
        pred, _ = model.call_with_gate(u0, dsc, cf, geom_mask=geom)
        return tf.cast(pred, tf.float32)

    xin, xtg, cm, dsc, cf, fam = (test["xin"], test["xtg"], test["cm"], test["desc"],
                                  test["coef"], test["fam"])
    mean, std = test["mean"], test["std"]
    geom = test.get("geom")
    if geom is None:
        geom = np.ones((xin.shape[0], xin.shape[2], xin.shape[3], 1), np.float32)
    tm = test["tm"]; fam_names = test["fam_names"]

    # one sample index per family (first occurrence)
    idx = []
    for fi in range(len(fam_names)):
        w = np.where(fam == fi)[0]
        if len(w): idx.append((fi, int(w[0])))
    sel = [i for _, i in idx]
    while len(sel) < 8: sel.append(sel[-1])           # pad batch to 8 (EVAL_EB regime)
    sel = np.array(sel[:max(8, len(sel))])

    sc = std[sel][:, None, None, None, :] + 1e-6
    mc = mean[sel][:, None, None, None, :]
    p = ev(xin[sel], dsc[sel], cf[sel], geom[sel]).numpy()
    p_phys = p[:, 1:] * sc + mc                       # (b, nf, H, W, C) physical futures
    t_phys = xtg[sel][:, 1:] * sc + mc

    nfam = len(idx)
    fig, axes = plt.subplots(nfam, 3, figsize=(9.5, 2.6 * nfam))
    for row, (fi, gi) in enumerate(idx):
        s = list(sel).index(gi)
        c = int(np.argmax(cm[gi]))                    # primary active channel
        nval = int(round(tm[gi, 1:].sum()))
        if fam_names[fi] == "pdearena_uncond": nval = min(nval, 4)
        fr = max(nval - 1, 0)                          # last valid future frame
        gt = t_phys[s, fr, :, :, c]; pr = p_phys[s, fr, :, :, c]
        err = np.abs(pr - gt)
        rel = np.sqrt(((pr - gt) ** 2).sum()) / (np.sqrt((gt ** 2).sum()) + 1e-7) * 100
        vmin, vmax = float(min(gt.min(), pr.min())), float(max(gt.max(), pr.max()))
        for col, (img, ttl, cmap, vm) in enumerate([
                (gt, "GT", "RdBu_r", (vmin, vmax)),
                (pr, "pred", "RdBu_r", (vmin, vmax)),
                (err, "|err|", "magma", (0, float(err.max()) + 1e-9))]):
            ax = axes[row, col]
            im = ax.imshow(img, cmap=cmap, vmin=vm[0], vmax=vm[1], origin="lower")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if col == 0:
                ax.set_ylabel(f"{fam_names[fi]}\nch{c} fr{fr}\nrelL2 {rel:.1f}%", fontsize=8)
            ax.set_title(ttl, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Recursive ADA (combo-rec): GT vs prediction, last valid frame", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = os.environ.get("OUT_DIR", "/out"); os.makedirs(out, exist_ok=True)
    for d in [out, os.path.join("/eu", cfg.save_dir)]:
        try:
            os.makedirs(d, exist_ok=True)
            fig.savefig(os.path.join(d, "recursive_gt_vs_pred.png"), dpi=150)
            print("saved", os.path.join(d, "recursive_gt_vs_pred.png"), flush=True)
        except Exception as e:
            print("save failed", d, e, flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "motion_tf.train.configs.phase1_eff_ada_pwb_archbias_combo_rec")
