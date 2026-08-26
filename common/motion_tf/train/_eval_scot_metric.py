"""Evaluate OUR model with Poseidon/scOT's metric basis so the numbers are directly comparable to the
Poseidon-B inference CSVs: per PHYSICAL channel, MEDIAN relative-L1 error AT THE FINAL TIME (single-shot
IC->frame Nt-1, physical space), plus mean-over-channel-medians. Velocity reported as a combined 'uv'
(vx,vy as one vector field) to match scOT's uv column.
Run: python -u -m motion_tf.train._eval_scot_metric <config>   (CKPT env optional)
Channel map (Phase-2 8-slot): CE -> density=slot3, uv=slots0,1, pressure=slot4 (energy slot5 ignored, as
scOT does); NS -> uv=slots0,1, tracer=slot2.
"""
from __future__ import annotations
import glob, importlib, os, sys
import numpy as np
import tensorflow as tf
from ..model import PhysicsOperatorMixture
from ..data import poseidon
from ..utils import ckpt as ckptlib

# per-family physical-channel groups (label -> list of slot indices)
GROUPS = {
    "CE-RP":    {"rho": [3], "uv": [0, 1], "p": [4]},
    "CE-KH":    {"rho": [3], "uv": [0, 1], "p": [4]},
    "CE-CRP":   {"rho": [3], "uv": [0, 1], "p": [4]},
    "CE-Gauss": {"rho": [3], "uv": [0, 1], "p": [4]},
    "NS-Sines": {"uv": [0, 1], "tracer": [2]},
    "NS-Gauss": {"uv": [0, 1], "tracer": [2]},
    "NS-PwC":   {"uv": [0, 1], "tracer": [2]},          # downstream
    "CE-RM":    {"rho": [3], "uv": [0, 1], "p": [4]},   # downstream
    "ACE":      {"u": [2]},                              # downstream scalar (tracer slot)
    "Wave-Layer": {"u": [2]},                            # wave solution (slot2); slot6=c is fed-not-scored
}


def main(cfg_path):
    cfg = importlib.import_module(cfg_path).Cfg
    if bool(getattr(cfg, "bf16", False)):
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
    _, test = poseidon.build_poseidon_stream(cfg, cfg.batch)
    xt, mt, st, famt, names = test["traj"], test["mean"], test["std"], test["fam"], test["fam_names"]
    Nt = cfg.Nt
    # TARGET_FRAME: which frame to score (single-shot IC->frame). Default = last (Nt-1, t=1.0). Set to 14
    # to MATCH Poseidon's NS-PwC paper protocol (final time = step 14 = t=0.7); frame 20 = t=1.0 is the
    # paper's EXTRAPOLATION horizon. Lets us compare both models at the SAME physical time.
    TF = int(os.environ.get("TARGET_FRAME", Nt - 1))
    print(f"TARGET_FRAME={TF} (Nt-1={Nt-1}; 14=t=0.7 paper-final, 20=t=1.0 extrapolation)", flush=True)
    model = PhysicsOperatorMixture(cfg)
    _ = model.call_with_gate(xt[:1, 0], tf.zeros((1, cfg.desc_dim)), tf.zeros((1, 4)))
    ck = os.environ.get("CKPT")
    if not ck:
        cks = sorted(glob.glob(os.path.join("/code-vol", cfg.save_dir, "ckpt_*.npz")),
                     key=lambda p: int(p.split("_")[-1].split(".")[0]))
        ck = cks[-1]
    ckptlib.load(model, ck); print(f"restored {ck}\n", flush=True)

    FULL = bool(os.environ.get("FULL_TRAJ"))   # average rel-L1 over frames 1..Nt-1 (the "solve-the-PDE" metric)
    for fi, nm in enumerate(names):
        idx = np.where(famt == fi)[0]
        if len(idx) == 0:
            continue
        EB = int(getattr(cfg, "eval_batch", 8))
        # accumulate per-sample per-frame rel-L1 per channel-group: rels[label] -> list of (b,Nt) arrays
        rels = {label: [] for label in GROUPS.get(nm, {})}
        for s0 in range(0, len(idx), EB):
            sub = idx[s0:s0 + EB]
            u0 = xt[sub, 0].astype(np.float32)
            pr = tf.cast(model.call_with_gate(u0, tf.zeros((len(sub), cfg.desc_dim)),
                                              tf.zeros((len(sub), 4)))[0], tf.float32).numpy()  # (b,Nt,H,W,C)
            if os.environ.get("NORM_EVAL"):
                prp, rfp = pr, xt[sub]
            else:
                sc5 = st[sub][:, None, None, None, :]; mc5 = mt[sub][:, None, None, None, :]
                prp = pr * sc5 + mc5; rfp = xt[sub] * sc5 + mc5          # physical (b,Nt,H,W,C)
            for label, slots in GROUPS.get(nm, {}).items():
                p = prp[..., slots]; g = rfp[..., slots]                 # (b,Nt,H,W,k)
                num = np.sum(np.abs(p - g), axis=(2, 3, 4))              # (b,Nt) L1 over xy+components
                den = np.sum(np.abs(g), axis=(2, 3, 4)) + 1e-9
                rels[label].append(num / den)                           # (b,Nt)
        meds = {}
        for label in rels:
            r = np.concatenate(rels[label], 0)                          # (Nsamp, Nt)
            if FULL:
                per_sample = r[:, 1:Nt].mean(axis=1)                    # mean over predicted frames 1..Nt-1
            else:
                per_sample = r[:, TF]                                   # single target frame
            meds[label] = float(np.median(per_sample))                 # median over samples
            if os.environ.get("PER_FRAME"):                            # per-timestep: median over samples @ each frame
                pf = [float(np.median(r[:, t])) for t in range(1, Nt)]
                print(f"  [{nm}] {label} per-frame (t=1..{Nt-1}): " +
                      " ".join(f"{100*x:.2f}" for x in pf), flush=True)
        mom = float(np.mean(list(meds.values()))) if meds else float("nan")
        s = "  ".join(f"{k}={v*100:.4f}%" for k, v in meds.items())
        tag = f"FULL-TRAJ(mean t=1..{Nt-1})" if FULL else f"@frame{TF}"
        print(f"[{nm}] {tag} mean_over_median={mom*100:.2f}%  | {s}", flush=True)
    print("SCOT_METRIC_DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "motion_tf.train.configs.poseidon_pretrain_ivp")
