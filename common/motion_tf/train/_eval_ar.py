"""AR-rollout eval for OUR model: rel-L1 @final via K-step autoregressive rollout (vs single-shot K=1).
Our model maps IC->all Nt frames in one call; AR feeds the lead-L prediction back as the next IC, K times,
to reach the final frame (K*L = Nt-1). env: CKPT, AR_STEPS (default 10). Mirrors _eval_scot_metric metric."""
import os, importlib, numpy as np, tensorflow as tf
from ..model import PhysicsOperatorMixture
from ..data import poseidon
from ..utils import ckpt as ckptlib

GROUPS = {
    "CE-RP": {"rho":[3],"uv":[0,1],"p":[4]}, "CE-KH": {"rho":[3],"uv":[0,1],"p":[4]},
    "CE-CRP":{"rho":[3],"uv":[0,1],"p":[4]}, "CE-Gauss":{"rho":[3],"uv":[0,1],"p":[4]},
    "NS-Sines":{"uv":[0,1],"tracer":[2]}, "NS-Gauss":{"uv":[0,1],"tracer":[2]},
    "NS-PwC":{"uv":[0,1],"tracer":[2]}, "CE-RM":{"rho":[3],"uv":[0,1],"p":[4]}, "ACE":{"u":[2]},
}

def main(cfg_path):
    cfg = importlib.import_module(cfg_path).Cfg
    if bool(getattr(cfg,"bf16",False)): tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
    Nt = cfg.Nt
    K = int(os.environ.get("AR_STEPS","10"))
    L = (Nt-1)//K
    assert K*L == Nt-1, f"AR_STEPS {K} must divide Nt-1={Nt-1} (got L={L})"
    _, test = poseidon.build_poseidon_stream(cfg, cfg.batch)
    xt,cmt,mt,st,famt = test["traj"],test["cm"],test["mean"],test["std"],test["fam"]
    names = test["fam_names"]
    model = PhysicsOperatorMixture(cfg)
    _ = model.call_with_gate(xt[:1,0], tf.zeros((1,cfg.desc_dim)), tf.zeros((1,4)))
    ck = os.environ["CKPT"]; ckptlib.load(model, ck); print(f"AR eval K={K} L={L}  ck={ck}", flush=True)
    EB = int(getattr(cfg,"eval_batch",8))
    for fi,nm in enumerate(names):
        idx = np.where(famt==fi)[0]
        if len(idx)==0: continue
        pf_list=[]
        for s0 in range(0,len(idx),EB):
            sub = idx[s0:s0+EB]
            u = xt[sub,0].astype(np.float32)                       # IC (normalized)
            for seg in range(K):
                pr = tf.cast(model.call_with_gate(u, tf.zeros((len(sub),cfg.desc_dim)),
                                                  tf.zeros((len(sub),4)))[0], tf.float32).numpy()
                u = pr[:, L]                                       # advance L frames -> new IC
            pf_list.append(u)                                      # after K steps = frame Nt-1
        pf = np.concatenate(pf_list,0)
        scb = st[idx][:,None,None,:]; mcb = mt[idx][:,None,None,:]
        pf = pf*scb + mcb                                          # denorm final pred
        rf = xt[idx, Nt-1]*scb + mcb                               # denorm final ref
        meds={}
        for label,slots in GROUPS.get(nm,{}).items():
            p=pf[...,slots]; g=rf[...,slots]
            num=np.sum(np.abs(p-g),axis=(1,2,3)); den=np.sum(np.abs(g),axis=(1,2,3))+1e-9
            meds[label]=float(np.median(num/den))
        s="  ".join(f"{k}={v*100:.2f}%" for k,v in meds.items())
        print(f"[{nm}] AR(K={K}) mean_over_median={np.mean(list(meds.values()))*100:.2f}%  | {s}", flush=True)
    print("AR_DONE", flush=True)

if __name__=="__main__":
    import sys; main(sys.argv[1])
