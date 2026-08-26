"""Reproduce Walrus's PoolBoil(Subcooled) numbers -- dt=0.1 release (the discretization their
notebook actually uses: np.linspace(0, t_final, 2001)). The dt=1.0 audit scored one-step at a
10x longer physical lead and is superseded; rollout verdicts were put on hold pending this run.
Conversion follows their demo notebook translate_bubble VERBATIM (t0=[gas-interface-sdf,
temperature], t1=velocity, swapaxes, wall/open BCs, simulation params); protocol = context 6,
prediction starts T=17 (paper, 2D); metrics = official VRMSE one-step / avg T[1:10] / T[11:30],
median over the 10 trajectories (their split folders are unpublished -> all 10, leakage favors
them)."""
import copy
import glob
import os

import h5py
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf, open_dict

REF = "/corpus/raw/poseidon_ckpt/walrus_ft_poolboil"
# PB_OFF=300 drops the pre-nucleation transient the BubbleML docs flag, putting THEIR eval
# window (loader t=11) into the developed-boiling regime our corpus build lives in.
OFF = int(os.environ.get("PB_OFF", "0"))
WELLBASE = f"/corpus/scratch/wellds01_off{OFF}" if OFF else "/corpus/scratch/wellds01"
WORK = f"{WELLBASE}/bubbleML_PoolBoiling-Subcooled"
SRC = sorted(glob.glob("/corpus/raw/bubbleml_fine/PoolBoiling-SubCooled-FC72-2D-0.1/Twall-*.hdf5"))
TF_STEPS = int(os.environ.get('TF_STEPS', '0'))
# WITH_P=1 tested and REJECTED: BubbleML also stores `pressure`, and their FT config selects
# no fields, so the channel set is fixed only by their unreleased conversion. Including it
# moves T[1:10] from 0.477 (their 0.4656) to 1.158 and degrades even the well-predicted sdf
# channel (0.031 -> 0.054) -- their conversion does NOT carry pressure.
WITH_P = int(os.environ.get('WITH_P', '0'))
T_START = 0          # offset lives in the loader (start_rollout_valid_output_at_t)
CTX = 6
for sub in ("train", "valid", "test"):
    os.makedirs(f"{WORK}/data/{sub}", exist_ok=True)

def params_of(f):
    if "real-runtime-params" not in f:
        return {}
    rp = f["real-runtime-params"][:]
    d = {n.decode().strip(): float(v) for n, v in rp}
    return d

def convert(fp, out, off=0):
    with h5py.File(fp, "r") as f:
        d = params_of(f)
        sdf = np.swapaxes(np.asarray(f["dfun"][off:], np.float32), 1, 2)[None]
        vx = np.swapaxes(np.asarray(f["velx"][off:], np.float32), 1, 2)[None]
        vy = np.swapaxes(np.asarray(f["vely"][off:], np.float32), 1, 2)[None]
        vel = np.stack([vx, vy], -1)
        temp = np.swapaxes(np.asarray(f["temperature"][off:], np.float32), 1, 2)[None]
        pres = np.swapaxes(np.asarray(f["pressure"][off:], np.float32), 1, 2)[None]
        xc = np.asarray(f["x"][0][0, :], np.float64)
        yc = np.asarray(f["y"][0][:, 0], np.float64)
        T = sdf.shape[1]
        dt_full = d.get("tmax", 200.0) / max(T + off - 1, 1)
        t = off * dt_full + np.arange(T) * dt_full
    with h5py.File(out, "w") as g:
        g.attrs["dataset_name"] = "bubbleML_PoolBoiling-Subcooled"
        g.attrs["grid_type"] = "cartesian"
        g.attrs["n_spatial_dims"] = 2
        g.attrs["n_trajectories"] = 1
        sim = ["inv_reynolds", "cpgas", "mugas", "rhogas", "thcogas", "stefan",
               "prandtl", "heater-nucWaitTime", "heater-wallTemp"]
        vals = [d.get("ins_invreynolds", 0.0042), d.get("mph_cpgas", 0.83),
                d.get("mph_mugas", 1.0), d.get("mph_rhogas", 0.0083),
                d.get("mph_thcogas", 0.25), d.get("mph_stefan", 0.6642),
                d.get("ht_prandtl", 8.4), d.get("ht_nucseeddist", 0.0),
                d.get("ht_twall_high", 1.0)]
        g.attrs["simulation_parameters"] = sim
        for k, v in zip(sim, vals):
            g.attrs[k] = v
        dims = g.create_group("dimensions")
        dims.attrs["spatial_dims"] = ["x", "y"]
        tt = dims.create_dataset("time", data=t)
        tt.attrs["time_varying"] = True; tt.attrs["sample_varying"] = False
        for nm, cc in (("x", xc), ("y", yc)):
            dd = dims.create_dataset(nm, data=cc)
            dd.attrs["time_varying"] = False; dd.attrs["sample_varying"] = False
        bc = g.create_group("boundary_conditions")
        def mk(nm, mask, typ, dim):
            gg = bc.create_group(nm)
            gg.create_dataset("mask", data=mask, dtype=np.int8)
            gg.attrs["bc_type"] = typ; gg.attrs["associated_dims"] = [dim]
            gg.attrs["associated_fields"] = []
            gg.attrs["sample_varying"] = False; gg.attrs["time_varying"] = False
        mx = np.zeros(len(xc)); mx[0] = 1; mx[-1] = 1
        mk("x_wall_noslip", mx, "WALL", "x")
        my0 = np.zeros(len(yc)); my0[0] = 1
        mk("y_wall_noslip", my0, "WALL", "y")
        my1 = np.zeros(len(yc)); my1[-1] = 1
        mk("y_open", my1, "OPEN", "y")
        sc = g.create_group("scalars")
        sc.attrs["field_names"] = sim
        for k, v in zip(sim, vals):
            ds = sc.create_dataset(k, data=v)
            ds.attrs["sample_varying"] = False; ds.attrs["time_varying"] = False
        t0 = g.create_group("t0_fields")
        t0f = ["gas-interface-sdf", "temperature"] + (["pressure"] if WITH_P else [])
        t0.attrs["field_names"] = t0f
        for nm, arr in ([("gas-interface-sdf", sdf), ("temperature", temp)]
                        + ([("pressure", pres)] if WITH_P else [])):
            ds = t0.create_dataset(nm, data=arr, dtype=np.float32)
            ds.attrs["dim_varying"] = np.array([True, True])
            ds.attrs["sample_varying"] = True; ds.attrs["time_varying"] = True
        t1 = g.create_group("t1_fields")
        t1.attrs["field_names"] = ["velocity"]
        ds = t1.create_dataset("velocity", data=vel, dtype=np.float32)
        ds.attrs["dim_varying"] = np.array([True, True])
        ds.attrs["sample_varying"] = True; ds.attrs["time_varying"] = True
        g.create_group("t2_fields").attrs["field_names"] = []
    return d

import glob as _g
SKIP_CONV = len(_g.glob(f"{WORK}/data/valid/*.hdf5")) == len(SRC) and \
    os.path.exists(f"{WORK}/stats.yaml") and int(os.environ.get("FORCE_CONV", "0")) == 0
if SKIP_CONV:
    print(f"[skip] conversion present ({len(SRC)} files + stats)", flush=True)
stats_acc = {k: [] for k in ("sdf", "temp", "pres", "vx", "vy",
                             "dsdf", "dtemp", "dpres", "dvx", "dvy")}
for i, fp in enumerate([] if SKIP_CONV else SRC):
    base = os.path.basename(fp).replace(".hdf5", "")
    convert(fp, f"{WORK}/data/valid/{base}.hdf5", off=OFF)   # split folders unpublished -> score all 15
    for sub in ("train", "test"):
        lnk = f"{WORK}/data/{sub}/{base}.hdf5"
        if not os.path.exists(lnk):
            os.link(f"{WORK}/data/valid/{base}.hdf5", lnk)
    with h5py.File(fp, "r") as f:
        for key, nm in (("dfun", "sdf"), ("temperature", "temp"), ("pressure", "pres"),
                        ("velx", "vx"), ("vely", "vy")):
            a = np.asarray(f[key][OFF::20], np.float32)
            stats_acc[nm].append(a.ravel())
            stats_acc["d" + nm].append((np.asarray(f[key][OFF + 1::20], np.float32)[:a.shape[0] - 1]
                                        - a[:-1]).ravel())
    print("converted", os.path.basename(fp), flush=True)

if SKIP_CONV:
    import yaml as _y
    _st = _y.safe_load(open(f"{WORK}/stats.yaml"))
    class _SS:  # rebuild S["..."][1] (std) lookups used below from the saved stats
        pass
    S = {"sdf": [None, _st["std"]["gas-interface-sdf"]],
         "temp": [None, _st["std"]["temperature"]],
         "pres": [None, _st["std"].get("pressure", 1.0)],
         "vx": [None, _st["std"]["velocity"][0]],
         "vy": [None, _st["std"]["velocity"][1]]}

def s_of(key):
    v = np.concatenate(stats_acc[key]); dv = np.concatenate(stats_acc["d" + key])
    return (float(v.mean()), float(v.std()), float(np.sqrt((v**2).mean())),
            float(dv.mean()), float(dv.std()), float(np.sqrt((dv**2).mean())))
S = S if SKIP_CONV else {k: s_of(k) for k in ("sdf", "temp", "pres", "vx", "vy")}
stats = {}
for i, key in enumerate(() if SKIP_CONV else ("mean", "std", "rms", "mean_delta", "std_delta", "rms_delta")):
    stats[key] = {"gas-interface-sdf": S["sdf"][i], "temperature": S["temp"][i],
                  "velocity": [S["vx"][i], S["vy"][i]]}
    if WITH_P:
        stats[key]["pressure"] = S["pres"][i]
if not SKIP_CONV:
    yaml.safe_dump(stats, open(f"{WORK}/stats.yaml", "w"))
    yaml.safe_dump(stats, open(f"{WORK}/data/stats.yaml", "w"))
    print("stats written", flush=True)

# ---------------- model ----------------
from hydra.utils import instantiate
from walrus.data import MixedWellDataModule
from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter
from walrus.trainer.training import expand_mask_to_match

cfg = OmegaConf.load(f"{REF}/extended_config.yaml")
with open_dict(cfg):
    info = cfg.data.module_parameters.well_dataset_info
    for k in list(info.keys()):
        del info[k]
    info["bubbleML_PoolBoiling-Subcooled"] = {
        "include_filters": [], "exclude_filters": [],
        "path": WORK, "normalization_path": "stats.yaml"}
    cfg.data.well_base_path = WELLBASE
fmap = dict(cfg.data.field_index_map_override)
n_states = max(fmap.values()) + 1
dm_kwargs = OmegaConf.to_container(cfg.data.module_parameters, resolve=True)
dm_kwargs.pop("_target_")
dm = MixedWellDataModule(**{**dm_kwargs, "well_base_path": WELLBASE,
                            "field_index_map_override": fmap,
                            "batch_size": 1, "data_workers": 2})
device = torch.device("cuda")
model = instantiate(cfg.model, n_states=n_states)
if "finetuning_mods" in cfg:
    model.add_ft_options(OmegaConf.to_container(cfg.finetuning_mods, resolve=True))
ck = torch.load(f"{REF}/coalesced.pth", map_location="cpu", weights_only=True)["app"]["model"]
model.load_state_dict(ck)
model.to(device).eval()
formatter = ChannelsFirstWithTimeFormatter()
revin = instantiate(cfg.trainer.revin)(train_dataset=dm.train_dataset, device=device)
print("rollout params:", {k: v for k, v in dm_kwargs.items()
      if "roll" in k or "step" in k or "max" in k}, flush=True)
print("model + revin ready", flush=True)


def rollout(batch, n_steps, teacher=False):
    metadata = batch["metadata"]
    batch = {k: v.to(device) if k not in {"metadata", "boundary_conditions"} else v
             for k, v in batch.items()}
    cf = metadata.constant_field_names[0] if len(metadata.constant_field_names) else []
    mask = None
    if "mask" in cf:
        mi = cf.index("mask")
        mask = batch["constant_fields"][..., mi:mi + 1].to(device, dtype=torch.bool)
    inputs, y_ref = formatter.process_input(batch, causal_in_time=model.causal_in_time,
                                            predict_delta=True, train=False)
    steps = min(y_ref.shape[1], n_steps)
    y_ref = y_ref[:, :steps]
    moving = copy.deepcopy(batch)
    preds = []
    for i in range(steps):
        inputs, _ = formatter.process_input(moving)
        inputs = list(inputs)
        with torch.no_grad():
            st = revin.compute_stats(inputs[0], metadata, epsilon=1e-5)
            ni = inputs[:]
            ni[0] = revin.normalize_stdmean(ni[0], st)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y_pred = model(ni[0], ni[1], ni[2].tolist(), metadata=metadata)
            if model.causal_in_time:
                y_pred = y_pred[-1:]
            y_pred = (inputs[0][-y_pred.shape[0]:].float()
                      + revin.denormalize_delta(y_pred.float(), st))
            y_pred = formatter.process_output(y_pred, metadata)[..., :y_ref.shape[-1]]
            if mask is not None:
                y_pred.masked_fill_(expand_mask_to_match(mask, y_pred), 0)
            y_pred = y_pred.masked_fill(~batch["padded_field_mask"], 0.0)
        if i != steps - 1:
            # TEACHER FORCING: feed the ground-truth frame back instead of the prediction, so
            # every step is scored from a clean context. Their reported "one-step" is 7.5x
            # better than their own T[1:10], a ratio a free rollout's first step cannot produce
            # (ours is 1.8x) -- so it has to be this measurement, averaged over all start times.
            nxt = y_ref[:, i:i + 1] if teacher else y_pred[:, -1:]
            moving["input_fields"] = torch.cat(
                [moving["input_fields"][:, 1:], nxt], dim=1)
        preds.append(y_pred[:, -1:] if not (model.causal_in_time and i == 0) else y_pred)
    return torch.cat(preds, dim=1), y_ref


# the datamodule's rollout loaders start at trajectory start; we need start at T_START-CTX.
# The dataset yields full trajectories in rollout mode -> slice inside the batch tensors.
dls = dm.rollout_val_dataloaders() + [dm.rollout_train_dataloaders()] if hasattr(dm, "rollout_train_dataloaders") else dm.rollout_val_dataloaders()
res = []
pop_res = []
rel_res = []
tf_res = []
count = 0
for dl in dm.rollout_val_dataloaders():
    for batch in dl:
        if count == 0:
            print({k: tuple(v.shape) for k, v in batch.items()
                   if hasattr(v, "shape")}, flush=True)
        yp, yg = rollout(batch, n_steps=30)
        tp, tg = (rollout(batch, n_steps=TF_STEPS, teacher=True) if TF_STEPS
                  else (yp[:, :1] * 0, yg[:, :1]))
        p = yp[0].float().cpu().numpy()
        g = yg[0].float().cpu().numpy()
        last = batch["input_fields"][0, -1].float().cpu().numpy()   # persistence reference
        C = p.shape[-1]

        # POP: the population per-channel variance (stats.yaml), the denominator a global
        # z-score convention would use, vs the_well's official per-sample spatial variance.
        # Velocity decorrelates between plotfiles here (persistence VRMSE 1.51), so which
        # denominator is used decides the headline by more than an order of magnitude.
        POP = np.array(([S["sdf"][1] ** 2, S["temp"][1] ** 2]
                        + ([S["pres"][1] ** 2] if WITH_P else [])
                        + [S["vx"][1] ** 2, S["vy"][1] ** 2, 1.0]))[:C]

        def vr_of(pred, pop=False):
            out = []
            for k in range(min(30, pred.shape[0])):
                out.append([np.sqrt(((pred[k, ..., c] - g[k, ..., c]) ** 2).mean()
                                    / ((POP[c] if pop else g[k, ..., c].var()) + 1e-7))
                            for c in range(C)])
            return np.array(out)                                    # (K, C)

        def rel_of(pred):                                  # joint (all-channel) rel-L2 per step
            out = []
            for k in range(min(30, pred.shape[0])):
                out.append(np.linalg.norm(pred[k] - g[k]) / (np.linalg.norm(g[k]) + 1e-12))
            return np.array(out)
        rl = rel_of(p)
        vc = vr_of(p)
        vb = vr_of(p, pop=True)
        vp = vr_of(np.repeat(last[None, ..., :C], p.shape[0], 0))
        vr = vc.mean(1)
        res.append(list(vr))
        if count == 0:
            md = batch["metadata"]
            print("per-channel one-step  [spatial-var]:", np.array2string(vc[0], precision=4),
                  "\n per-channel one-step [population-var]:", np.array2string(vb[0], precision=4),
                  "| fields:", getattr(md, "field_names", "?"),
                  "| input_fields", tuple(batch["input_fields"].shape),
                  "| gt", g.shape, flush=True)
        print(f"traj {count}: one-step={vr[0]:.4f} T1_10={vr[:10].mean():.4f} "
              f"T11_30={vr[10:30].mean():.4f} | persistence={vp[0].mean():.4f} "
              f"| POP one-step={vb[0].mean():.4f} T1_10={vb[:10].mean(1).mean():.4f} "
              f"T11_30={vb[10:30].mean(1).mean():.4f}", flush=True)
        pop_res.append([vb[0].mean(), vb[:10].mean(1).mean(), vb[10:30].mean(1).mean()])
        rel_res.append([rl[0], rl[:10].mean(), rl[10:30].mean()])
        print(f"    joint rel-L2: one-step={rl[0]:.4f} T1_10={rl[:10].mean():.4f} "
              f"T11_30={rl[10:30].mean():.4f}", flush=True)
        tpn = tp[0].float().cpu().numpy(); tgn = tg[0].float().cpu().numpy()
        tf_pop, tf_sp = [], []
        for k in range(tpn.shape[0]):
            tf_pop.append(np.mean([np.sqrt(((tpn[k, ..., c] - tgn[k, ..., c]) ** 2).mean()
                                           / (POP[c] + 1e-7)) for c in range(C)]))
            tf_sp.append(np.mean([np.sqrt(((tpn[k, ..., c] - tgn[k, ..., c]) ** 2).mean()
                                          / (tgn[k, ..., c].var() + 1e-7)) for c in range(C)]))
        tf_res.append([np.mean(tf_pop), np.mean(tf_sp)])
        print(f"    teacher-forced over {tpn.shape[0]} starts: POP={np.mean(tf_pop):.4f} "
              f"spatial={np.mean(tf_sp):.4f}", flush=True)
        count += 1
A = np.array([r[:30] for r in res if len(r) >= 30])
print(f"=== WALRUS-FT PoolBoil ({len(A)} traj): one-step median={np.median(A[:,0]):.4f} "
      f"T[1:10] median={np.median(A[:, :10].mean(1)):.4f} "
      f"T[11:30] median={np.median(A[:, 10:30].mean(1)):.4f}", flush=True)
P = np.array(pop_res)
print(f"=== POP-var variant: one-step={np.median(P[:, 0]):.4f} "
      f"T[1:10]={np.median(P[:, 1]):.4f} T[11:30]={np.median(P[:, 2]):.4f}", flush=True)
R = np.array(rel_res)
print(f"=== joint rel-L2 (our metric): one-step median={np.median(R[:, 0]):.4f} "
      f"T[1:10]={np.median(R[:, 1]):.4f} T[11:30]={np.median(R[:, 2]):.4f} "
      f"| mean {R[:, 0].mean():.4f}/{R[:, 1].mean():.4f}/{R[:, 2].mean():.4f}", flush=True)
F = np.array(tf_res)
print(f"=== teacher-forced one-step: POP={np.median(F[:, 0]):.4f} "
      f"spatial={np.median(F[:, 1]):.4f}", flush=True)
print("PB_SCORE_DONE", flush=True)
