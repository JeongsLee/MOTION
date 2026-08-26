"""Walrus flip-TTA control (08-12): score the released Walrus-FT CNS3D_128_Rand ckpt on
the CURRENT val dump under {id,fx,fy,fz} isotropic flips, native vs 4-flip ensemble --
(a) fairness control for the MOTION+ens comparison, (b) does a 1.3B data-driven model
inherit flip equivariance from ~90 trajectories? Also aligns MOTION cmp preds to the same
gt and prints the unified table. Based on walrus_score.py (same conversion/rollout)."""
import copy
import itertools
import os

import h5py
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf, open_dict

REF = "/corpus/results/walrus_ref"
NPZ = os.environ.get("NPZDIR", "/corpus/results/walrus_ref_cur")
DT = 1.0 / 20.0
FLIPS = tuple(os.environ.get("FLIPS", "id,fx,fy,fz").split(","))
import itertools as _it
_OHG = []
for _perm in _it.permutations(range(3)):
    for _sg in _it.product([1, -1], repeat=3):
        _OHG.append((_perm, _sg))                # 48; rN token = index
AXOF = {"fx": 2, "fy": 3, "fz": 4}          # array axes of (N,T,X,Y,Z,C)
VCH = {"fx": 0, "fy": 1, "fz": 2}           # our channel order [vx,vy,vz,rho,p]

trajs = [np.load(f"{NPZ}/val_{i}.npz")["fields"][7:21] for i in range(8)]
A0 = np.stack(trajs)                        # (8,14,128,128,128,5) physical
_BC = int(os.environ.get("BOOST", "0"))
if _BC:
    _AX = int(os.environ.get("BOOST_AX", "0"))           # 0=x,1=y,2=z of (N,T,X,Y,Z,C)
    _V = _BC * (1.0 / 128.0) / DT
    # rho'(x,t)=rho(x-Vt,t), u'=u(x-Vt,t)+V: frame t rolls FORWARD by _BC*t cells, which
    # pairs with the +V added below. The opposite pairing is not a solution.
    A0 = np.stack([np.stack([np.roll(A0[n, t], _BC * t, axis=_AX)
                             for t in range(A0.shape[1])]) for n in range(A0.shape[0])])
    A0 = np.array(A0, copy=True)
    A0[..., _AX] += _V
    print(f"[boost] {_BC} cell/frame axis {_AX} -> V={_V:.4f}", flush=True)
xs = (np.arange(128) + 0.5) / 128.0


def write_well(path, V, R, P):
    with h5py.File(path, "w") as f:
        f.attrs["dataset_name"] = "CNS3D_128_Rand"
        f.attrs["grid_type"] = "cartesian"
        f.attrs["n_spatial_dims"] = 3
        f.attrs["n_trajectories"] = V.shape[0]
        f.attrs["simulation_parameters"] = []
        dims = f.create_group("dimensions")
        dims.attrs["spatial_dims"] = ["x", "y", "z"]
        t = dims.create_dataset("time", data=np.arange(V.shape[1]) * DT)
        t.attrs["time_varying"] = True
        t.attrs["sample_varying"] = False
        for nm in ("x", "y", "z"):
            d = dims.create_dataset(nm, data=xs)
            d.attrs["time_varying"] = False
            d.attrs["sample_varying"] = False
        bc = f.create_group("boundary_conditions")
        for nm in ("x", "y", "z"):
            g = bc.create_group(f"{nm}_periodic")
            m = np.zeros(128)
            m[0] = 1
            m[-1] = 1
            g.create_dataset("mask", data=m, dtype=np.int8)
            g.attrs["bc_type"] = "PERIODIC"
            g.attrs["associated_dims"] = [nm]
            g.attrs["associated_fields"] = []
            g.attrs["sample_varying"] = False
            g.attrs["time_varying"] = False
        sc = f.create_group("scalars")
        sc.attrs["field_names"] = []
        t0 = f.create_group("t0_fields")
        t0.attrs["field_names"] = ["density", "pressure"]
        for nm, arr in (("density", R), ("pressure", P)):
            ds = t0.create_dataset(nm, data=arr, dtype=np.float32)
            ds.attrs["dim_varying"] = np.array([True, True, True])
            ds.attrs["sample_varying"] = True
            ds.attrs["time_varying"] = True
        t1 = f.create_group("t1_fields")
        t1.attrs["field_names"] = ["velocity"]
        ds = t1.create_dataset("velocity", data=V, dtype=np.float32)
        ds.attrs["dim_varying"] = np.array([True, True, True])
        ds.attrs["sample_varying"] = True
        ds.attrs["time_varying"] = True
        t2 = f.create_group("t2_fields")
        t2.attrs["field_names"] = []


def flip_arr(A, g):
    if g == "id":
        return A
    if g.startswith("t") and g[1:].isdigit():
        # PURE TRANSLATION by n cells (exact on the periodic lattice, exact for the
        # physics): input window AND targets move together, so this asks only whether
        # the operator is translation-equivariant -- absolute position encodings are not.
        _n = int(g[1:])
        return np.ascontiguousarray(np.roll(A, _n, axis=AXOF["fx"]))
    if g.startswith("r") and g[1:].isdigit():
        perm, sg = _OHG[int(g[1:])]
        # A: (N,T,X,Y,Z,C) physical [vx,vy,vz,rho,p]; rotate grid axes + rewire velocity
        v = np.transpose(A, (0, 1) + tuple(2 + p for p in perm) + (5,))
        for a in range(3):
            if sg[a] < 0:
                v = np.flip(v, axis=2 + a)
        v = np.array(v, copy=True)
        vel = v[..., :3].copy()
        for a in range(3):
            v[..., a] = sg[a] * vel[..., perm[a]]
        return np.ascontiguousarray(v)
    A = np.array(A, copy=True)
    for tok in ("fx", "fy", "fz"):
        if tok in g:
            A = np.flip(A, axis=AXOF[tok])
            A[..., VCH[tok]] *= -1.0
    return np.ascontiguousarray(A)


base_stats = yaml.safe_load(open(os.environ.get(
    "STATS_YAML", f"{REF}/train_stats.yaml")))
print("stats loaded:", {k: base_stats["mean"][k] for k in base_stats["mean"]}, flush=True)

for g in FLIPS:
    Ag = flip_arr(A0, g)
    W = f"/tmp/wellds_{g}/CNS3D_128_Rand"
    for sub in ("train", "valid", "test"):
        os.makedirs(f"{W}/data/{sub}", exist_ok=True)
    vel, rho, prs = Ag[..., 0:3], Ag[..., 3], Ag[..., 4]
    write_well(f"{W}/data/valid/cns3d_ourval.hdf5", vel, rho, prs)
    write_well(f"{W}/data/train/cns3d_ourval.hdf5", vel[:1], rho[:1], prs[:1])
    write_well(f"{W}/data/test/cns3d_ourval.hdf5", vel[:1], rho[:1], prs[:1])
    st = copy.deepcopy(base_stats)
    if g.startswith("r") and g[1:].isdigit():
        perm, sg = _OHG[int(g[1:])]
        for key in ("mean", "mean_delta"):
            mv = list(st[key]["velocity"])
            st[key]["velocity"] = [sg[a] * mv[perm[a]] for a in range(3)]
        for key in ("std", "rms", "std_delta", "rms_delta"):
            if key in st and isinstance(st[key].get("velocity"), list):
                sv = list(st[key]["velocity"])
                st[key]["velocity"] = [sv[perm[a]] for a in range(3)]
    for tok in ("fx", "fy", "fz"):           # flipped-axis velocity mean changes sign
        if tok in g and not g.startswith("r"):
            st["mean"]["velocity"][VCH[tok]] *= -1.0
            st["mean_delta"]["velocity"][VCH[tok]] *= -1.0
    yaml.safe_dump(st, open(f"{W}/data/logged_stats.yaml", "w"))
    yaml.safe_dump(st, open(f"{W}/logged_stats.yaml", "w"))
print("well datasets written (4 flips)", flush=True)

from hydra.utils import instantiate
from walrus.data import MixedWellDataModule
from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter
from walrus.trainer.training import expand_mask_to_match

cfg = OmegaConf.load(f"{REF}/extended_config.yaml")
with open_dict(cfg):
    info = cfg.data.module_parameters.well_dataset_info
    for k in list(info.keys()):
        if k != "CNS3D_128_Rand":
            del info[k]
    info.CNS3D_128_Rand.normalization_path = "logged_stats.yaml"
fmap = dict(cfg.data.field_index_map_override)
n_states = max(fmap.values()) + 1
dm_kwargs = OmegaConf.to_container(cfg.data.module_parameters, resolve=True)
dm_kwargs.pop("_target_")
device = torch.device("cuda")
model = instantiate(cfg.model, n_states=n_states)
if "finetuning_mods" in cfg:
    model.add_ft_options(OmegaConf.to_container(cfg.finetuning_mods, resolve=True))
ckpt = torch.load(f"{REF}/coalesced.pth", map_location="cpu", weights_only=True)["app"]["model"]
model.load_state_dict(ckpt)
model.to(device).eval()
formatter = ChannelsFirstWithTimeFormatter()
print("model ready", flush=True)


def rollout_model(revin, batch, max_rollout_steps=200, eps=1e-5):
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
    T_in = batch["input_fields"].shape[1]
    max_rollout_steps = max_rollout_steps + (T_in - 1)
    rollout_steps = min(y_ref.shape[1], max_rollout_steps)
    y_ref = y_ref[:, :rollout_steps]
    moving = copy.deepcopy(batch)
    preds = []
    for i in range(rollout_steps):
        inputs, _ = formatter.process_input(moving)
        inputs = list(inputs)
        with torch.no_grad():
            st = revin.compute_stats(inputs[0], metadata, epsilon=eps)
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
        if i != rollout_steps - 1:
            moving["input_fields"] = torch.cat(
                [moving["input_fields"][:, 1:], y_pred[:, -1:]], dim=1)
        preds.append(y_pred[:, -1:] if not (model.causal_in_time and i == 0) else y_pred)
    return torch.cat(preds, dim=1), y_ref


P_by, G_id = {}, None
for g in FLIPS:
    kw = copy.deepcopy(dm_kwargs)
    kw["well_dataset_info"] = {"CNS3D_128_Rand": {
        **kw["well_dataset_info"]["CNS3D_128_Rand"],
        "path": f"/tmp/wellds_{g}/CNS3D_128_Rand",
        "normalization_path": "logged_stats.yaml"}}
    dm = MixedWellDataModule(**{**kw, "well_base_path": f"/tmp/wellds_{g}",
                                "field_index_map_override": fmap,
                                "batch_size": 1, "data_workers": 2})
    revin = instantiate(cfg.trainer.revin)(train_dataset=dm.train_dataset, device=device)
    ps, gs = [], []
    for bi, batch in enumerate(dm.rollout_val_dataloaders()[0]):
        yp, yg = rollout_model(revin, batch, max_rollout_steps=11)
        ps.append(yp[0].float().cpu().numpy())
        gs.append(yg[0].float().cpu().numpy())
    P_by[g] = np.stack(ps)[:, :10]                       # (8,10,X,Y,Z,C)
    if g == "id":
        G_id = np.stack(gs)[:, :10]
    print(f"[{g}] rollout done {P_by[g].shape}", flush=True)

# The walrus val dataloader may REORDER trajectories and process_output may permute
# spatial axes: jointly search (loader sample j, axis perm, out channel, window frame t)
# aligning G_id to raw A0 (sample 0, rho).
ss = (slice(None, None, 8),) * 3
best = (0, None, None, None, None)
for j in range(8):
    for perm in itertools.permutations(range(3)):
        gtj = np.transpose(G_id[j, 0], perm + (3,))[ss]
        for c in range(G_id.shape[-1]):
            b = gtj[..., c].ravel()
            if b.std() == 0:
                continue
            for t in range(A0.shape[1]):
                a = A0[0, t][ss][..., 3].ravel()
                r = abs(np.corrcoef(a, b)[0, 1])
                if r > best[0]:
                    best = (r, j, perm, c, t)
print("align: corr=%.4f loader_j=%d perm=%s rho_out_ch=%d raw_win_t=%d" % best, flush=True)
assert best[0] > 0.98, "cannot align walrus output to raw data"
_, j0, P0, _, t0 = best
assert t0 == 3, f"frame offset unexpected: G_id[0] = raw window frame {t0} (expected 3)"
tr6 = (0, 1) + tuple(p + 2 for p in P0) + (5,)
G_id = np.ascontiguousarray(np.transpose(G_id, tr6))
for g in FLIPS:
    P_by[g] = np.ascontiguousarray(np.transpose(P_by[g], tr6))
# loader j -> raw i mapping (rho channel, frame 0 of prediction window)
rho_out = best[3]
smap = []                                                # loader j -> raw i
for j in range(8):
    b = G_id[j, 0][ss][..., rho_out].ravel()
    cc = [abs(np.corrcoef(A0[i, 3][ss][..., 3].ravel(), b)[0, 1]) for i in range(8)]
    smap.append(int(np.argmax(cc)))
    print(f"loader j={j} -> raw i={smap[-1]} corr={max(cc):.4f}", flush=True)
assert len(set(smap)) == 8, "loader->raw sample map not a bijection"
raw_fut = A0[:, 3:13]                                    # frames 10..19, raw sample order
chmap = []                                               # our_ch -> out_ch
for oc in range(5):
    a = raw_fut[smap[0], 0][ss][..., oc].ravel()
    cc = [abs(np.corrcoef(a, G_id[0, 0][ss][..., c].ravel())[0, 1])
          for c in range(G_id.shape[-1])]
    chmap.append(int(np.argmax(cc)))
    print(f"our ch{oc} -> out ch{chmap[-1]} corr={max(cc):.4f}", flush=True)
assert len(set(chmap)) == 5

def inv_pred(P, g):
    if g == "id":
        return P
    if g.startswith("t") and g[1:].isdigit():
        return np.ascontiguousarray(np.roll(P, -int(g[1:]), axis=AXOF["fx"]))
    if g.startswith("r") and g[1:].isdigit():
        perm, sg = _OHG[int(g[1:])]
        v = np.array(P, copy=True)
        for a in range(3):
            if sg[a] < 0:
                v = np.flip(v, axis=AXOF["fx"] + a - 2 + 2)   # spatial axes 2,3,4
        inv = [0, 0, 0]
        for a in range(3):
            inv[perm[a]] = a
        v = np.transpose(v, (0, 1) + tuple(2 + i for i in inv) + (5,))
        v = np.array(v, copy=True)
        vel = v[..., [chmap[0], chmap[1], chmap[2]]].copy()
        for a in range(3):
            v[..., chmap[perm[a]]] = sg[a] * vel[..., a]
        return np.ascontiguousarray(v)
    P = np.array(P, copy=True)
    for tok in ("fx", "fy", "fz"):
        if tok in g:
            P = np.flip(P, axis=AXOF[tok])
            P[..., chmap[VCH[tok]]] *= -1.0
    return np.ascontiguousarray(P)

P_tta = np.mean([inv_pred(P_by[g], g) for g in FLIPS], 0)
os.makedirs(f"{NPZ}", exist_ok=True)
_save = {"pred_id": P_by["id"].astype(np.float32),
         "pred_tta": P_tta.astype(np.float32), "gt": G_id.astype(np.float32),
         "chmap": np.array(chmap)}
if int(os.environ.get("SAVE_PERG", "0")):        # Fig2c: per-transform inverse preds
    for _g in FLIPS:
        _save[f"pred_{_g}"] = inv_pred(P_by[_g], _g).astype(np.float32)
np.savez_compressed(f"{NPZ}/walrus_{os.environ.get('TAG','tta')}_preds.npz", **_save)

def vrmse(p, gsrc):
    return np.mean([np.mean([np.sqrt(((p[k, ..., c] - gsrc[k, ..., c]) ** 2).mean()
                                     / (gsrc[k, ..., c].var() + 1e-7))
                             for c in range(p.shape[-1])]) for k in range(p.shape[0])])

def relj(p, gsrc):
    return 100 * np.mean([np.linalg.norm((p[k] - gsrc[k]).ravel())
                          / (np.linalg.norm(gsrc[k].ravel()) + 1e-12)
                          for k in range(p.shape[0])])

rows = {"Walrus native": P_by["id"], f"Walrus +TTA{len(FLIPS)}": P_tta}
per = {}
for nm, P in rows.items():
    vs = [vrmse(P[s], G_id[s]) for s in range(8)]
    rs = [relj(P[s], G_id[s]) for s in range(8)]
    per[nm] = (vs, rs)
    print(f"{nm}: per-sample VRMSE " + " ".join(f"{v:.3f}" for v in vs), flush=True)
    print(f"{nm}: per-sample relL2 " + " ".join(f"{r:.2f}" for r in rs), flush=True)

# per-flip native accuracy (equivariance probe): each flipped run vs its own flipped gt
for g in FLIPS[1:]:
    vg = np.mean([vrmse(inv_pred(P_by[g], g)[s], G_id[s]) for s in range(8)])
    print(f"[equivariance probe] {g}-rollout (inv) VRMSE mean={vg:.3f} "
          f"vs id {np.mean(per['Walrus native'][0]):.3f}", flush=True)

# MOTION+ens on the same gt (32^3 subgrid, same frames), axis-perm + slot map search
FAMK = "pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08"
ax32 = np.arange(0, 128, 4)[:32]
def subgrid(a, perm):                                   # a (10,X,Y,Z,C)
    a = np.transpose(a, (0,) + tuple(p + 1 for p in perm) + (4,))
    return a[:, ax32][:, :, ax32][:, :, :, ax32].reshape(a.shape[0], -1, a.shape[-1])
OG, OP = [], []
for s in range(8):
    Z = np.load(f"/corpus/scratch/cmp/fig1_ensv2_cmp_s{s}.npz")
    OG.append(Z[f"gt_{FAMK}_s{s}"])
    OP.append(Z[f"pred_{FAMK}_s{s}"])
PERM = (0, 1, 2)                                        # already raw-aligned above
inv = {ri: j for j, ri in enumerate(smap)}              # raw i -> loader j
r0 = max(abs(np.corrcoef(subgrid(G_id[inv[0]], PERM)[0, ::7, chmap[3]],
                         OG[0][0, ::7, sl])[0, 1])
         for sl in range(OG[0].shape[-1]) if OG[0][0, :, sl].std() > 0)
print("motion-gt align: corr=%.4f" % r0, flush=True)
if r0 > 0.98:
    slmap = []
    gw0 = subgrid(G_id[inv[0]], PERM)
    for oc in range(5):
        cc = [abs(np.corrcoef(gw0[0, ::7, chmap[oc]], OG[0][0, ::7, sl])[0, 1])
              if OG[0][0, :, sl].std() > 0 else 0.0 for sl in range(OG[0].shape[-1])]
        slmap.append(int(np.argmax(cc)))
    print("slot map (our raw ch -> cmp slot):", slmap, flush=True)
    wv_n, wv_t, wr_n, wr_t, mv, mr = [], [], [], [], [], []
    for i in range(8):
        j = inv[i]
        gw = subgrid(G_id[j], PERM)[..., [chmap[c] for c in range(5)]]
        pn = subgrid(P_by["id"][j], PERM)[..., [chmap[c] for c in range(5)]]
        pt = subgrid(P_tta[j], PERM)[..., [chmap[c] for c in range(5)]]
        om = OG[i][:, :, slmap]
        op = OP[i][:, :, slmap]
        gc = np.corrcoef(gw.ravel()[::97], om.ravel()[::97])[0, 1]
        wv_n.append(vrmse(pn, gw)); wr_n.append(relj(pn, gw))
        wv_t.append(vrmse(pt, gw)); wr_t.append(relj(pt, gw))
        mv.append(vrmse(op, om)); mr.append(relj(op, om))
        print(f"raw i={i} (loader j={j}) gt-corr={gc:.4f} | W-nat {wv_n[-1]:.3f}/{wr_n[-1]:.2f}%"
              f" | W-tta {wv_t[-1]:.3f}/{wr_t[-1]:.2f}%"
              f" | MOTION+ens {mv[-1]:.3f}/{mr[-1]:.2f}%", flush=True)
    print("=== UNIFIED TABLE (8 samples, frames 10-19, 32^3 grid, VRMSE / joint rel-L2):")
    for nm, (v, r) in (("Walrus-FT 1.3B native", (wv_n, wr_n)),
                       (f"Walrus-FT 1.3B +TTA{len(FLIPS)}", (wv_t, wr_t)),
                       ("MOTION 20.7M +ens    ", (mv, mr))):
        print(f"  {nm}: median {np.median(v):.3f} / {np.median(r):.2f}%   "
              f"mean {np.mean(v):.3f} / {np.mean(r):.2f}%", flush=True)
else:
    print("!! MOTION cmp gt STILL misaligned (corr %.3f) -- cmp preds predate current val"
          " split; regenerate cmp dump against walrus_ref_cur" % r0, flush=True)
    print("=== WALRUS-ONLY TABLE (frames 10-19, full 128^3, VRMSE / joint rel-L2):")
    for nm in rows:
        v, r = per[nm]
        print(f"  {nm}: median {np.median(v):.3f} / {np.median(r):.2f}%   "
              f"mean {np.mean(v):.3f} / {np.mean(r):.2f}%", flush=True)
print("WALRUS_TTA_DONE", flush=True)
