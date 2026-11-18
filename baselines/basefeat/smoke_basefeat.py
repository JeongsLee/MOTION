"""CPU smoke test of the basefeat patch (2026-11-04). Not a training run.

  python smoke_basefeat.py <bcat_src> <prose_fd_dir> <prebuilt_n900_dir>

For each baseline (ConvEmbedder with the fair-run sizes, data_dim 4, x_num 128):
  1. off: patched embedder == unpatched embedder (same init seed, same input) -> max |diff|
  2. phys: shapes, finite forward/backward, extra parameters, encode time vs off
  3. feature-group knockout zeroes exactly its channels
  4. dihedral consistency of the features: rotating the input by 90 deg (with the velocity slots rotated) rotates the
     scalar features (laplacian, advection, divergence) the same way
Real data: 2 trajectories each of com_ns, incom_ns, pdearena_ns, shallow_water from prebuilt_n900 (padded to 4 slots).
"""
import glob, importlib, os, shutil, subprocess, sys, tempfile, time, types
import numpy as np
import torch

bcat_src, prose_dir, pre = sys.argv[1], sys.argv[2], sys.argv[3]
HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(8)

# stub neuralop (bcat embedder imports FNO2d at module level; the fair runner stubs it the same way)
nm = types.ModuleType("neuralop"); nmm = types.ModuleType("neuralop.models"); nmm.FNO2d = None
sys.modules["neuralop"] = nm; sys.modules["neuralop.models"] = nmm
# stub attention-only deps (not used by the embedder): rotary_embedding_torch, torchtune
for _m in ("rotary_embedding_torch", "torchtune", "torchtune.modules"):
    _x = types.ModuleType(_m); _x.RotaryEmbedding = object; _x.RotaryPositionalEmbeddings = object; sys.modules[_m] = _x


class Cfg(dict):
    def __getattr__(self, k): return self[k]
    def get(self, k, d=None): return dict.get(self, k, d)


def load_embedder(src_models, tag, mode, ko=""):
    """import <src>/models/embedder.py as models.embedder in a fresh module namespace (MODE is read at import)."""
    os.environ["BASEFEAT"] = mode; os.environ["BASEFEAT_KO"] = ko
    for k in [k for k in sys.modules if k == "models" or k.startswith("models.")]:
        del sys.modules[k]
    root = os.path.dirname(src_models)
    sys.path.insert(0, root)
    try:
        m = importlib.import_module("models.embedder")
        if mode != "off" or "basefeat" in open(m.__file__).read():
            pass
    finally:
        sys.path.pop(0)
    return m


def data_batch():
    xs = []
    for n in ("com_ns", "incom_ns", "pdearena_ns", "shallow_water"):
        p = glob.glob(os.path.join(pre, f"{n}_n900_t20*"))[0]
        a = np.load(p, mmap_mode="r") if p.endswith(".npy") else np.load(sorted(glob.glob(p + "/shard_*.npy"))[0], mmap_mode="r")
        x = np.asarray(a[:2, :10]).astype(np.float32)
        x = (x - x.mean((1, 2, 3), keepdims=True)) / (x.std((1, 2, 3), keepdims=True) + 1e-6)   # per-sample z-score
        pad = np.zeros(x.shape[:-1] + (4 - x.shape[-1],), np.float32)
        xs.append(np.concatenate([x, pad], -1))
    return torch.from_numpy(np.concatenate(xs, 0))   # (8, 10, 128, 128, 4)


X = data_batch(); T = torch.arange(10, dtype=torch.float32).view(1, 10, 1).repeat(X.shape[0], 1, 1)
for tag, root, kind, cfg in [
    ("bcat", bcat_src, "bcat", Cfg(dim=1024, patch_num=16, patch_num_output=16, time_embed="learnable",
                                   max_time_len=20, conv_dim=32, early_conv=0, deep=0)),
    ("prose", prose_dir, "prose", Cfg(dim=1024, patch_num=8, patch_num_output=16, time_embed="learnable",
                                      max_time_len=20, conv_dim=32, early_conv=0, deep=0, initialize_small_output=0))]:
    orig = os.path.join(root, "models")
    work = tempfile.mkdtemp(); shutil.copytree(root, os.path.join(work, "src"))
    subprocess.run([sys.executable, os.path.join(HERE, "patch_basefeat.py"), kind, os.path.join(work, "src")], check=True)
    patched = os.path.join(work, "src", "models")
    res = {}
    E0 = load_embedder(orig, tag + "o", "off").ConvEmbedder
    E1 = load_embedder(patched, tag + "p", "off").ConvEmbedder
    torch.manual_seed(0); e0 = E0(cfg, 128, 4)
    torch.manual_seed(0); e1 = E1(cfg, 128, 4)
    with torch.no_grad():
        res["off_maxdiff"] = float((e0.encode(X, T) - e1.encode(X, T)).abs().max())
    mphys = load_embedder(patched, tag + "p", "phys")
    torch.manual_seed(0); e2 = mphys.ConvEmbedder(cfg, 128, 4)
    n0 = sum(p.numel() for p in e0.parameters()); n2 = sum(p.numel() for p in e2.parameters())
    res["embedder_params_off/phys"] = (n0, n2, n2 - n0)
    t0 = time.time(); y0 = e0.encode(X, T); t_off = time.time() - t0
    t0 = time.time(); y2 = e2.encode(X, T); t_phys = time.time() - t0
    y2.square().mean().backward()
    res["phys_out_shape"] = tuple(y2.shape); res["finite"] = bool(torch.isfinite(y2).all())
    res["grads_finite"] = all(torch.isfinite(p.grad).all().item() for p in e2.parameters() if p.grad is not None)
    res["encode_s_off/phys"] = (round(t_off, 3), round(t_phys, 3))
    # feature checks on the raw feature maps
    x = X[:, 0].permute(0, 3, 1, 2)          # (N, C, H, W)
    f = mphys._bf.features(x)[:, 4:]
    sl = mphys._bf.group_slices(4)
    for g in ("lap", "adv"):
        mko = load_embedder(patched, tag + "k", "phys", g)
        fk = mko._bf.features(x)[:, 4:]
        a, b = sl[g]
        res[f"ko_{g}_zeroed"] = bool((fk[:, a:b] == 0).all()) and bool(torch.allclose(
            torch.cat([fk[:, :a], fk[:, b:]], 1), torch.cat([f[:, :a], f[:, b:]], 1)))
    # rotation by 90 deg: (H, W) -> rot90; velocity (u along H, v along W) -> (u', v') = (-v, u) for torch.rot90(k=1)
    xr = torch.rot90(x, 1, (2, 3)).clone()
    u, v = xr[:, 0].clone(), xr[:, 1].clone()
    vel = torch.tensor([s[:2].abs().sum() > 0 for s in x])            # families with velocity slots
    xr[vel, 0] = -v[vel]; xr[vel, 1] = u[vel]
    fr = mphys._bf.features(xr)[:, 4:]
    idx = vel.nonzero().flatten()
    for g in ("lap", "adv", "div"):
        a, b = sl[g]
        ref = torch.rot90(f[:, a:b], 1, (2, 3)).clone()
        if g in ("lap", "adv"):          # per-channel scalar operator: the velocity channels mix like the state
            r0, r1 = ref[:, 0].clone(), ref[:, 1].clone()
            ref[:, 0] = -r1; ref[:, 1] = r0
        res[f"rot90_{g}_maxrel(vel families, interior)"] = float(
            ((fr[idx, a:b] - ref[idx])[..., 2:-2, 2:-2]).abs().max() / ref[idx].abs().max())
    print(tag, res)
