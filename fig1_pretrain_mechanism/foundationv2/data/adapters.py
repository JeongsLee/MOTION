"""Format adapters: raw corpus file -> iterator of (unit_id, UnifiedSample).

Each adapter reads ONLY the FamilySpec (location, channels, param rules) + a unit
id, and returns a deterministic sample for that unit. Unit granularity matches
spec.unit (trajectory/design/case) so split manifests are leak-proof. Adapters are
lazy: they enumerate unit ids cheaply (headers only) and materialize one unit at a
time (RAM-safe on 13GB CPU nodes).

Registered by fmt:
  grid_npy     prose128 single .npy or shard-dir  (NHWC, T on axis 1)
  grid_nchw    Poseidon .nc via h5py (sample,time,[channel],x,y)
  pdebench3d   PDEBench 3D CNS hdf5 (per-key volumetric arrays)
  mesh_geofno_* / mesh_airfrans / mesh_shapenet_car / mesh_drivaernet
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np

from . import symbolic as SY
from .schema import grid_sample, normalize_coords, point_sample

CORPUS = os.environ.get("CORPUS_ROOT", "/corpus")


def _path(spec):
    return os.path.join(CORPUS, spec.location)


def _surface_normals(mesh, n):
    """Per-point surface normals (n,3) via pyvista, or None on failure. DoMINO geom lever: the
    local surface orientation is highly informative for surface-pressure prediction. Tries the mesh
    directly, then extract_surface() (shapenet's quad .vtk reads as UnstructuredGrid where
    compute_normals fails); only accepts the fallback if the point count still covers n (alignment)."""
    # extract_surface().triangulate() rescues shapenet's QUAD .vtk (compute_normals needs tris);
    # triangulate preserves point order/count so normals stay aligned with mesh.points -> coords.
    for src_fn in (lambda: mesh, lambda: mesh.extract_surface().triangulate()):
        try:
            m = src_fn().compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)
            nrm = np.asarray(m.point_data["Normals"], np.float32)
            if nrm.shape[0] >= n and nrm.shape[1] == 3:
                return nrm[:n]
        except Exception:
            continue
    return None


def _sdf_gradient_normal(mesh, k):
    """Normalized gradient of the signed-distance field (n,k) = the geometry normal direction
    everywhere in the volume (DoMINO/AB-UPT geom lever), or None on failure."""
    try:
        g = mesh.compute_derivative(scalars="implicit_distance", gradient="grad")
        grad = np.asarray(g.point_data["grad"], np.float32)[:, :k]
        return (grad / (np.linalg.norm(grad, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
    except Exception:
        return None


def estimate_normals(coords01, k=16):
    """Per-point normals from the POINT CLOUD alone: local PCA (smallest eigenvector of the
    k-NN covariance), oriented outward from the centroid.

    Why this exists (2026-08-01): pyvista's compute_normals silently failed on the container's
    pyvista build for shapenet's quad VTK — BOTH the direct and extract_surface().triangulate()
    attempts returned nothing, so `_surface_normals` handed back None and the surface families
    were training with all-zero normal channels (verified: geom[:,2:5] absmean == 0). Surface
    orientation is the primary geometric signal for surface-pressure prediction, so it cannot
    depend on a mesh library version. This estimator needs only coordinates.

    Outward orientation via the centroid is exact for star-shaped bodies (cars, airfoils) and
    at worst flips a few points in concave pockets — acceptable for conditioning."""
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return None
    p = np.asarray(coords01, np.float64)
    k = min(int(k), len(p))
    if k < 3:
        return None
    _, idx = cKDTree(p).query(p, k=k, workers=-1)
    nb = p[idx]                                                     # (N,k,K)
    nb = nb - nb.mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb) / max(k - 1, 1)
    _, vecs = np.linalg.eigh(cov)                                   # ascending eigenvalues
    nrm = vecs[:, :, 0]                                             # smallest -> surface normal
    out = p - p.mean(0, keepdims=True)                              # outward reference
    flip = np.einsum("nk,nk->n", nrm, out) < 0
    nrm[flip] *= -1.0
    return nrm.astype(np.float32)


def estimate_curvature(coords01, normals=None, k=24):
    """Per-point [mean, gaussian] curvature from the POINT CLOUD alone, by fitting a local
    quadric height field over the k-NN neighbourhood in the tangent frame.

    Why (2026-08-03): curvature is what couples geometry to surface pressure — along a curved
    wall the normal pressure gradient is ~rho*U^2*kappa, and the tangential pressure gradient
    sets boundary-layer separation. We already carry curvature slots (geom ch5/ch6) but they are
    populated for drivaernet ONLY: shapenet's pyvista call fails (measured absmean 0.0) and the
    airfrans adapter never computed it. So the mechanism had no input on two of the three
    turbulent-aero families. This estimator needs coordinates only.

    Method: build a tangent basis from the normal, express neighbours as (u, v, w) with w along
    the normal, least-squares fit w = a*u^2 + b*u*v + c*v^2, then H = a + c and K = 4ac - b^2
    (the standard second-fundamental-form result for a Monge patch at the origin). Values are
    robustly scale-normalised the same way the pyvista path was, so the two are interchangeable."""
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return None
    p = np.asarray(coords01, np.float64)
    n_pts, K = p.shape
    if K != 3 or n_pts < k:
        return None                                   # a quadric patch needs a 2-manifold in 3D
    nrm = estimate_normals(p, k=16) if normals is None else np.asarray(normals, np.float64)
    if nrm is None:
        return None
    nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)
    _, idx = cKDTree(p).query(p, k=k, workers=-1)
    nb = p[idx] - p[:, None, :]                       # (N,k,3) neighbour offsets
    # tangent frame: any two unit vectors orthogonal to the normal
    a0 = np.tile(np.array([1.0, 0.0, 0.0]), (n_pts, 1))
    flip = np.abs(np.einsum("nk,nk->n", nrm, a0)) > 0.9
    a0[flip] = np.array([0.0, 1.0, 0.0])
    t1 = a0 - np.einsum("nk,nk->n", a0, nrm)[:, None] * nrm
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True) + 1e-12
    t2 = np.cross(nrm, t1)
    u = np.einsum("nkj,nj->nk", nb, t1)
    v = np.einsum("nkj,nj->nk", nb, t2)
    w = np.einsum("nkj,nj->nk", nb, nrm)
    A = np.stack([u * u, u * v, v * v], -1)           # (N,k,3) quadric design matrix
    AtA = np.einsum("nki,nkj->nij", A, A)
    Atw = np.einsum("nki,nk->ni", A, w)
    AtA += 1e-12 * np.eye(3)[None]
    try:
        coef = np.linalg.solve(AtA, Atw)              # (N,3) = [a, b, c]
    except np.linalg.LinAlgError:
        return None
    a, b, c = coef[:, 0], coef[:, 1], coef[:, 2]
    H = a + c                                         # mean curvature (x2, scale-normalised below)
    G = 4.0 * a * c - b * b                           # gaussian curvature
    out = []
    for x in (H, G):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        s = float(np.percentile(np.abs(x), 95)) + 1e-6
        out.append(np.clip(x / s, -4.0, 4.0))
    return np.stack(out, -1).astype(np.float32)


def sdf_volume(coords01, normals, res=48):
    """GINO-style SDF VOLUME from a co-dimension-1 point cloud.

    coords01 (N,K) in [0,1]^K + per-point outward normals (N,K) -> (res,)*K signed distance,
    normalized by the box diagonal. Distance via a KD-tree to the surface samples; SIGN from the
    dot product between the nearest-surface normal and the (cell - surface point) offset (outside
    => positive). Sign is robust enough for conditioning even where normals are noisy, and the
    magnitude — which is what fills the empty box — is exact up to the point sampling.
    Falls back to PCA-estimated normals when the mesh library gives none."""
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return None
    if normals is None:
        normals = estimate_normals(coords01)
        if normals is None:
            return None
    K = coords01.shape[1]
    ax = (np.arange(res) + 0.5) / res
    grid = np.stack(np.meshgrid(*([ax] * K), indexing="ij"), -1).reshape(-1, K)
    tree = cKDTree(coords01)
    dist, idx = tree.query(grid, k=1, workers=-1)
    off = grid - coords01[idx]                                      # cell - nearest surface point
    sign = np.sign(np.einsum("nk,nk->n", off, normals[idx]))
    sign[sign == 0] = 1.0
    sdf = (sign * dist / np.sqrt(K)).astype(np.float32)             # ÷ box diagonal
    return sdf.reshape(*([res] * K))


def _surface_curvature(mesh, n):
    """Per-point [mean, gaussian] curvature (n,2) via pyvista, robustly SCALE-NORMALIZED (raw
    curvature on unit-normalized coords can be ~1e3 and saturate). Each channel divided by its
    95th-|percentile| and clipped to +-4, so the feature is O(1) like the SDF. None on failure."""
    def _cv(src, t):
        c = np.nan_to_num(np.asarray(src.curvature(curv_type=t), np.float32).reshape(-1),
                          nan=0.0, posinf=0.0, neginf=0.0)
        s = float(np.percentile(np.abs(c), 95)) + 1e-6         # robust per-channel scale
        return np.clip(c / s, -4.0, 4.0)
    for src_fn in (lambda: mesh, lambda: mesh.extract_surface().triangulate()):  # quad -> tris (shapenet)
        try:
            src = src_fn()
            c = np.stack([_cv(src, "mean"), _cv(src, "gaussian")], -1).astype(np.float32)
            if c.shape[0] >= n:
                return c[:n]
        except Exception:
            continue
    return None


def _geom8(n, sdf=None, normals=None, curv=None, interior=None):
    """Assemble the canonical 8-channel geom (n,8): [SDF, 0(mask, loader fills), nx, ny, nz,
    mean_curv, gauss_curv, interior]. Missing components left 0. normals=(n,2|3), curv=(n,2)."""
    g = np.zeros((n, 8), np.float32)
    if sdf is not None:
        g[:, 0] = np.asarray(sdf, np.float32).reshape(-1)[:n]
    if normals is not None:
        nn = np.asarray(normals, np.float32); nc = min(nn.shape[1], 3)
        g[:, 2:2 + nc] = nn[:n, :nc]
    if curv is not None:
        cc = np.asarray(curv, np.float32).reshape(len(curv), -1); ncv = min(cc.shape[1], 2)
        g[:, 5:5 + ncv] = cc[:n, :ncv]
    if interior is not None:
        g[:, 7] = np.asarray(interior, np.float32).reshape(-1)[:n]
    return g


# --------------------------------------------------------------------------- #
# grid_npy : prose128 cache
# --------------------------------------------------------------------------- #
def _npy_units(spec):
    p = _path(spec)
    if os.path.isdir(p):                                   # shard dir
        shards = sorted(glob.glob(os.path.join(p, "shard_*.npy")))
        for si, sh in enumerate(shards):
            n = np.load(sh, mmap_mode="r").shape[0]
            for i in range(n):
                yield f"{si:04d}:{i}", (sh, i)
    else:
        n = np.load(p, mmap_mode="r").shape[0]
        for i in range(n):
            yield f"0:{i}", (p, i)


def _npy_load(spec, ref, cond):
    sh, i = ref
    arr = np.load(sh, mmap_mode="r")
    raw = np.asarray(arr[i], np.float32)                   # (T, H, W, C_raw)
    dims = raw.shape[1:3]
    tv = spec.t_valid
    geom = None
    if spec.geom_kind == "mask" and raw.shape[-1] > len(spec.channels):
        # static boundary/material mask rides along after the field channels (cfdbench):
        # route it to the canonical geometry layout (ch7 = interior/material mask), NOT a
        # field slot — it is problem geometry, not a physical quantity to predict.
        mask = raw[0, ..., len(spec.channels)].reshape(-1)  # static -> frame 0
        geom = np.zeros((mask.shape[0], 8), np.float32)
        geom[:, 7] = mask
    return grid_sample(spec.name, 2, raw, spec, cond, dims=dims, t_valid=tv, geom=geom)


# --------------------------------------------------------------------------- #
# grid_nchw : Poseidon .nc  (sample, time, [channel], x, y). location may be a
# single .nc OR a directory of .nc files (shards); dataset key auto-detected.
# --------------------------------------------------------------------------- #
def _nc_key(f):
    """The data variable = the highest-rank dataset (data/solution/velocity/...)."""
    best, best_nd = None, -1
    for k in f:
        nd = getattr(f[k], "ndim", 0)
        if nd > best_nd:
            best, best_nd = k, nd
    return best


def _nc_files(spec):
    p = _path(spec)
    if os.path.isdir(p):
        return sorted(glob.glob(os.path.join(p, "*.nc")))
    return [p]


def _nc_units(spec):
    import h5py
    for fi, fp in enumerate(_nc_files(spec)):
        with h5py.File(fp, "r") as f:
            n = f[_nc_key(f)].shape[0]
        for i in range(n):
            yield f"{fi:04d}:{i}", (fp, i)


def _nc_load(spec, ref, cond):
    import h5py
    fp, i = ref
    with h5py.File(fp, "r") as f:
        fkey = spec.nc_keys[0] if spec.nc_keys else _nc_key(f)      # explicit field key for BVPs
        a = np.asarray(f[fkey][i], np.float32)             # (T,[C],x,y) or (x,y)
        src = np.asarray(f[spec.nc_keys[1]][i], np.float32) if spec.nc_keys else None
    if a.ndim == 2:                                        # steady 2D field (x,y) -> (T=1,C=1,x,y)
        a = a[None, None]
    elif a.ndim == 3:                                      # (T,x,y) single channel -> (T,1,x,y)
        a = a[:, None]
    raw = np.transpose(a, (0, 2, 3, 1))                    # (T,x,y,C_raw)
    geom = None
    if src is not None:                                    # elliptic BVP: source term -> conditioning
        s = src.reshape(-1, 1)                             # (Ncell,1), cell-raveled (C-order = fields)
        geom = (s / (s.std() + 1e-6)).astype(np.float32)   # normalized source as an input channel
    if spec.coef_file:                                     # per-sample COEFFICIENT FIELD (closes the IVP)
        cpath, ckey = spec.coef_file
        with h5py.File(os.path.join(CORPUS, cpath), "r") as cf:
            c = np.asarray(cf[ckey][i], np.float32).reshape(-1, 1)
        c = (c - c.mean()) / (c.std() + 1e-6)              # zero-mean/unit-var: the LAYOUT is the signal
        geom = c if geom is None else np.concatenate([geom, c], -1)
    return grid_sample(spec.name, 2, raw, spec, cond, dims=raw.shape[1:3], geom=geom)


# --------------------------------------------------------------------------- #
# pdebench3d : PDEBench 3D CNS hdf5 (Vx,Vy,Vz,density,pressure keys)
# --------------------------------------------------------------------------- #
_3D_KEYS = ["Vx", "Vy", "Vz", "density", "pressure"]


def _filename_params(spec):
    """Parse M/Eta/Zeta from the filename for pdebench3d/mesh filename params."""
    base = os.path.basename(spec.location)
    out = {}
    for name, pat in [("mach", r"M([0-9.]+)"), ("eta", r"Eta([0-9.e+-]+)"),
                      ("zeta", r"Zeta([0-9.e+-]+)")]:
        m = re.search(pat, base)
        if m:
            out[name] = float(m.group(1))
    return out


def _pdebench3d_units(spec):
    import h5py
    with h5py.File(_path(spec), "r") as f:
        n = f["Vx"].shape[0]
    for i in range(n):
        yield f"0:{i}", i


def _pdebench3d_load(spec, ref, cond):
    import h5py
    with h5py.File(_path(spec), "r") as f:
        chans = [np.asarray(f[k][ref], np.float32) for k in _3D_KEYS]   # each (T,X,Y,Z)
    raw = np.stack(chans, -1)                              # (T,X,Y,Z,5)
    cond = SY.encode(spec, param_values=_filename_params(spec))         # resolve filename params
    return grid_sample(spec.name, 3, raw, spec, cond, dims=raw.shape[1:4])


# --------------------------------------------------------------------------- #
# mesh_geofno_* : structured .npy (airfoil/elasticity/pipe) — coords + field
# --------------------------------------------------------------------------- #
def _geofno_files(spec, which):
    d = _path(spec)
    return {k: os.path.join(d, v) for k, v in which.items()}


def _geofno_airfoil_units(spec):
    q = np.load(os.path.join(_path(spec), "naca_interp", "NACA_Q_interp.npy"), mmap_mode="r")
    for i in range(q.shape[0]):
        yield f"0:{i}", i


def _geofno_airfoil_load(spec, ref, cond):
    d = os.path.join(_path(spec), "naca_interp")
    X = np.load(os.path.join(d, "NACA_X_interp.npy"), mmap_mode="r")[ref]   # (Hx,Wx)
    Y = np.load(os.path.join(d, "NACA_Y_interp.npy"), mmap_mode="r")[ref]
    Q = np.load(os.path.join(d, "NACA_Q_interp.npy"), mmap_mode="r")[ref]   # Mach field
    coords = np.stack([np.ravel(X), np.ravel(Y)], -1)
    coords, _ = normalize_coords(coords)
    fields = np.ravel(Q).astype(np.float32)[None, :, None]                 # (1,N,1)
    # NO geometry: the C-grid's airfoil surface index is ambiguous (wall=row0 gave a constant
    # normal -> that row is a far-field/wake boundary, not the body). Feeding a wrong-wall SDF would
    # build a garbage BL mask + mis-bias near-wall collocation, so emit none until the grid
    # convention is confirmed. Shape still carries via the (already deformed) coordinates.
    return point_sample(spec.name, 2, coords, fields, spec, cond, geom=None)


# --------------------------------------------------------------------------- #
# mesh_geofno_elasticity : grid-interpolated unit-cell stress (npy)
# --------------------------------------------------------------------------- #
def _elasticity_batches(spec):
    d = os.path.join(_path(spec), "Interp")
    sig = sorted(glob.glob(os.path.join(d, "*_sigma_*_interp.npy")),
                 key=lambda p: int(re.search(r"_sigma_(\d+)_", p).group(1)))
    return d, sig


def _geofno_elasticity_units(spec):
    """Each *_sigma_<b>_interp.npy is a batch of unit cells laid out (H,W,S); unit=(batch,i)."""
    _, sig = _elasticity_batches(spec)
    for p in sig:
        S = np.load(p, mmap_mode="r").shape[-1]                     # samples on LAST axis
        b = re.search(r"_sigma_(\d+)_", p).group(1)
        for i in range(S):
            yield f"{b}:{i}", (p, i)


def _geofno_elasticity_load(spec, ref, cond):
    p, i = ref
    d = os.path.dirname(p)
    b = re.search(r"_sigma_(\d+)_", p).group(1)
    sigma = np.asarray(np.load(p, mmap_mode="r")[..., i])          # (H,W)
    mask = np.asarray(np.load(os.path.join(d, f"Random_UnitCell_mask_{b}_interp.npy"),
                              mmap_mode="r")[..., i])
    H, W = sigma.shape
    yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing="ij")
    coords = np.stack([xx.ravel(), yy.ravel()], -1).astype(np.float32)
    fields = sigma.ravel().astype(np.float32)[None, :, None]        # (1,N,1)
    try:                                                            # signed distance from material boundary
        from scipy.ndimage import distance_transform_edt
        mb = mask.astype(bool)
        sdf_g = ((distance_transform_edt(~mb) - distance_transform_edt(mb))
                 .astype(np.float32) / float(H))                    # outside(+) - inside(-), grid units
        gy, gx = np.gradient(sdf_g)                                 # normal ~ grad(SDF)
        nmag = np.sqrt(gx ** 2 + gy ** 2) + 1e-8
        nx, ny = gx / nmag, gy / nmag
        curv_g = np.gradient(nx, axis=1) + np.gradient(ny, axis=0)  # curvature ~ div(normal)
        geom = _geom8(len(coords), sdf=sdf_g.ravel(),
                      normals=np.stack([nx.ravel(), ny.ravel()], -1).astype(np.float32),
                      curv=curv_g.reshape(-1, 1).astype(np.float32), interior=mask.ravel())
    except Exception:                                              # fall back: material mask only
        geom = _geom8(len(coords), interior=mask.ravel())
    return point_sample(spec.name, 2, coords, fields, spec, cond, geom=geom)


# --------------------------------------------------------------------------- #
# mesh_geofno_pipe : deformed-grid pipe flow (npy). X,Y = node coords, Q[:,0] = velocity field
# --------------------------------------------------------------------------- #
def _geofno_pipe_units(spec):
    n = np.load(os.path.join(_path(spec), "Pipe_X.npy"), mmap_mode="r").shape[0]
    for i in range(n):
        yield f"0:{i}", i


def _geofno_pipe_load(spec, ref, cond):
    d = _path(spec)
    X = np.load(os.path.join(d, "Pipe_X.npy"), mmap_mode="r")[ref]      # (H,W)
    Y = np.load(os.path.join(d, "Pipe_Y.npy"), mmap_mode="r")[ref]
    Q = np.load(os.path.join(d, "Pipe_Q.npy"), mmap_mode="r")[ref, 0]   # (H,W) streamwise velocity
    coords = np.stack([np.ravel(X), np.ravel(Y)], -1)
    coords, _ = normalize_coords(coords)
    fields = np.ravel(Q).astype(np.float32)[None, :, None]             # (1,N,1) -> slot 0
    try:                                                              # SDF to pipe walls (top+bottom edges)
        H, W = X.shape
        grid = coords.reshape(H, W, 2)
        wall = np.concatenate([grid[0, :, :], grid[-1, :, :]], 0).reshape(-1, 2)
        try:
            from scipy.spatial import cKDTree
            sdf_flat = cKDTree(wall).query(coords)[0].astype(np.float32)  # positive nearest-wall dist
        except Exception:
            d = np.sqrt(((coords[:, None, :] - wall[None, :, :]) ** 2).sum(-1))
            sdf_flat = d.min(1).astype(np.float32)
        sdf_g = sdf_flat.reshape(H, W)
        gy, gx = np.gradient(sdf_g)                                  # normal ~ grad(SDF)
        nmag = np.sqrt(gx ** 2 + gy ** 2) + 1e-8
        nx, ny = gx / nmag, gy / nmag
        curv_g = np.gradient(nx, axis=1) + np.gradient(ny, axis=0)   # curvature ~ div(normal)
        nrm_flat = np.stack([nx.ravel(), ny.ravel()], -1).astype(np.float32)
        curv_flat = curv_g.reshape(-1, 1).astype(np.float32)
        geom = _geom8(len(coords), sdf=sdf_flat, normals=nrm_flat, curv=curv_flat)
    except Exception:
        geom = None
    return point_sample(spec.name, 2, coords, fields, spec, cond, geom=geom)


# --------------------------------------------------------------------------- #
# mesh_airfrans : OpenFOAM RANS volume mesh (pyvista .vtu), implicit_distance = SDF
# --------------------------------------------------------------------------- #
def _airfrans_units(spec):
    for d in sorted(glob.glob(os.path.join(_path(spec), "airFoil2D_*"))):
        yield os.path.basename(d), d


def _airfrans_load(spec, ref, cond):
    import pyvista as pv
    name = os.path.basename(ref)
    vtu = os.path.join(ref, f"{name}_internal.vtu")
    m = pv.read(vtu)
    pts = np.asarray(m.points)[:, :2]                              # 2D airfoil
    coords, _ = normalize_coords(pts)
    U = np.asarray(m.point_data["U"])[:, :2]                       # Ux, Uy
    p = np.asarray(m.point_data["p"]).reshape(-1, 1)
    raw = np.concatenate([U, p], -1)[None]                         # (1,N,3) -> slots 0,1,4
    sdf = np.asarray(m.point_data["implicit_distance"]).reshape(-1, 1).astype(np.float32)
    nrm = _sdf_gradient_normal(m, 2)                               # (N,2) geometry normal or None
    geom = _geom8(len(coords), sdf=sdf.reshape(-1), normals=nrm,   # ch0 SDF, ch2/3 nx/ny (2D: nz=0)
                  curv=estimate_curvature(coords, nrm))            # ch5-6: curvature drives dp/dn~rho U^2 k
    toks = [float(x) for x in re.findall(r"-?\d+\.?\d*", name)]
    pv_ = {n: toks[i] for i, n in enumerate(spec.param_names) if i < len(toks)}
    cond = SY.encode(spec, param_values=pv_)
    return point_sample(spec.name, 2, coords, raw.astype(np.float32), spec, cond, geom=geom)


# --------------------------------------------------------------------------- #
# mesh_shapenet_car : per-design car surface (Umetani), .vtk mesh + press.npy
# --------------------------------------------------------------------------- #
def _shapenet_extract_tars(base):
    """Extract any param*.tar.gz not yet unpacked (one-time, idempotent)."""
    import tarfile
    for tg in sorted(glob.glob(os.path.join(base, "param*.tar.gz"))):
        pdir = tg[:-len(".tar.gz")]
        if os.path.isdir(pdir):
            continue
        try:
            with tarfile.open(tg) as t:
                t.extractall(base)
        except Exception:
            pass


def _shapenet_complete(design):
    """A design is usable iff it has surface pressure + a mesh .vtk."""
    return (os.path.exists(os.path.join(design, "press.npy"))
            and (os.path.exists(os.path.join(design, "quadpress_smpl.vtk"))
                 or glob.glob(os.path.join(design, "*.vtk"))))


def _shapenet_car_units(spec):
    base = os.path.join(_path(spec), "training_data")
    _shapenet_extract_tars(base)
    for pdir in sorted(glob.glob(os.path.join(base, "param*"))):
        if not os.path.isdir(pdir):
            continue
        for design in sorted(glob.glob(os.path.join(pdir, "*"))):
            if os.path.isdir(design) and _shapenet_complete(design):
                yield f"{os.path.basename(pdir)}:{os.path.basename(design)}", design


def _shapenet_car_load(spec, ref, cond):
    import pyvista as pv
    press = np.load(os.path.join(ref, "press.npy")).reshape(-1)     # (N,) surface pressure
    qp = os.path.join(ref, "quadpress_smpl.vtk")
    mesh = pv.read(qp) if os.path.exists(qp) else pv.read(glob.glob(os.path.join(ref, "*.vtk"))[0])
    pts = np.asarray(mesh.points)
    n = min(len(pts), len(press))
    coords, _ = normalize_coords(pts[:n])
    fields = press[:n].astype(np.float32)[None, :, None]            # (1,N,1) -> slot 4
    nrm = _surface_normals(mesh, n)
    if nrm is None:                                                # mesh-lib normals unavailable
        nrm = estimate_normals(coords)                             # -> point-cloud PCA normals
    crv = _surface_curvature(mesh, n)
    if crv is None or not np.any(crv):                             # measured 0 for shapenet's quad VTK
        crv = estimate_curvature(coords, nrm)                      # -> local quadric fit
    geom = _geom8(n, normals=nrm,                                  # surface family: no per-point SDF
                  curv=crv)                                        # ch2-4 normals, ch5-6 curvature
    sv = sdf_volume(coords, nrm, res=_SDF_RES)                     # dense volumetric shape context
    return point_sample(spec.name, 3, coords, fields, spec, cond, geom=geom, sdf_vol=sv)


# --------------------------------------------------------------------------- #
# mesh_drivaernet : DrivAerNet++ surface pressure. Raw = pressure/<cat>.zip, each
# unpacking to PressureVTK/<cat>/<cat>_<id>.vtk (surface points + pressure array;
# the STL meshes zips are redundant for a surface-pressure task). One design = one unit.
# --------------------------------------------------------------------------- #
_SDF_RES = 48                 # stored SDF-volume resolution (resampled to the model box at load);
                              # 48^3 float32 = 442 KB/design — 3.6 GB over drivaernet's 8129 designs
_DRIVAER_CAP = 32768          # surface points kept per design (encoder subsamples to enc_n anyway;
                              # ~520k raw pts/design would bloat the cache ~16x for no training gain)


def _vtk_pressure(mesh):
    """Extract the surface pressure array from a DrivAerNet++ VTK (point data preferred;
    cell data interpolated to points; else the first/active scalar). Name auto-detected."""
    def _find(store):
        for k in store.keys():
            if "p" == k.lower() or "press" in k.lower():
                return k
        return None
    k = _find(mesh.point_data)
    if k is not None:
        return np.asarray(mesh.point_data[k]).reshape(-1)
    k = _find(mesh.cell_data)
    if k is not None:
        return np.asarray(mesh.cell_data_to_point_data().point_data[k]).reshape(-1)
    if mesh.active_scalars is not None:
        return np.asarray(mesh.active_scalars).reshape(-1)
    if len(mesh.point_data):
        return np.asarray(mesh.point_data[list(mesh.point_data.keys())[0]]).reshape(-1)
    raise ValueError(f"no pressure array (point_data={list(mesh.point_data.keys())})")


def _drivaernet_units(spec):
    """Enumerate designs by READING zip central directories (no bulk extraction): one unit per
    PressureVTK/<cat>/<cat>_<id>.vtk. ref = (zip_path, member) so load extracts just that design."""
    import zipfile
    for zf in sorted(glob.glob(os.path.join(_path(spec), "*.zip"))):
        with zipfile.ZipFile(zf) as z:
            for m in z.namelist():
                if m.endswith(".vtk"):
                    yield os.path.splitext(os.path.basename(m))[0], (zf, m)


def _drivaernet_load(spec, ref, cond):
    import tempfile
    import zipfile

    import pyvista as pv
    zf, member = ref
    with zipfile.ZipFile(zf) as z:                                   # extract just this one VTK
        data = z.read(member)
    tmp = tempfile.NamedTemporaryFile(suffix=".vtk", delete=False)
    try:
        tmp.write(data); tmp.close()
        mesh = pv.read(tmp.name)
    finally:
        os.remove(tmp.name)
    pts = np.asarray(mesh.points)                                    # (N,3) surface coords
    p = _vtk_pressure(mesh)                                          # (N,) surface pressure
    n = min(len(pts), len(p))
    pts, p = pts[:n], p[:n]
    nrm = _surface_normals(mesh, n)                                  # (n,3) surface normals or None
    crv = _surface_curvature(mesh, n)                               # (n,2) [mean, gauss] curvature or None
    if n > _DRIVAER_CAP:                                             # cap points (deterministic per design)
        import zlib
        rng = np.random.default_rng(zlib.crc32(os.path.basename(member).encode()))
        idx = rng.choice(n, _DRIVAER_CAP, replace=False)
        pts, p = pts[idx], p[idx]
        if nrm is not None:
            nrm = nrm[idx]                                           # subsample normals with points
        if crv is not None:
            crv = crv[idx]                                           # subsample curvature with SAME idx
    coords, _ = normalize_coords(pts)
    fields = p.astype(np.float32)[None, :, None]                     # (1,N,1) -> slot 4
    if nrm is None:                                                  # mesh-lib normals unavailable
        nrm = estimate_normals(coords)                               # -> point-cloud PCA normals
    if crv is None or not np.any(crv):
        crv = estimate_curvature(coords, nrm)                        # -> local quadric fit
    geom = _geom8(len(coords), normals=nrm, curv=crv)              # surface family: ch2-4 nrm, ch5-6 curv
    sv = sdf_volume(coords, nrm, res=_SDF_RES)                       # dense volumetric shape context
    return point_sample(spec.name, 3, coords, fields, spec, cond, geom=geom, sdf_vol=sv)


# --------------------------------------------------------------------------- #
# adapter registry
# --------------------------------------------------------------------------- #
_UNITS = {
    "grid_npy": _npy_units, "grid_nchw": _nc_units, "pdebench3d": _pdebench3d_units,
    "mesh_geofno_airfoil": _geofno_airfoil_units,
    "mesh_geofno_elasticity": _geofno_elasticity_units,
    "mesh_airfrans": _airfrans_units,
    "mesh_shapenet_car": _shapenet_car_units,
    "mesh_geofno_pipe": _geofno_pipe_units,
    "mesh_drivaernet": _drivaernet_units,
}
_LOAD = {
    "grid_npy": _npy_load, "grid_nchw": _nc_load, "pdebench3d": _pdebench3d_load,
    "mesh_geofno_airfoil": _geofno_airfoil_load,
    "mesh_geofno_elasticity": _geofno_elasticity_load,
    "mesh_airfrans": _airfrans_load,
    "mesh_shapenet_car": _shapenet_car_load,
    "mesh_geofno_pipe": _geofno_pipe_load,
    "mesh_drivaernet": _drivaernet_load,
}
# DEFERRED: mesh_geofno_pipe (only PDFs present in the drive folder — data not downloaded).
# The adapters above exercise every schema path (grid 2D/3D + point mesh, VTU/npy/VTK/STL).

IMPLEMENTED = set(_UNITS)


def units(spec):
    """Yield (unit_id, ref) for a family (headers only)."""
    if spec.fmt not in _UNITS:
        raise NotImplementedError(f"adapter for fmt={spec.fmt} ({spec.name}) not yet implemented")
    return _UNITS[spec.fmt](spec)


def load(spec, ref):
    """Materialize one unit -> UnifiedSample."""
    cond = SY.encode(spec)
    return _LOAD[spec.fmt](spec, ref, cond)
