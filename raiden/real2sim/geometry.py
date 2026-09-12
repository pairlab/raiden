"""Small geometry helpers shared by the real2sim calibration scripts.

Conventions
-----------
* Camera frames are OpenCV optical frames: +x right, +y down, +z forward.
* ``T_a_b`` is a 4x4 homogeneous transform mapping points in frame ``b`` to frame ``a``.
* The robot *base* frame is the i2rt YAM model base (z up, +x along the folded arm at home).
* A plane is ``(n, d)`` with ``n . p + d = 0``; we orient ``n`` so that the camera is on the
  positive side (``d > 0``), i.e. ``n`` points from the table towards the camera / up.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

# OpenCV optical frame -> MuJoCo camera frame (MuJoCo cameras look along -z with +y up).
CV_TO_MJ = np.diag([1.0, -1.0, -1.0])


def depth_to_points(depth_m: np.ndarray, fx: float, fy: float, cx: float, cy: float):
    """Back-project a metric depth image into an (H, W, 3) array of camera-frame points."""
    h, w = depth_m.shape
    yy, xx = np.mgrid[:h, :w]
    z = depth_m
    return np.stack([(xx - cx) * z / fx, (yy - cy) * z / fy, z], axis=-1)


def fit_plane_ransac(
    points: np.ndarray, thresh: float = 0.006, iters: int = 3000, seed: int = 0
):
    """RANSAC plane fit followed by a least-squares refinement on the inliers.

    Returns ``(n, d, inlier_mask)`` with ``d > 0`` (camera on the positive side).
    """
    rng = np.random.default_rng(seed)
    best = (0, None, None)
    for _ in range(iters):
        s = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        d = -n @ s[0]
        count = int((np.abs(points @ n + d) < thresh).sum())
        if count > best[0]:
            best = (count, n, d)
    _, n, d = best
    inl = np.abs(points @ n + d) < thresh
    P = points[inl]
    mu = P.mean(0)
    _, v = np.linalg.eigh(np.cov((P - mu).T))
    n = v[:, 0]
    d = -n @ mu
    if d < 0:
        n, d = -n, -d
    inl = np.abs(points @ n + d) < thresh
    return n, d, inl


def rotation_aligning(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation R with R @ a ∥ b (unit vectors)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(a @ b)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # 180 degree flip about any axis orthogonal to a
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(np.pi * axis).as_matrix()
    return Rotation.from_rotvec(v / s * np.arctan2(s, c)).as_matrix()


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    return make_T(R.T, -R.T @ t)


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def project(K: np.ndarray, pts_cam: np.ndarray):
    """Project camera-frame points with pinhole K. Returns (uv, z)."""
    z = pts_cam[:, 2]
    u = K[0, 0] * pts_cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / z + K[1, 2]
    return np.stack([u, v], -1), z


def kabsch(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rigid transform T with T(src) ≈ dst (least squares)."""
    ms, md = src.mean(0), dst.mean(0)
    H = (src - ms).T @ (dst - md)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return make_T(R, md - R @ ms)


def icp(
    model: np.ndarray,
    scene: np.ndarray,
    T_init: np.ndarray,
    iters: int = 60,
    trim: float = 0.8,
    max_dist: float | None = 0.08,
):
    """Trimmed point-to-point ICP aligning ``model`` onto ``scene``.

    ``T_init`` maps model -> scene. Returns ``(T, rms_of_kept_pairs, n_kept)``.
    The scene is a partial view, so we match every *model* point to its nearest scene point
    only when the model point is plausibly visible; using scene->model matching would pull the
    model toward unmodelled scene clutter. We use both directions with trimming to balance.
    """
    tree_scene = cKDTree(scene)
    T = T_init.copy()
    rms = np.inf
    n_kept = 0
    for _ in range(iters):
        m = transform_points(T, model)
        tree_model = cKDTree(m)
        # model -> scene
        d1, i1 = tree_scene.query(m)
        # scene -> model
        d2, i2 = tree_model.query(scene)
        src = np.concatenate([model, model[i2]])
        dst = np.concatenate([scene[i1], scene])
        d = np.concatenate([d1, d2])
        keep = d <= np.quantile(d, trim)
        if max_dist is not None:
            keep &= d < max_dist
        if keep.sum() < 50:
            break
        T_new = kabsch(src[keep], dst[keep])
        delta = np.linalg.norm(T_new[:3, 3] - T[:3, 3]) + np.linalg.norm(
            T_new[:3, :3] - T[:3, :3]
        )
        T = T_new
        rms = float(np.sqrt(np.mean(d[keep] ** 2)))
        n_kept = int(keep.sum())
        if delta < 1e-6:
            break
    return T, rms, n_kept


def sample_mesh_surface(
    verts: np.ndarray, faces: np.ndarray, n: int, rng
) -> np.ndarray:
    """Uniformly sample ``n`` points on a triangle mesh surface."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    if areas.sum() <= 0:
        return verts[rng.choice(len(verts), n)]
    idx = rng.choice(len(faces), n, p=areas / areas.sum())
    r1 = np.sqrt(rng.random(n))
    r2 = rng.random(n)
    return (
        (1 - r1)[:, None] * v0[idx]
        + (r1 * (1 - r2))[:, None] * v1[idx]
        + (r1 * r2)[:, None] * v2[idx]
    )


def robot_surface_points(
    model, data, n_total: int = 30000, seed: int = 0, body_filter=None
) -> np.ndarray:
    """Sample surface points from every mesh geom of a MuJoCo model in the world frame.

    Includes both visual and collision mesh geoms (i2rt models only have mesh geoms).
    """
    import mujoco

    rng = np.random.default_rng(seed)
    geoms = []
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if body_filter is not None and not body_filter(
            model.body(model.geom_bodyid[g]).name
        ):
            continue
        geoms.append(g)
    # area-weighted split of samples between geoms
    per_geom = []
    for g in geoms:
        mid = model.geom_dataid[g]
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        verts = model.mesh_vert[va : va + vn]
        faces = model.mesh_face[fa : fa + fn]
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        per_geom.append(0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum())
    per_geom = np.array(per_geom)
    counts = np.maximum(50, (n_total * per_geom / per_geom.sum()).astype(int))
    out = []
    for g, cnt in zip(geoms, counts):
        mid = model.geom_dataid[g]
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        verts = model.mesh_vert[va : va + vn]
        faces = model.mesh_face[fa : fa + fn]
        pts = sample_mesh_surface(verts, faces, int(cnt), rng)
        R = data.geom_xmat[g].reshape(3, 3)
        out.append(pts @ R.T + data.geom_xpos[g])
    return np.concatenate(out)
