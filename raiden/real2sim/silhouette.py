"""Silhouette-based robot pose fit for a fixed camera.

RealSense depth is unreliable on the matte-black YAM links and the lab has dark partitions at
the same depth as the arm, so ICP on depth is fragile. Instead we render the robot model's
silhouette from candidate camera poses and maximise IoU with a colour-based robot mask.

The table plane (from depth, which is solid on the wooden table) pins roll, pitch and the
height reference, so the search is over (x, y, yaw, z_above_table) only.
"""

from __future__ import annotations

import os

import cv2
import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from raiden.real2sim.geometry import CV_TO_MJ, make_T


class SilhouetteRenderer:
    """Render the robot segmentation mask through a pinhole camera with given intrinsics."""

    def __init__(
        self,
        xml_path: str,
        K: np.ndarray,
        width: int,
        height: int,
        joints,
        gripper=0.04,
    ):
        os.environ.setdefault("MUJOCO_GL", "glfw")
        xml = open(xml_path).read()
        fovy = 2 * np.degrees(np.arctan2(height / 2, K[1, 1]))
        cam = f'<camera name="calib" pos="0 0 1" fovy="{fovy:.6f}"/>'
        assert "<worldbody>" in xml
        xml = xml.replace("<worldbody>", "<worldbody>" + cam, 1)
        xml = xml.replace(
            "<worldbody>",
            f'<visual><global offwidth="{max(width, 640)}" offheight="{max(height, 480)}"/></visual><worldbody>',
            1,
        )
        # relative mesh paths: resolve against the original xml directory
        self.model = mujoco.MjModel.from_xml_string(xml, self._assets(xml_path))
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:6] = joints
        if self.model.nq > 6:
            self.data.qpos[6:] = gripper
        mujoco.mj_forward(self.model, self.data)
        self.cam_id = self.model.camera("calib").id
        self.width, self.height = width, height
        self.K = K
        self.renderer = mujoco.Renderer(self.model, height, width)
        self.renderer.enable_segmentation_rendering()
        self.depth_renderer = mujoco.Renderer(self.model, height, width)
        self.depth_renderer.enable_depth_rendering()
        # geometry only: site spheres (tcp/grasp sites) must not end up in the silhouette
        self.scene_option = mujoco.MjvOption()
        self.scene_option.sitegroup[:] = 0
        # principal point offset relative to the render centre
        self.dx = K[0, 2] - (width / 2 - 0.5)
        self.dy = K[1, 2] - (height / 2 - 0.5)
        # non-square pixels are not supported by mujoco's camera; warn if fx != fy notably
        self.aspect_scale = K[0, 0] / K[1, 1]

    @staticmethod
    def _assets(xml_path):
        """Load every file under the xml directory tree so relative mesh paths resolve."""
        root = os.path.dirname(os.path.abspath(xml_path))
        assets = {}
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if fn.endswith((".stl", ".obj", ".msh")):
                    p = os.path.join(dp, fn)
                    rel = os.path.relpath(p, root)
                    if rel in assets or fn in assets:
                        continue  # the xml may live in /tmp; ignore unrelated duplicates
                    assets[rel] = open(p, "rb").read()
                    assets[fn] = assets[rel]
        return assets

    def _set_camera(self, T_cam_base):
        T_base_cam = np.linalg.inv(T_cam_base)
        R_mj = T_base_cam[:3, :3] @ CV_TO_MJ
        self.model.cam_pos[self.cam_id] = T_base_cam[:3, 3]
        self.model.cam_quat[self.cam_id] = np.roll(
            Rotation.from_matrix(R_mj).as_quat(), 1
        )
        mujoco.mj_forward(self.model, self.data)

    def _warp(self, img, interp=cv2.INTER_NEAREST):
        M = np.float32(
            [
                [
                    self.aspect_scale,
                    0,
                    self.dx + (1 - self.aspect_scale) * self.width / 2,
                ],
                [0, 1, self.dy],
            ]
        )
        return cv2.warpAffine(img, M, (self.width, self.height), flags=interp)

    def render_mask(self, T_cam_base: np.ndarray) -> np.ndarray:
        """Boolean silhouette of the robot for camera pose T_cam_base (OpenCV camera frame)."""
        self._set_camera(T_cam_base)
        self.renderer.update_scene(
            self.data, camera="calib", scene_option=self.scene_option
        )
        seg = self.renderer.render()
        return self._warp((seg[:, :, 0] >= 0).astype(np.uint8)).astype(bool)

    def render_depth(self, T_cam_base: np.ndarray) -> np.ndarray:
        """Metric depth (m, along the optical axis) of the robot; 0 where no robot."""
        self._set_camera(T_cam_base)
        self.depth_renderer.update_scene(
            self.data, camera="calib", scene_option=self.scene_option
        )
        z = self.depth_renderer.render().astype(np.float32)
        z[z > 50] = 0  # far plane / background
        return self._warp(z)


def pose_from_params(params, n, d, ex0, ey0):
    """Build T_cam_base from (x, y, yaw, z) in the table frame.

    The table frame has origin at the foot of the camera's perpendicular, z = table normal n
    (pointing up / towards the camera), and in-plane axes ex0, ey0.
    """
    x, y, yaw, z = params
    ex = np.cos(yaw) * ex0 + np.sin(yaw) * ey0
    ey = np.cross(n, ex)
    R = np.stack([ex, ey, n], 1)
    foot = -d * n  # point on the plane closest to the camera origin
    t = foot + x * ex0 + y * ey0 + z * n
    return make_T(R, t)


def fit_silhouette(
    renderer: SilhouetteRenderer,
    target: np.ndarray,
    roi: np.ndarray,
    n,
    d,
    init_xy,
    z_init=0.08,
    yaw_inits=None,
    verbose=True,
    depth=None,
    depth_weight=2.0,
    landmarks=None,
    landmark_weight=0.004,
    rail_dir_cam=None,
    rail_weight=0.01,
    init_params=None,
    z_fixed=None,
):
    """Maximise IoU(render, target) inside roi over (x, y, yaw, z). Returns (T_cam_base, score, params).

    If ``depth`` (measured metric depth image) is given, a depth-consistency term is added:
    mean |z_render - z_measured| over pixels where both the render and the measurement cover
    the robot mask. This anchors the fit along the viewing direction, which a silhouette alone
    constrains weakly.

    ``rail_dir_cam`` (unit vector, camera frame) is the measured direction of the extrusion rail
    the base is bolted to. The base is square on the rail, so the base y axis must be parallel
    to it; the angle between them (degrees) is penalised with ``rail_weight``. This pins yaw far
    better than the nearly symmetric arm silhouette does.

    ``init_params`` (x, y, yaw, z) skips the global coarse search and only scans yaw around the
    given value (+-40 deg) before the local refinement. ``z_fixed`` pins the base height above
    the table (e.g. measured by the wrist camera), leaving (x, y, yaw) free.
    """
    ex0 = np.cross(n, [0.0, 0.0, 1.0])
    ex0 /= np.linalg.norm(ex0)
    ey0 = np.cross(n, ex0)
    tgt = target & roi
    dvalid = (depth > 0.1) & tgt if depth is not None else None
    # landmarks: [(name, (u, v))] where name is a site or body of the robot model; the 3D point
    # (base frame) is projected with the camera intrinsics and compared to the clicked pixel.
    lm_pts, lm_uv = [], []
    for name, uv in landmarks or []:
        if name == "base":
            lm_pts.append(
                np.zeros(3)
            )  # base frame origin (i2rt base geom sits in the worldbody)
        else:
            try:
                lm_pts.append(renderer.data.site(name).xpos.copy())
            except KeyError:
                lm_pts.append(renderer.data.body(name).xpos.copy())
        lm_uv.append(uv)
    lm_pts = np.array(lm_pts).reshape(-1, 3)
    lm_uv = np.array(lm_uv, float).reshape(-1, 2)
    K = renderer.K

    def landmark_error(T):
        if len(lm_pts) == 0:
            return 0.0
        pc = lm_pts @ T[:3, :3].T + T[:3, 3]
        if np.any(pc[:, 2] <= 0.05):
            return 500.0
        uv = pc @ K.T
        uv = uv[:, :2] / uv[:, 2:3]
        return float(np.linalg.norm(uv - lm_uv, axis=1).mean())

    def rail_error(T):
        if rail_dir_cam is None:
            return 0.0
        y_cam = T[:3, :3] @ np.array([0.0, 1.0, 0.0])
        c = abs(float(y_cam @ rail_dir_cam))
        return float(np.degrees(np.arccos(np.clip(c, 0.0, 1.0))))

    def iou(params):
        if z_fixed is not None:
            params = np.array([params[0], params[1], params[2], z_fixed])
        T = pose_from_params(params, n, d, ex0, ey0)
        m = renderer.render_mask(T) & roi
        inter = (m & tgt).sum()
        union = (m | tgt).sum()
        score = inter / union if union else 0.0
        if depth is not None:
            zr = renderer.render_depth(T)
            both = (zr > 0) & dvalid
            if both.sum() > 200:
                err = np.abs(zr[both] - depth[both])
                err = np.minimum(err, 0.15)  # robust cap
                score -= depth_weight * float(err.mean())
            else:
                score -= depth_weight * 0.15
        score -= landmark_weight * landmark_error(T)
        score -= rail_weight * rail_error(T)
        return score

    best = None
    if init_params is not None:
        p0 = np.asarray(init_params, float)
        for dyaw in np.radians(np.arange(-40, 41, 5)):
            for dx in (-0.06, 0.0, 0.06):
                for dy in (-0.06, 0.0, 0.06):
                    p = p0 + np.array([dx, dy, dyaw, 0.0])
                    v = iou(p)
                    if best is None or v > best[0]:
                        best = (v, p)
    else:
        if yaw_inits is None:
            yaw_inits = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        # coarse search: yaw x xy grid (+-24 cm, 6 cm steps) around the mask-centroid guess
        offsets = np.arange(-0.18, 0.181, 0.06)
        for yaw in yaw_inits:
            for dx in offsets:
                for dy in offsets:
                    p = np.array([init_xy[0] + dx, init_xy[1] + dy, yaw, z_init])
                    v = iou(p)
                    if best is None or v > best[0]:
                        best = (v, p)
    if verbose:
        print(
            f"  coarse best score {best[0]:.3f} at x={best[1][0]:+.3f} y={best[1][1]:+.3f} "
            f"yaw={np.degrees(best[1][2]):.1f} z={best[1][3]:.3f}"
        )
    # local refinement (Nelder-Mead on 1-IoU), two rounds with shrinking simplex
    p = best[1]
    for scale in (1.0, 0.3):
        simplex = np.vstack([p, p + np.diag([0.05, 0.05, 0.15, 0.03]) * scale])
        res = minimize(
            lambda q: 1 - iou(q),
            p,
            method="Nelder-Mead",
            options={
                "xatol": 1e-4,
                "fatol": 1e-4,
                "maxiter": 400,
                "initial_simplex": simplex,
            },
        )
        p = res.x
    if z_fixed is not None:
        p = np.array([p[0], p[1], p[2], z_fixed])
    v = iou(p)
    if verbose:
        Tf = pose_from_params(p, n, d, ex0, ey0)
        print(
            f"  refined score {v:.3f} at x={p[0]:+.3f} y={p[1]:+.3f} yaw={np.degrees(p[2]):.1f} z={p[3]:.3f}"
            f"  landmark err {landmark_error(Tf):.1f} px  rail angle {rail_error(Tf):.1f} deg"
        )
    return pose_from_params(p, n, d, ex0, ey0), v, p


def fit_wrist_offsets(
    renderer: SilhouetteRenderer,
    target: np.ndarray,
    T_base_tool: np.ndarray,
    R_tool_cam: np.ndarray,
    up_t: np.ndarray,
    fwd_t: np.ndarray,
    lat_t: np.ndarray,
    d_t: float,
    d_c: float,
    init=(0.03, 0.0, 0.0),
    verbose=True,
):
    """Fit the wrist camera's forward/lateral offset and yaw from the gripper-finger silhouette.

    The table plane seen by the wrist camera fixes its pitch, roll and height above the table
    (``R_tool_cam``, ``d_c``). The remaining three degrees of freedom (offset along the gripper
    axis ``fwd_t``, sideways ``lat_t``, and rotation ``yaw`` about the table normal ``up_t``) are
    found by maximising IoU between the rendered robot silhouette and ``target`` (the black
    fingers in the wrist image). A fourth parameter ``dz`` shifts the camera along the table
    normal: the plane fixes the camera height above the *table*, but the table height in the
    tool frame (``d_t``) comes from the scene fit, whose base height is weakly constrained.
    The finger scale in the wrist image pins it; ``dz`` is the correction to apply to the
    scene fit's base height. Returns (T_tool_cam, iou, (forward, lateral, yaw, dz)).
    """
    from raiden.real2sim.geometry import inv_T

    n_t = up_t

    def pose(params):
        f, l, yaw, dz = params
        R = Rotation.from_rotvec(yaw * up_t).as_matrix() @ R_tool_cam
        c = f * fwd_t + l * lat_t
        c = c - up_t * (n_t @ c + d_t) + up_t * (d_c + dz)
        H = make_T(R, c)
        return H, inv_T(T_base_tool @ H)

    def iou(params):
        _, T_cam_base = pose(params)
        m = renderer.render_mask(T_cam_base)
        union = (m | target).sum()
        return (m & target).sum() / union if union else 0.0

    best = None
    for f in np.arange(0.0, 0.201, 0.02):
        for l in np.arange(-0.06, 0.061, 0.02):
            for yaw in np.radians(np.arange(-30, 31, 10)):
                for dz in np.arange(-0.04, 0.041, 0.02):
                    p = np.array([f, l, yaw, dz])
                    v = iou(p)
                    if best is None or v > best[0]:
                        best = (v, p)
    if verbose:
        print(
            f"  coarse best IoU {best[0]:.3f} at forward={best[1][0]:.3f} lateral={best[1][1]:+.3f} "
            f"yaw={np.degrees(best[1][2]):+.1f} dz={best[1][3]:+.3f}"
        )
    p = best[1]
    for scale in (1.0, 0.3):
        simplex = np.vstack([p, p + np.diag([0.02, 0.02, 0.15, 0.02]) * scale])
        res = minimize(
            lambda q: 1 - iou(q),
            p,
            method="Nelder-Mead",
            options={
                "xatol": 1e-4,
                "fatol": 1e-4,
                "maxiter": 400,
                "initial_simplex": simplex,
            },
        )
        p = res.x
    v = iou(p)
    if verbose:
        print(
            f"  refined IoU {v:.3f} at forward={p[0]:.3f} lateral={p[1]:+.3f} yaw={np.degrees(p[2]):+.1f} "
            f"dz={p[3]:+.3f} (base height correction)"
        )
    H, _ = pose(p)
    return H, float(v), p
