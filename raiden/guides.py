"""The MESA data-collector's grasp guides, projected into the calibrated camera views.

Robosuite's teleop window draws two sites on the gripper -- ``grip_site``, a sphere at the
grasp point, and ``grip_site_cylinder``, a long thin cylinder along the approach axis -- and
MESA's ``collect_data.py`` turns them on for every on-screen render.  They are what tells the
operator where the jaws will close before the jaws are anywhere near the object.

The raiden monitor shows the *recorded* camera streams rather than a free MuJoCo view, so the
sites are not available: rig cameras deliberately render with ``sitegroup = 0`` so that a
guide can never leak into a recorded frame.  Instead we draw the same two guides with OpenCV,
projected through each camera's own intrinsics and extrinsics.  That costs no extra render,
keeps the recorded frames untouched, and works against the real cameras too.

The grasp point and the approach direction come from FK of the i2rt ``grasp_site``, whose
local +z is the axis from ``tcp_site`` out through the jaws -- the same convention as
robosuite's cylinder.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# Matches the robosuite site colours: translucent green axis, red grasp point.
AXIS_COLOUR = (0, 255, 0)
SITE_COLOUR = (0, 0, 255)
AXIS_BEHIND = 0.45  # m of approach axis drawn back through the wrist
AXIS_AHEAD = 0.45  # m drawn ahead of the jaws
NEAR = 1e-3  # m; points closer than this to the image plane cannot be projected

RIG_JSON = Path("data/real2sim/calibration/rig.json")


class GraspGuides:
    """Draws the grasp guides for every camera it has geometry for.

    Cameras are registered with :meth:`add_camera`; a scene camera is fixed in the arm base
    frame, a wrist camera is fixed in the ``grasp_site`` frame and follows the arm.  The
    caller parks joint angles with :meth:`set_joints` (cheap, called from the robot loop) and
    the drawing thread calls :meth:`draw`, which runs FK at its own, much lower rate.
    """

    def __init__(self) -> None:
        self._cams: Dict[str, dict] = {}
        self._joints: Dict[str, np.ndarray] = {}
        self._kin = None

    # -- setup -------------------------------------------------------------
    def add_camera(
        self,
        name: str,
        K: np.ndarray,
        *,
        T_base_cam: Optional[np.ndarray] = None,
        T_tool_cam: Optional[np.ndarray] = None,
    ) -> None:
        """Register a camera. Give ``T_base_cam`` for a fixed camera, ``T_tool_cam`` for a
        wrist camera (both camera-to-frame, OpenCV optical convention)."""
        if (T_base_cam is None) == (T_tool_cam is None):
            raise ValueError(
                f"camera {name!r}: give exactly one of T_base_cam, T_tool_cam"
            )
        self._cams[name] = {
            "K": np.asarray(K, dtype=np.float64).reshape(3, 3),
            "T_base_cam": None
            if T_base_cam is None
            else np.asarray(T_base_cam, dtype=np.float64),
            "T_tool_cam": None
            if T_tool_cam is None
            else np.asarray(T_tool_cam, dtype=np.float64),
        }

    def __bool__(self) -> bool:
        return bool(self._cams)

    def has(self, name: str) -> bool:
        return name in self._cams

    # -- state -------------------------------------------------------------
    def set_joints(self, arm: str, q: np.ndarray) -> None:
        """Park the newest arm joint angles (first 6 entries used).  Never blocks."""
        self._joints[arm] = np.asarray(q, dtype=np.float64)[:6].copy()

    def _fk(self, q: np.ndarray) -> np.ndarray:
        if self._kin is None:
            from i2rt.robots.kinematics import Kinematics

            from raiden._xml_paths import get_yam_4310_linear_xml_path

            self._kin = Kinematics(get_yam_4310_linear_xml_path(), "grasp_site")
        nq = self._kin._configuration.model.nq
        q_full = np.zeros(nq, dtype=np.float64)
        q_full[: min(len(q), nq)] = q[:nq]
        return np.asarray(self._kin.fk(q_full), dtype=np.float64)

    # -- drawing -----------------------------------------------------------
    def draw(self, name: str, image_bgr: np.ndarray) -> np.ndarray:
        """Draw the guides for every known arm onto ``image_bgr`` (modified in place)."""
        cam = self._cams.get(name)
        if cam is None or not self._joints:
            return image_bgr
        for q in list(self._joints.values()):
            try:
                T_base_tool = self._fk(q)
            except Exception:  # a viewer must never take the recording down
                return image_bgr
            T_base_cam = cam["T_base_cam"]
            if T_base_cam is None:
                T_base_cam = T_base_tool @ cam["T_tool_cam"]
            T_cam_base = np.linalg.inv(T_base_cam)
            grasp = T_base_tool[:3, 3]
            approach = T_base_tool[:3, 2]  # grasp_site local +z: out through the jaws
            self._draw_one(
                image_bgr,
                cam["K"],
                T_cam_base,
                grasp - AXIS_BEHIND * approach,
                grasp + AXIS_AHEAD * approach,
                grasp,
            )
        return image_bgr

    @staticmethod
    def _draw_one(img, K, T_cam_base, a_base, b_base, grasp_base) -> None:
        a = T_cam_base[:3, :3] @ a_base + T_cam_base[:3, 3]
        b = T_cam_base[:3, :3] @ b_base + T_cam_base[:3, 3]
        g = T_cam_base[:3, :3] @ grasp_base + T_cam_base[:3, 3]
        seg = _clip_to_front(a, b)
        if seg is not None:
            p, q = (_project(K, v) for v in seg)
            cv2.line(img, p, q, AXIS_COLOUR, 2, cv2.LINE_AA)
        if g[2] > NEAR:
            centre = _project(K, g)
            cv2.circle(img, centre, 5, SITE_COLOUR, -1, cv2.LINE_AA)
            cv2.circle(img, centre, 7, (255, 255, 255), 1, cv2.LINE_AA)


def _project(K: np.ndarray, p_cam: np.ndarray) -> Tuple[int, int]:
    u = K @ p_cam
    return int(round(u[0] / u[2])), int(round(u[1] / u[2]))


def _clip_to_front(a: np.ndarray, b: np.ndarray):
    """Clip the segment a-b against the camera's near plane, or None if fully behind it."""
    if a[2] <= NEAR and b[2] <= NEAR:
        return None
    if a[2] > NEAR and b[2] > NEAR:
        return a, b
    if a[2] <= NEAR:
        a, b = b, a
    t = (a[2] - NEAR) / (a[2] - b[2])
    return a, a + t * (b - a)


# ----------------------------------------------------------------------------------------
def make_guides(cameras, *, sim: str = "") -> GraspGuides:
    """Build guides for ``cameras`` (raiden ``Camera`` objects), resolving the geometry.

    Against the sim the geometry is exactly the rig the twin was built from; against the real
    rig it comes from the same ``rig.json`` that real2sim_calibrate wrote, falling back to the
    hand-eye/extrinsic calibration file.  Cameras with no known pose are simply left out.
    """
    guides = GraspGuides()
    rig = _load_rig(sim)
    calib = _load_calibration()
    for camera in cameras:
        name = camera.name
        try:
            K, _dist, _size = camera.get_intrinsics()
        except Exception:
            continue
        entry = (rig.get("cameras") or {}).get(name, {})
        if "T_base_cam" in entry:
            guides.add_camera(
                name, K, T_base_cam=np.array(entry["T_base_cam"], dtype=np.float64)
            )
            continue
        if "T_tool_cam" in entry:
            guides.add_camera(
                name, K, T_tool_cam=_grasp_from_tcp(np.array(entry["T_tool_cam"]))
            )
            continue
        cam_calib = (calib.get("cameras") or {}).get(name, {})
        for key, kwarg in (
            ("extrinsics", "T_base_cam"),
            ("hand_eye_calibration", "T_tool_cam"),
        ):
            block = cam_calib.get(key)
            if block and block.get("success"):
                T = np.eye(4)
                T[:3, :3] = np.array(block["rotation_matrix"], dtype=np.float64)
                T[:3, 3] = np.array(
                    block["translation_vector"], dtype=np.float64
                ).reshape(3)
                guides.add_camera(name, K, **{kwarg: T})
                break
    return guides


def _load_rig(sim: str) -> dict:
    if sim:
        try:
            from raiden.sim import SimConnection

            conn = SimConnection(sim)
            try:
                return conn.call("get_rig")
            finally:
                conn.close()
        except Exception:
            return {}
    try:
        return json.loads(RIG_JSON.read_text())
    except Exception:
        return {}


def _load_calibration() -> dict:
    from raiden._config import CALIBRATION_FILE

    try:
        if os.path.exists(CALIBRATION_FILE):
            return json.loads(Path(CALIBRATION_FILE).read_text())
    except Exception:
        pass
    return {}


def _grasp_from_tcp(T_tcp_cam: np.ndarray) -> np.ndarray:
    """rig.json stores the wrist camera in the i2rt ``tcp_site`` frame; FK gives ``grasp_site``."""
    from raiden.sim.calibration import _grasp_from_tcp as convert

    return convert(T_tcp_cam)
