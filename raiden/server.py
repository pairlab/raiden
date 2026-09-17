"""Policy server for the YAM robot — serves live observations via chiral.

Bridges the raiden robot stack (ZED/RealSense cameras + YAM follower arms) to
any remote policy client that speaks the chiral WebSocket protocol.

Usage::

    rd serve

Or with custom options::

    rd serve --port 8765 --no-depth

Thread layout::

    camera-<name>       : grabs frames from ZED or RealSense at ~30 Hz (per camera)
    proprio-<name>      : reads joint state from follower arms at ~100 Hz (per stream)
    command             : sends the interpolated joint command to the followers at 100 Hz
    asyncio event loop  : handles WebSocket connections (chiral protocol)

Extrinsics convention
---------------------
All extrinsics are expressed in the left-arm base frame (matching the dataset
convention used by the converter).

*Wrist cameras* have moving extrinsics that are recomputed on every
``_make_obs()`` call::

    T_left_base→cam  =  [T_left_base_from_right_base @]  FK(q[:6])  @  T_cam→ee

*Scene camera* extrinsics are static and loaded once from the calibration file.

Image orientation
-----------------
Cameras in ``_FLIP_CAMERAS`` (e.g. ``right_wrist_camera``) are physically
mounted upside-down.  The capture loop rotates their images by 180° to match
the right-side-up orientation used by the training dataset.  The principal
point in the intrinsics and the ``T_cam→ee`` rotation are corrected accordingly.
"""

import asyncio
import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional, Tuple

import chiral
import cv2
import numpy as np
from chiral.types import CameraInfo, Observation

from raiden.camera_config import CameraConfig as RaidenCameraConfig
from raiden.robot.controller import (
    _GRIPPER_MIN_OPENING,
    RobotController,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Cameras mounted upside-down: images are rotated 180° and extrinsics/
# intrinsics are corrected to match the right-side-up frame.
_FLIP_CAMERAS = {"right_wrist_camera"}

# Maps wrist camera name → which follower arm drives its extrinsics.
_WRIST_CAMERA_ARM: dict[str, str] = {
    "left_wrist_camera": "left",
    "right_wrist_camera": "right",
}

# 4×4 homogeneous 180° rotation around the Z (optical) axis.
# Right-multiplying T_cam→ee by this converts from the physical (upside-down)
# camera frame used during calibration to the standard (right-side-up) frame
# that matches the rotated images served by the policy server.
_R_FLIP_180 = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

DOF = 7  # joints per arm (6 revolute + 1 gripper)

# EE pose action layout — matches vla_foundry action_fields concatenation order:
#   [l_xyz(3), r_xyz(3), l_rot6d(6), r_rot6d(6), l_grip(1), r_grip(1)]  (20-D)
#   [l_xyz(3), l_rot6d(6), l_grip(1)]                         (10-D, single arm)
EE_POSE_ARM_DOF = 10  # per arm: xyz(3) + rot_6d(6) + grip(1)

# Robot name used in proprioception key names — must match the shardify convention.
_ROBOT = "yam"

# Number of proprio samples kept in the interpolation ring buffer (~0.64 s at 100 Hz).
_PROPRIO_HISTORY_SIZE = 64

# Default maximum allowed joint position delta per policy step.
# At a 20 Hz control rate, 0.2 rad/step ≈ 4 rad/s — large enough to catch
# abrupt policy jumps while allowing normal motion.
_DEFAULT_MAX_JOINT_DELTA = 0.2  # radians

# Rate the client sends actions at; each action is reached one period after it arrives.
_DEFAULT_CONTROL_HZ = 10.0

# Rate the interpolated command is sent to the followers — the rate of the teleop
# loops that recorded the training commands.
_COMMAND_HZ = 100.0

# Per-arm Kinematics instances, each protected by its own lock.
#
# Kinematics.fk() and Kinematics.ik() both call
# self._configuration.update(q) which modifies the mink.Configuration
# in-place, and the IK solver may carry internal warm-start state between
# calls.  Using one instance per arm ensures that the left-arm IK solution
# never contaminates the right-arm IK starting state (and vice versa).
# Each lock serializes concurrent FK/IK calls on that arm.
_kin_l: Any = None
_kin_r: Any = None
_kin_l_lock: threading.Lock = threading.Lock()
_kin_r_lock: threading.Lock = threading.Lock()


def _make_kinematics() -> Any:
    from i2rt.robots.kinematics import Kinematics

    from raiden._xml_paths import get_yam_4310_linear_xml_path

    return Kinematics(get_yam_4310_linear_xml_path(), "grasp_site")


def _get_kin_l() -> Any:
    """Return the left-arm Kinematics instance (caller must hold _kin_l_lock)."""
    global _kin_l
    if _kin_l is None:
        _kin_l = _make_kinematics()
    return _kin_l


def _get_kin_r() -> Any:
    """Return the right-arm Kinematics instance (caller must hold _kin_r_lock)."""
    global _kin_r
    if _kin_r is None:
        _kin_r = _make_kinematics()
    return _kin_r


def _rot6d_to_mat(v: np.ndarray) -> np.ndarray:
    """Convert a 6D rotation representation to a 3×3 rotation matrix.

    Uses the Gram-Schmidt orthonormalisation of the first two rows, matching
    vla_foundry's ``rot_6d_to_matrix`` convention.

    Args:
        v: (6,) array — first two rows of a rotation matrix: [R[0,:], R[1,:]].

    Returns:
        (3, 3) float64 rotation matrix.
    """
    a1, a2 = v[:3].astype(np.float64), v[3:6].astype(np.float64)
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=0)  # rows


def _fk_padded(kin: Any, q6: np.ndarray) -> np.ndarray:
    """Call FK, padding q6 with zeros to the model's nq (combined XML has nq=8)."""
    nq = kin._configuration.model.nq
    q = np.zeros(nq, dtype=np.float64)
    q[: len(q6)] = q6
    return kin.fk(q)


def _fk_to_xyz_rot6d(kin: Any, q6: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run FK and return ``(xyz, rot_6d)`` for the end-effector site.

    Args:
        kin: Kinematics instance.
        q6: (6,) arm joint angles (no gripper).

    Returns:
        xyz: (3,) float32 position.
        rot_6d: (6,) float32 — first two rows of the rotation matrix.
    """
    T = _fk_padded(kin, q6)
    xyz = T[:3, 3].astype(np.float32)
    rot_6d = (
        T[:2, :3].flatten().astype(np.float32)
    )  # row0 + row1 → [R00,R01,R02,R10,R11,R12]
    return xyz, rot_6d


def _pose_from_xyz_rot6d(xyz: np.ndarray, rot6d: np.ndarray) -> np.ndarray:
    """Build a 4×4 TCP pose from a position and 6D rotation vector."""
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = xyz.astype(np.float64)
    T[:3, :3] = _rot6d_to_mat(rot6d)
    return T


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class RaidenPolicyServer(chiral.PolicyServer):
    """Policy server for the YAM robot system (left arm only or bimanual).

    Streams live camera images, depth maps, and joint-state observations to a
    remote policy over WebSocket using the chiral protocol.

    Args:
        camera_config_file: Path to ``camera.json``.
        calibration_file: Path to ``calibration_results.json``.  Camera
            intrinsics are read from the SDK at the actual serving resolution;
            the calibration file supplies extrinsics and ``T_cam→ee``.
        host: WebSocket host to bind to.
        port: WebSocket port to listen on.
        arms: ``"single"`` drives the left follower only; ``"bimanual"`` both.
        control_hz: Rate the client sends actions at.  Each action is
            interpolated from the current command over one period.
    """

    def __init__(
        self,
        camera_config_file: str = "./config/camera.json",
        calibration_file: str = "./config/calibration_results.json",
        host: str = "0.0.0.0",
        port: int = 8765,
        max_joint_delta: float = _DEFAULT_MAX_JOINT_DELTA,
        action_type: str = "ee_pose",
        no_depth: bool = False,
        resize_images_size: Optional[Tuple[int, int]] = None,
        visualize: bool = False,
        arms: str = "bimanual",
        control_hz: float = _DEFAULT_CONTROL_HZ,
    ):
        self._no_depth = no_depth
        self._resize = resize_images_size  # (H, W) or None
        if action_type not in ("joint", "ee_pose"):
            raise ValueError(
                f"action_type must be 'joint' or 'ee_pose', got {action_type!r}"
            )
        if arms not in ("single", "bimanual"):
            raise ValueError(f"arms must be 'single' or 'bimanual', got {arms!r}")
        self._action_type = action_type
        # Action order: left arm first, then right arm.
        self._arms = ("left",) if arms == "single" else ("left", "right")
        self._control_hz = control_hz
        self._raiden_cam_cfg = RaidenCameraConfig(camera_config_file)
        self._calibration = self._load_calibration(calibration_file)

        # Per-camera data populated across _open_cameras() and
        # _prepare_camera_transforms():
        #   _cam_handles     : hardware handles keyed by camera name
        #   _cam_intrinsics  : 3×3 float64 camera matrix (from SDK, flip-corrected)
        #   _cam_extrinsics  : 4×4 float64 static extrinsics (scene cameras only)
        #   _T_cam2ee        : 4×4 float64 T_cam→ee (wrist cameras, flip-corrected)
        self._cam_handles: dict = {}
        self._cam_intrinsics: dict[str, np.ndarray] = {}
        self._cam_extrinsics: dict[str, np.ndarray] = {}
        self._T_cam2ee: dict[str, np.ndarray] = {}

        # T_left_base_from_right_base = inv(T_right_base_to_left_base)
        # Brings right-arm FK results into the left-arm base frame.
        self._T_left_base_from_right_base: Optional[np.ndarray] = None

        # Open cameras first so we can query actual H, W, and SDK intrinsics
        # before super().__init__() calls camera_configs().
        print("\nOpening cameras...")
        self._open_cameras()
        self._prepare_camera_transforms()

        # Per-camera capture timestamps (Unix ns) updated by capture loops.
        # ZED cameras use sl.TIME_REFERENCE.IMAGE (hardware capture time).
        # RealSense uses time.time_ns() recorded immediately after wait_for_frames().
        # Used as the reference for proprio interpolation, matching converter's
        # timestamps.npy.  Initialised before threads start.
        self._cam_capture_ts_ns: dict[str, int] = {
            name: 0 for name in self._cam_handles
        }
        self._cam_ts_locks: dict[str, threading.Lock] = {
            name: threading.Lock() for name in self._cam_handles
        }
        self._cam_phase_offset_ns: dict[str, int] = {
            name: 0 for name in self._cam_handles
        }

        # Initialize follower robots only (leaders not needed for inference).
        self._robot = RobotController(
            use_right_leader=False,
            use_left_leader=False,
            use_right_follower="right" in self._arms,
            use_left_follower=True,
        )
        self._robot.initialize_robots()
        self._followers = {
            "left": self._robot.follower_l,
            "right": self._robot.follower_r,
        }

        # super().__init__() calls camera_configs() and proprio_configs() to
        # pre-allocate self.images, self.depths, and self.proprios.
        super().__init__(host=host, port=port)

        # Proprio interpolation ring buffers — keyed by proprio name, each holds
        # (Unix ns via ZED SDK clock, value) pairs.  Initialised after super().__init__()
        # since proprio names come from proprio_configs().
        self._proprio_history: dict[str, deque] = {
            name: deque(maxlen=_PROPRIO_HISTORY_SIZE) for name in self.proprios
        }
        self._proprio_history_locks: dict[str, threading.Lock] = {
            name: threading.Lock() for name in self.proprios
        }

        self._max_joint_delta = max_joint_delta
        self._step_count = 0
        self._t_sum = 0.0
        self._running = True
        # The command thread sends start + alpha * (target - start), with alpha
        # rising from 0 to 1 over one control period after _cmd_t0.  None until
        # the first action after startup or reset.  The target is also the IK seed.
        self._cmd_lock = threading.Lock()
        self._cmd_start: Optional[np.ndarray] = None
        self._cmd_target: Optional[np.ndarray] = None
        self._cmd_t0 = 0.0
        # Set when an emergency stop fires; causes apply_action() to reject new
        # actions and the command thread to stop commanding immediately.
        self._estop_active = threading.Event()
        threading.Thread(target=self._command_loop, daemon=True).start()

        if visualize:
            threading.Thread(target=self._rerun_loop, daemon=True).start()

        for name in self._raiden_cam_cfg.list_camera_names():
            threading.Thread(
                target=self._camera_loop, args=(name,), daemon=True
            ).start()

        threading.Thread(target=self._proprio_loop, daemon=True).start()

        # Wait for the first frame from every camera, then compute phase offsets.
        print("\nComputing camera phase offsets...")
        self._compute_camera_offsets()

        # Attach footpedal hard e-stop (optional — warns and continues if absent).
        # Use a custom callback so _estop_active is set *before* emergency_stop()
        # runs, guaranteeing that any concurrent step() call sees the flag and
        # returns immediately instead of racing with the hold loop.
        print("Initializing footpedal...")
        self._robot.attach_footpedal(callback=self._trigger_estop)

        print("\nRaiden policy server ready.")

    # -------------------------------------------------------------------------
    # Calibration helpers
    # -------------------------------------------------------------------------

    def _load_calibration(self, path: str) -> dict:
        p = Path(path)
        if p.exists():
            with open(p) as f:
                return json.load(f)
        print(
            f"Note: calibration file not found at '{path}'; "
            "using default intrinsics / extrinsics."
        )
        return {}

    def _prepare_camera_transforms(self) -> None:
        """Load extrinsics and T_cam→ee from the calibration file.

        Intrinsics come from the SDK (already stored in _cam_handles by
        _open_cameras()); the calibration file is only used for extrinsics.
        """
        # The calibration file nests per-camera data under a top-level "cameras" key.
        calib_cameras: dict = self._calibration.get("cameras", {})

        # "right_base_to_left_base" maps left_arm_base → right_arm_base (despite the name).
        # Invert to bring right_arm_base points into left_arm_base.
        bimanual = self._calibration.get("bimanual_transform", {})
        mat = bimanual.get("right_base_to_left_base")
        if mat is not None:
            self._T_left_base_from_right_base = np.linalg.inv(
                np.array(mat, dtype=np.float64)
            )
        else:
            print(
                "Note: bimanual_transform not found in calibration; "
                "right wrist extrinsics will be in right-arm base frame."
            )

        for name in self._raiden_cam_cfg.list_camera_names():
            handle = self._cam_handles.get(name)
            flip = name in _FLIP_CAMERAS

            # ── intrinsics (from SDK, already at serving resolution) ──────
            if handle is not None:
                K = handle["intrinsics"].copy()
                if flip:
                    # Reflect the principal point for the 180°-rotated image.
                    w, h = handle["w"], handle["h"]
                    K[0, 2] = (w - 1) - K[0, 2]
                    K[1, 2] = (h - 1) - K[1, 2]
                if self._resize is not None:
                    h_out, w_out = self._resize
                    h_src, w_src = handle["h"], handle["w"]
                    K[0, 0] *= w_out / w_src  # fx
                    K[0, 2] *= w_out / w_src  # cx
                    K[1, 1] *= h_out / h_src  # fy
                    K[1, 2] *= h_out / h_src  # cy
                self._cam_intrinsics[name] = K
            else:
                # Camera failed to open — use a placeholder.
                self._cam_intrinsics[name] = np.eye(3, dtype=np.float64)

            # ── extrinsics ────────────────────────────────────────────────
            cam_data: dict = calib_cameras.get(name, {})

            if name in _WRIST_CAMERA_ARM:
                # Wrist cameras: load T_cam→ee; extrinsics are dynamic (FK).
                he: dict = cam_data.get("hand_eye_calibration", {})
                if he.get("success") and "rotation_matrix" in he:
                    T = np.eye(4, dtype=np.float64)
                    T[:3, :3] = np.array(he["rotation_matrix"])
                    T[:3, 3] = np.array(he["translation_vector"]).flatten()
                    if flip:
                        # Calibration was done with raw (upside-down) images.
                        # Right-multiplying by _R_FLIP_180 converts T_cam→ee
                        # to the right-side-up camera frame used for serving.
                        T = T @ _R_FLIP_180
                    self._T_cam2ee[name] = T
                else:
                    print(
                        f"Note: hand-eye calibration missing for '{name}'; "
                        "wrist extrinsics will be identity."
                    )
                # Placeholder; overwritten per-step in _make_obs().
                self._cam_extrinsics[name] = np.eye(4, dtype=np.float64)
            else:
                # Scene camera: static extrinsics.
                ext: dict = cam_data.get("extrinsics", {})
                if ext.get("success") and "rotation_matrix" in ext:
                    frame = ext.get("reference_frame", "left_arm_base")
                    if frame != "left_arm_base":
                        raise RuntimeError(
                            f"{name}: calibrated extrinsics are in the {frame!r} "
                            "frame; the policy needs the left arm base frame. "
                            "Refit with scripts/write_calib.py or "
                            "scripts/write_scene_calib.py."
                        )
                    T = np.eye(4, dtype=np.float64)
                    T[:3, :3] = np.array(ext["rotation_matrix"])
                    T[:3, 3] = np.array(ext["translation_vector"]).flatten()
                    self._cam_extrinsics[name] = T
                else:
                    self._cam_extrinsics[name] = np.eye(4, dtype=np.float64)

    # -------------------------------------------------------------------------
    # chiral.PolicyServer interface
    # -------------------------------------------------------------------------

    def camera_configs(self) -> list[chiral.CameraConfig]:
        configs = []
        for name in self._raiden_cam_cfg.list_camera_names():
            handle = self._cam_handles.get(name, {})
            if self._resize is not None:
                h, w = self._resize
            else:
                h = handle.get("h", 0)
                w = handle.get("w", 0)
            configs.append(
                chiral.CameraConfig(
                    name=name,
                    height=h,
                    width=w,
                    channels=3,
                    has_depth=handle.get("depth", False),
                    intrinsics=self._cam_intrinsics[name],
                    extrinsics=self._cam_extrinsics[name],
                )
            )
        return configs

    def proprio_configs(self) -> list[chiral.ProprioConfig]:
        streams = []
        if self._robot.follower_r:
            streams.append(chiral.ProprioConfig(name="follower_r_joint_pos", size=DOF))
            streams.append(
                chiral.ProprioConfig(
                    name=f"robot__actual__poses__right::{_ROBOT}__xyz", size=3
                )
            )
            streams.append(
                chiral.ProprioConfig(
                    name=f"robot__actual__poses__right::{_ROBOT}__rot_6d", size=6
                )
            )
            streams.append(
                chiral.ProprioConfig(
                    name=f"robot__actual__grippers__right::{_ROBOT}_hand", size=1
                )
            )
        if self._robot.follower_l:
            streams.append(chiral.ProprioConfig(name="follower_l_joint_pos", size=DOF))
            streams.append(
                chiral.ProprioConfig(
                    name=f"robot__actual__poses__left::{_ROBOT}__xyz", size=3
                )
            )
            streams.append(
                chiral.ProprioConfig(
                    name=f"robot__actual__poses__left::{_ROBOT}__rot_6d", size=6
                )
            )
            streams.append(
                chiral.ProprioConfig(
                    name=f"robot__actual__grippers__left::{_ROBOT}_hand", size=1
                )
            )
        return streams

    def _trigger_estop(self) -> None:
        """Set the server-level e-stop flag then invoke the robot emergency stop.

        Called from the footpedal callback thread.  Setting _estop_active first
        ensures that any concurrent step() call in the asyncio event loop
        sees the flag and returns immediately rather than racing with the
        emergency_stop() hold loop.
        """
        self._estop_active.set()
        self._robot.emergency_stop()

    async def _handle(self, websocket) -> None:
        try:
            await super()._handle(websocket)
        finally:
            print(
                "[RaidenPolicyServer] Client disconnected — triggering emergency stop."
            )
            self._estop_active.set()
            self._robot.emergency_stop()

    def _action_dim(self) -> int:
        per_arm = EE_POSE_ARM_DOF if self._action_type == "ee_pose" else DOF
        return per_arm * len(self._arms)

    async def get_metadata(self) -> dict:
        if self._action_type == "ee_pose":
            layout = "+".join(
                [f"{arm}_xyz(3)" for arm in self._arms]
                + [f"{arm}_rot6d(6)" for arm in self._arms]
                + [f"{arm}_grip(1)" for arm in self._arms]
            )
        else:
            layout = "+".join(f"{arm}_joints(7)" for arm in self._arms)
        return {
            "cameras": self._raiden_cam_cfg.list_camera_names(),
            "action_type": self._action_type,
            "action_shape": [self._action_dim()],
            "action_layout": layout,
            "proprio_names": list(self.proprios.keys()),
        }

    async def reset(self) -> tuple[Observation, dict]:
        # Stop the command thread before homing so it cannot fight the move.
        with self._cmd_lock:
            self._cmd_start = None
            self._cmd_target = None
        self._step_count = 0
        self._t_sum = 0.0
        self._robot.move_to_home_positions(simultaneous=True)
        loop = asyncio.get_running_loop()
        obs = await loop.run_in_executor(None, self._make_obs)
        return obs, {}

    async def apply_action(self, action: np.ndarray) -> None:
        """Fire-and-forget action application (chiral non-blocking API).

        Called by the network thread at the client's action rate.  Retargets
        the command thread and returns immediately — the client never waits
        for execution to finish, and ``get_obs()`` is polled independently.
        """
        if self._estop_active.is_set():
            return

        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()

        action = np.asarray(action).reshape(-1)  # (D,)
        if action.size != self._action_dim():
            raise ValueError(
                f"expected a {self._action_dim()}-D {self._action_type} action "
                f"for arms={'+'.join(self._arms)}, got {action.size}-D"
            )

        # Run IK in the executor so it doesn't block the event loop.
        # Seeded from the last commanded target for stable convergence.
        if self._action_type == "ee_pose":
            joint_cmd = await loop.run_in_executor(
                None, self._ee_pose_to_joint_cmd, action, self._cmd_target
            )
        else:
            joint_cmd = action.astype(np.float64)

        # Safety check: abort if any joint delta exceeds the threshold.
        self._check_joint_delta(joint_cmd)

        # Start from the command currently being sent, so an action that arrives
        # before the previous one is reached replaces it instead of queueing.
        now = time.monotonic()
        with self._cmd_lock:
            if self._cmd_target is None:
                start = np.concatenate(
                    [self._followers[arm].get_joint_pos() for arm in self._arms]
                )
            else:
                start = self._interpolated_cmd(now)
            self._cmd_start = start.astype(np.float64)
            self._cmd_target = np.asarray(joint_cmd, dtype=np.float64)
            self._cmd_t0 = now

        step_ms = (time.perf_counter() - t0) * 1e3
        self._step_count += 1
        self._t_sum += step_ms

        if self._step_count % 10 == 0:
            print(
                f"step={self._step_count:4d}  "
                f"ik_ms={step_ms:5.2f}  "
                f"avg={self._t_sum / self._step_count:5.2f}ms"
            )

    async def get_obs(self) -> Observation:
        """Return the current observation without blocking the event loop.

        Called independently by the client at its obs-polling rate (e.g. 30 Hz),
        fully decoupled from action application.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._make_obs)

    # -------------------------------------------------------------------------
    # Camera temporal alignment
    # -------------------------------------------------------------------------

    def _get_zed_current_time_ns(self) -> int:
        """Return the current time in nanoseconds using the ZED SDK hardware clock.

        Queries the reference camera's ZED SDK clock (sl.TIME_REFERENCE.CURRENT),
        which is Unix nanoseconds — the same clock used by the recorder to timestamp
        proprioception samples.  Falls back to ``time.time_ns()`` if no ZED camera
        is available.
        """
        ref = self._get_reference_camera()
        handle = self._cam_handles.get(ref)
        if handle is not None and "get_current_ts_ns" in handle:
            return handle["get_current_ts_ns"]()
        return time.time_ns()

    def _get_reference_camera(self) -> str:
        """Return the reference camera name for temporal alignment.
        Prefers 'scene_camera'; falls back to the first camera in the config.
        """
        names = list(self._cam_handles.keys())
        if "scene_camera" in names:
            return "scene_camera"
        return names[0] if names else ""

    def _wait_for_first_frames(self, timeout_s: float = 5.0) -> None:
        """Block until every camera loop has delivered at least one frame."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if all(self._cam_capture_ts_ns.get(n, 0) > 0 for n in self._cam_handles):
                return
            time.sleep(0.05)
        missing = [
            n for n in self._cam_handles if self._cam_capture_ts_ns.get(n, 0) == 0
        ]
        print(
            f"  Warning: cameras did not produce a first frame within {timeout_s:.0f}s: {missing}"
        )

    def _compute_camera_offsets(self) -> None:
        """Compute per-camera phase offsets relative to the reference camera.

        Captures frame timestamps from the first frame of each camera.  Since FPS
        is assumed consistent, this one-time offset corrects the fixed phase
        difference between capture loops so that::

            arrival_ts_ns + phase_offset_ns  →  unified time axis
        """
        self._wait_for_first_frames()
        names = list(self._cam_handles.keys())
        if not names:
            return

        arrival_ns: dict[str, int] = {}
        for name in names:
            with self._cam_ts_locks[name]:
                arrival_ns[name] = self._cam_capture_ts_ns[name]

        ref_name = self._get_reference_camera()
        ref_ts = arrival_ns.get(ref_name, 0)

        self._cam_phase_offset_ns = {
            name: ref_ts - arrival_ns.get(name, ref_ts) for name in names
        }

        print(f"  Reference camera: '{ref_name}'")
        for name in names:
            offset_ms = self._cam_phase_offset_ns[name] / 1e6
            print(f"    {name}: {offset_ms:+.1f} ms")

    # -------------------------------------------------------------------------
    # Proprio interpolation
    # -------------------------------------------------------------------------

    def _interpolate_proprio(
        self, name: str, target_ts_ns: int
    ) -> Optional[np.ndarray]:
        """Interpolate the proprio ring buffer to *target_ts_ns* (Unix ns, ZED clock).

        Falls back to the latest buffered value when the history is too short
        (e.g. at startup).  Returns None if no data is available at all.
        """
        if name not in self._proprio_history:
            return self._read_proprio(name)

        with self._proprio_history_locks[name]:
            history = list(self._proprio_history[name])

        if len(history) < 2:
            return self._read_proprio(name)

        ts_arr = np.array([h[0] for h in history], dtype=np.float64)
        val_arr = np.stack([h[1] for h in history], axis=0)

        t = float(target_ts_ns)
        # np.interp clamps at boundaries, so out-of-range values return the
        # nearest endpoint rather than extrapolating.
        return np.stack(
            [np.interp(t, ts_arr, val_arr[:, d]) for d in range(val_arr.shape[1])],
            axis=0,
        ).astype(np.float32)

    # -------------------------------------------------------------------------
    # EE pose → joint IK
    # -------------------------------------------------------------------------

    def _ee_pose_to_joint_cmd(
        self, action: np.ndarray, init_cmd: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Convert an EE pose action to a joint command via IK.

        Action layout (matching vla_foundry action_fields concatenation order)::

            [l_xyz(3), r_xyz(3), l_rot6d(6), r_rot6d(6), l_grip(1), r_grip(1)]
            [l_xyz(3), l_rot6d(6), l_grip(1)]                  (single arm)

        Poses are in each arm's own base frame (left arm in left-base frame,
        right arm in right-base frame) — same convention as the stored action
        in ``lowdim.npz``.

        Returns:
            (7 per arm,) float32 joint command — left arm first, then right arm,
            matching the joint action convention expected by ``apply_action()``.
        """
        n = len(self._arms)
        cmds = []
        for i, arm in enumerate(self._arms):
            T = _pose_from_xyz_rot6d(
                action[3 * i : 3 * i + 3], action[3 * n + 6 * i : 3 * n + 6 * i + 6]
            )
            grip = float(action[9 * n + i])
            kin_lock, get_kin = (
                (_kin_l_lock, _get_kin_l)
                if arm == "left"
                else (_kin_r_lock, _get_kin_r)
            )
            with kin_lock:
                kin = get_kin()
                # Seed IK from the last *commanded* joint positions so the solver
                # starts from the target the robot is moving toward rather than the
                # mid-motion measured position, which is less stable and may slow
                # convergence.  On the very first step (init_cmd is None), seed with
                # zeros so IK starts from the neutral configuration rather than
                # whatever configuration was left by the most recent FK call.
                q_init = np.zeros(kin._configuration.model.nq, dtype=np.float64)
                if init_cmd is not None:
                    q_init[:6] = init_cmd[DOF * i : DOF * i + 6]
                ok, q_full = kin.ik(T, "grasp_site", init_q=q_init)
                T_fk = kin.fk(q_full)

            # Diagnostic: log IK convergence and round-trip FK error.
            pos_err = float(np.linalg.norm(T_fk[:3, 3] - T[:3, 3]))
            print(
                f"[IK-{arm[0].upper()}] ok={ok}  target_xyz={T[:3, 3].round(4)}  "
                f"fk_xyz={T_fk[:3, 3].round(4)}  pos_err={pos_err * 1000:.2f}mm"
            )
            # IK returns full nq — take arm joints only, then append gripper.
            cmds.append(np.append(q_full[:6], grip).astype(np.float32))
        return np.concatenate(cmds)

    # -------------------------------------------------------------------------
    # Joint command stream
    # -------------------------------------------------------------------------

    def _interpolated_cmd(self, now: float) -> np.ndarray:
        """Return the command due at *now* (caller must hold _cmd_lock)."""
        alpha = min((now - self._cmd_t0) * self._control_hz, 1.0)
        return (1 - alpha) * self._cmd_start + alpha * self._cmd_target

    def _command_loop(self) -> None:
        """Send the interpolated joint command to the followers at _COMMAND_HZ.

        Each action is reached one control period after it arrives and then held
        until the next one, in 100 Hz sub-steps like the teleop loops that
        recorded the training data.  The command is interpolated between
        *commanded* targets (not actual positions) to reproduce the same
        trajectory shape as the training data.
        """
        dt = 1.0 / _COMMAND_HZ
        while self._running and not self._estop_active.is_set():
            loop_start = time.monotonic()
            try:
                with self._cmd_lock:
                    if self._cmd_target is not None:
                        cmd = self._interpolated_cmd(loop_start)
                        for i, arm in enumerate(self._arms):
                            self._command_arm(
                                self._followers[arm], cmd[DOF * i : DOF * (i + 1)]
                            )
            except Exception as e:
                print(f"Command error: {e}")
            remaining = dt - (time.monotonic() - loop_start)
            if remaining > 0:
                time.sleep(remaining)

    @staticmethod
    def _command_arm(follower: Any, cmd: np.ndarray) -> None:
        """Command one follower with the gripper limits the teleop loops apply.

        The gripper never closes past the minimum opening; squeeze force on a
        blocked grasp is bounded by i2rt's force limiter, motor-side.
        """
        cmd = cmd.copy()
        cmd[6] = np.clip(cmd[6], _GRIPPER_MIN_OPENING, 1.0)
        follower.command_joint_pos(cmd)

    # -------------------------------------------------------------------------
    # Joint-delta safety check
    # -------------------------------------------------------------------------

    def _check_joint_delta(self, action: np.ndarray) -> None:
        """Emergency-stop if the commanded action jumps too far from current positions.

        Compares each arm's commanded joint positions against the latest buffered
        proprioception values.  If any joint delta exceeds ``_max_joint_delta``
        (radians), prints an error and triggers the emergency stop — the command
        is never sent to the robot.

        Args:
            action: Flat array of 7 per arm (left arm then right arm).
        """
        pairs = []
        for i, arm in enumerate(self._arms):
            current = self._read_proprio(f"follower_{arm[0]}_joint_pos")
            if current is not None:
                pairs.append((arm, current, action[DOF * i : DOF * (i + 1)]))

        for arm, current, commanded in pairs:
            # Only check the 6 arm joints — gripper (index 6) uses linear position
            # units (not radians) and has a wide operating range; applying the
            # radian-based limit to it would falsely trigger on any large gripper step.
            delta = np.abs(commanded[:6] - current[:6])
            max_delta = float(delta.max())
            if max_delta > self._max_joint_delta:
                joint_idx = int(delta.argmax())
                print(
                    f"\n[SAFETY] Dangerously large joint delta on {arm} arm — "
                    f"joint {joint_idx}: {max_delta:.4f} rad "
                    f"(limit={self._max_joint_delta:.4f} rad). "
                    "Triggering emergency stop."
                )
                self._estop_active.set()
                self._robot.emergency_stop()

    # -------------------------------------------------------------------------
    # Observation construction (overrides base class to add dynamic extrinsics)
    # -------------------------------------------------------------------------

    def _make_obs(self) -> Observation:
        """Snapshot all buffers and compute per-step wrist camera extrinsics.

        All joint states are interpolated to the reference camera's latest frame
        arrival time so the observation is temporally coherent across cameras and
        the proprioception stream.
        """
        # Reference timestamp: the reference camera's arrival time shifted onto
        # the unified time axis (arrival_ts + phase_offset).
        ref_cam = self._get_reference_camera()
        if ref_cam and self._cam_capture_ts_ns.get(ref_cam, 0) > 0:
            with self._cam_ts_locks[ref_cam]:
                ref_ts_ns = self._cam_capture_ts_ns[
                    ref_cam
                ] + self._cam_phase_offset_ns.get(ref_cam, 0)
        else:
            ref_ts_ns = self._get_zed_current_time_ns()

        # Interpolate joint positions to the reference camera timestamp — matches
        # the converter which uses the first camera's timestamps.npy as ref_ts
        # for all proprio interpolation (joints and FK).
        q_r = self._interpolate_proprio("follower_r_joint_pos", ref_ts_ns)
        q_l = self._interpolate_proprio("follower_l_joint_pos", ref_ts_ns)

        cameras: list[CameraInfo] = []
        for c in self._configs:
            # Compute FK-based extrinsics outside the lock (CPU work, no shared state).
            extrinsics = self._compute_extrinsics(c.name, q_r, q_l)

            # Atomically update extrinsics and snapshot image/depth/intrinsics under
            # the same per-camera lock — mirrors chiral.PolicyServer._make_obs().
            with self._locks[c.name]:
                np.copyto(self.extrinsics[c.name], extrinsics)
                image = self.images[c.name].copy()
                depth = self.depths[c.name].copy() if c.name in self.depths else None
                intrinsics = self.intrinsics[c.name].copy()
                cam_ts = self._cam_capture_ts_ns[c.name]

            cameras.append(
                CameraInfo(
                    name=c.name,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    image=image,
                    depth=depth,
                    timestamp=cam_ts,
                )
            )

        # EE pose proprio names are derived from FK — skip ring-buffer lookup for them.
        _ee_pose_keys = {
            f"robot__actual__poses__left::{_ROBOT}__xyz",
            f"robot__actual__poses__left::{_ROBOT}__rot_6d",
            f"robot__actual__grippers__left::{_ROBOT}_hand",
            f"robot__actual__poses__right::{_ROBOT}__xyz",
            f"robot__actual__poses__right::{_ROBOT}__rot_6d",
            f"robot__actual__grippers__right::{_ROBOT}_hand",
        }

        proprios: dict[str, np.ndarray] = {}
        for p in self._proprio_configs:
            if p.name in _ee_pose_keys:
                continue  # filled below via FK
            val = self._interpolate_proprio(p.name, ref_ts_ns)
            if val is not None:
                proprios[p.name] = val
            else:
                with self._proprio_locks[p.name]:
                    proprios[p.name] = self.proprios[p.name].copy()

        # Compute actual EE poses via FK and inject as proprio.
        if q_l is not None:
            with _kin_l_lock:
                xyz_l, rot6d_l = _fk_to_xyz_rot6d(_get_kin_l(), q_l[:6])
            proprios[f"robot__actual__poses__left::{_ROBOT}__xyz"] = xyz_l
            proprios[f"robot__actual__poses__left::{_ROBOT}__rot_6d"] = rot6d_l
            proprios[f"robot__actual__grippers__left::{_ROBOT}_hand"] = q_l[6:7].astype(
                np.float32
            )
        if q_r is not None:
            with _kin_r_lock:
                xyz_r, rot6d_r = _fk_to_xyz_rot6d(_get_kin_r(), q_r[:6])
            proprios[f"robot__actual__poses__right::{_ROBOT}__xyz"] = xyz_r
            proprios[f"robot__actual__poses__right::{_ROBOT}__rot_6d"] = rot6d_r
            proprios[f"robot__actual__grippers__right::{_ROBOT}_hand"] = q_r[
                6:7
            ].astype(np.float32)

        return Observation(
            cameras=cameras, proprios=proprios, timestamp=ref_ts_ns * 1e-9
        )

    def _read_proprio(self, name: str) -> Optional[np.ndarray]:
        """Return a snapshot of a proprio buffer, or None if not available."""
        if name not in self.proprios:
            return None
        with self._proprio_locks[name]:
            return self.proprios[name].copy()

    def _compute_extrinsics(
        self,
        camera_name: str,
        q_r: Optional[np.ndarray],
        q_l: Optional[np.ndarray],
    ) -> np.ndarray:
        """Return the current 4×4 extrinsic matrix for *camera_name*.

        Scene cameras return their static matrix.  Wrist cameras compute::

            T_left_base→cam = [T_left_base_from_right_base @] FK(q[:6]) @ T_cam→ee
        """
        arm = _WRIST_CAMERA_ARM.get(camera_name)
        if arm is None or camera_name not in self._T_cam2ee:
            return self._cam_extrinsics[camera_name]

        T_cam2ee = self._T_cam2ee[camera_name]
        q = q_r if arm == "right" else q_l

        if q is None:
            return np.eye(4, dtype=np.float64)

        try:
            if arm == "left":
                with _kin_l_lock:
                    T_base_to_ee = _fk_padded(_get_kin_l(), q[:6])
            else:
                with _kin_r_lock:
                    T_base_to_ee = _fk_padded(_get_kin_r(), q[:6])
            T = T_base_to_ee @ T_cam2ee

            if arm == "right" and self._T_left_base_from_right_base is not None:
                T = self._T_left_base_from_right_base @ T

            return T
        except Exception as e:
            print(f"FK error for '{camera_name}': {e}")
            return np.eye(4, dtype=np.float64)

    # -------------------------------------------------------------------------
    # Camera management
    # -------------------------------------------------------------------------

    def _open_cameras(self) -> None:
        for name in self._raiden_cam_cfg.list_camera_names():
            cam_type = self._raiden_cam_cfg.get_camera_type(name) or "zed"
            serial = self._raiden_cam_cfg.get_serial_by_name(name)
            try:
                if cam_type == "zed":
                    handle = self._open_zed(int(serial))
                else:
                    handle = self._open_realsense(name)
                self._cam_handles[name] = handle
                print(
                    f"  ✓ Opened {cam_type} camera '{name}' "
                    f"(serial={serial}, {handle['w']}×{handle['h']})"
                )
            except Exception as e:
                print(f"  ✗ Failed to open camera '{name}': {e}")

    def _open_zed(self, serial: int) -> dict:
        import pyzed.sl as sl

        cam = sl.Camera()
        params = sl.InitParameters()
        params.set_from_serial_number(serial)
        params.camera_resolution = sl.RESOLUTION.HD720
        params.camera_fps = 30
        params.depth_mode = (
            sl.DEPTH_MODE.NONE if self._no_depth else sl.DEPTH_MODE.NEURAL_LIGHT
        )
        params.coordinate_units = sl.UNIT.METER
        params.depth_minimum_distance = 0.1
        status = cam.open(params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed (serial={serial}): {status}")

        info = cam.get_camera_information()
        res = info.camera_configuration.resolution
        cal_params = info.camera_configuration.calibration_parameters
        cal = cal_params.left_cam
        K = np.array(
            [[cal.fx, 0.0, cal.cx], [0.0, cal.fy, cal.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        handle = {
            "type": "zed",
            "camera": cam,
            "depth": not self._no_depth,
            "image_mat": sl.Mat(),
            "depth_mat": sl.Mat(),
            "h": res.height,
            "w": res.width,
            "intrinsics": K,
            # Convenience callables — avoid re-importing sl outside this method.
            "get_image_ts_ns": lambda: cam.get_timestamp(
                sl.TIME_REFERENCE.IMAGE
            ).get_nanoseconds(),
            "get_current_ts_ns": lambda: cam.get_timestamp(
                sl.TIME_REFERENCE.CURRENT
            ).get_nanoseconds(),
        }
        return handle

    def _open_realsense(self, name: str) -> dict:
        """Open a RealSense camera with the stream settings ``rd record`` uses.

        Resolution, fps, crop and depth come from ``camera.json``, so served
        frames and intrinsics match the converted recordings.
        """
        import pyrealsense2 as rs

        from raiden.cameras.realsense import RealSenseCamera

        entry = self._raiden_cam_cfg.cameras[name]
        depth = entry.get("depth", True) and not self._no_depth
        camera = RealSenseCamera(
            name,
            str(entry["serial"]),
            fps=entry.get("fps", 30),
            resolution=self._raiden_cam_cfg.get_resolution(name),
            crop=self._raiden_cam_cfg.get_crop(name),
            depth=depth,
        )
        camera.open()
        if depth:
            # Align depth to color, as conversion does on bag playback.
            camera._align = rs.align(rs.stream.color)
        K, _, (w, h) = camera.get_intrinsics()
        return {
            "type": "realsense",
            "camera": camera,
            "depth": depth,
            "h": h,
            "w": w,
            "intrinsics": K,
        }

    def _camera_loop(self, name: str) -> None:
        handle = self._cam_handles.get(name)
        if handle is None:
            return
        flip = name in _FLIP_CAMERAS
        if handle["type"] == "zed":
            self._zed_capture_loop(name, handle, flip)
        else:
            self._realsense_capture_loop(name, handle, flip)

    def _zed_capture_loop(self, name: str, handle: dict, flip: bool) -> None:
        import pyzed.sl as sl

        cam = handle["camera"]
        image_mat = handle["image_mat"]
        depth_mat = handle["depth_mat"]
        runtime = sl.RuntimeParameters(
            confidence_threshold=99, texture_confidence_threshold=100
        )
        while self._running:
            if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                # Record hardware capture timestamp immediately after grab, before
                # any processing, so it matches the ZED clock used in the recorder.
                frame_ts_ns = handle["get_image_ts_ns"]()
                cam.retrieve_image(image_mat, sl.VIEW.LEFT)
                # ZED returns BGRA; drop alpha channel → BGR.
                color_bgr = image_mat.get_data()[:, :, :3].copy()

                if not self._no_depth:
                    cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
                    raw = depth_mat.get_data().copy()
                    depth = np.where(np.isfinite(raw), raw, 0.0).astype(np.float32)
                    if flip:
                        color_bgr = cv2.rotate(color_bgr, cv2.ROTATE_180)
                        depth = cv2.rotate(depth, cv2.ROTATE_180)
                    if self._resize is not None:
                        h_out, w_out = self._resize
                        depth = cv2.resize(
                            depth, (w_out, h_out), interpolation=cv2.INTER_LANCZOS4
                        )
                    if name in self.depths:
                        self.update_depth(name, depth)
                else:
                    if flip:
                        color_bgr = cv2.rotate(color_bgr, cv2.ROTATE_180)

                # Resize and serve RGB to the policy.
                if self._resize is not None:
                    h_out, w_out = self._resize
                    color_bgr = cv2.resize(
                        color_bgr, (w_out, h_out), interpolation=cv2.INTER_LANCZOS4
                    )
                self.update_image(name, color_bgr[..., ::-1].copy())
                with self._cam_ts_locks[name]:
                    self._cam_capture_ts_ns[name] = frame_ts_ns

    def _realsense_capture_loop(self, name: str, handle: dict, flip: bool) -> None:
        camera = handle["camera"]
        while self._running:
            try:
                if not camera.grab():
                    continue
                frame = camera.get_frame()  # BGR uint8, cropped
                # Stamp with the RealSense global-time device timestamp, the
                # same field the recorder stores (realsense.py get_frame).
                # Host time after grab() runs ~33 ms (one frame period of
                # USB transport) later than this, which would pair each image
                # with joint state one control period newer than in training.
                # Global time is host-epoch, so it is comparable to the
                # time.time_ns() proprio clock used when no ZED is present.
                frame_ts_ns = frame.timestamp_ns
                if frame_ts_ns <= 0:
                    frame_ts_ns = time.time_ns()
                color_bgr = frame.color
                depth = frame.depth
                if flip:
                    color_bgr = cv2.rotate(color_bgr, cv2.ROTATE_180)
                    if depth is not None:
                        depth = cv2.rotate(depth, cv2.ROTATE_180)
                if self._resize is not None:
                    h_out, w_out = self._resize
                    color_bgr = cv2.resize(
                        color_bgr, (w_out, h_out), interpolation=cv2.INTER_LANCZOS4
                    )
                    if depth is not None:
                        depth = cv2.resize(
                            depth, (w_out, h_out), interpolation=cv2.INTER_LANCZOS4
                        )
                # Serve RGB to the policy.
                self.update_image(name, color_bgr[..., ::-1])
                if depth is not None and name in self.depths:
                    self.update_depth(name, depth)
                with self._cam_ts_locks[name]:
                    self._cam_capture_ts_ns[name] = frame_ts_ns
            except RuntimeError:
                time.sleep(0.01)

    # -------------------------------------------------------------------------
    # Proprioception
    # -------------------------------------------------------------------------

    def _proprio_loop(self) -> None:
        """Read robot joint state at ~100 Hz, push to all buffers under one timestamp.

        A single loop matches the recorder's _robot_loop: one get_all_observations()
        call per cycle so left and right joints are read atomically and share the
        same timestamp.  Timestamped before reading obs, using the ZED SDK clock
        (Unix ns) to match the training data clock.
        """
        target_dt = 1.0 / 100.0
        while self._running:
            loop_start = time.monotonic()
            try:
                # Timestamp before reading — matches recorder._robot_loop.
                ts = self._get_zed_current_time_ns()
                obs_all = self._robot.get_all_observations()

                updates: dict[str, np.ndarray] = {}
                if "follower_r" in obs_all:
                    obs = obs_all["follower_r"]
                    updates["follower_r_joint_pos"] = np.concatenate(
                        [obs["joint_pos"], obs["gripper_pos"].reshape(1)]
                    )
                if "follower_l" in obs_all:
                    obs = obs_all["follower_l"]
                    updates["follower_l_joint_pos"] = np.concatenate(
                        [obs["joint_pos"], obs["gripper_pos"].reshape(1)]
                    )

                for name, val in updates.items():
                    self.update_proprio(name, val)
                    if name in self._proprio_history:
                        with self._proprio_history_locks[name]:
                            self._proprio_history[name].append((ts, val.copy()))

            except Exception as e:
                print(f"Proprio read error: {e}")
            elapsed = time.monotonic() - loop_start
            sleep_time = target_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # -------------------------------------------------------------------------
    # Rerun visualization
    # -------------------------------------------------------------------------

    def _rerun_loop(self) -> None:
        """Stream camera images to a Rerun viewer at 30 FPS.

        Uses the same image buffers filled by the camera capture loops, so
        visualization adds no extra camera I/O.  Starts a gRPC + web server
        so the viewer is accessible from a browser or via SSH tunnel.
        """
        from urllib.parse import quote

        import rerun as rr

        rr.init("raiden_server")
        grpc_port = 9878
        web_port = 9877
        server_uri = rr.serve_grpc(grpc_port=grpc_port)
        rr.serve_web_viewer(web_port=web_port, open_browser=False)
        viewer_url = f"http://localhost:{web_port}?url={quote(server_uri, safe='')}"
        print(f"\nRerun viewer:    {viewer_url}")
        print(
            f"SSH tunnel:      ssh -L {web_port}:localhost:{web_port}"
            f" -L {grpc_port}:localhost:{grpc_port} <host>"
        )

        target_dt = 1.0 / 30.0
        frame_idx = 0
        while self._running:
            loop_start = time.monotonic()
            rr.set_time("frame", sequence=frame_idx)
            for c in self._configs:
                with self._locks[c.name]:
                    img = self.images[c.name].copy()
                rr.log(f"cameras/{c.name}", rr.Image(img))
            frame_idx += 1
            elapsed = time.monotonic() - loop_start
            remaining = target_dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def close(self) -> None:
        """Stop capture threads and release cameras and robot connections."""
        self._running = False
        time.sleep(0.1)
        self._robot.shutdown()
        for handle in self._cam_handles.values():
            try:
                handle["camera"].close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_server(
    camera_config_file: str = "",
    calibration_file: str = "",
    host: str = "0.0.0.0",
    port: int = 8765,
    max_joint_delta: float = _DEFAULT_MAX_JOINT_DELTA,
    action_type: str = "ee_pose",
    no_depth: bool = False,
    resize_images_size: Optional[Tuple[int, int]] = None,
    visualize: bool = False,
    arms: str = "bimanual",
    control_hz: float = _DEFAULT_CONTROL_HZ,
) -> None:
    """Start the Raiden chiral policy server."""
    from raiden._config import CALIBRATION_FILE, CAMERA_CONFIG

    server = RaidenPolicyServer(
        camera_config_file=camera_config_file or CAMERA_CONFIG,
        calibration_file=calibration_file or CALIBRATION_FILE,
        host=host,
        port=port,
        max_joint_delta=max_joint_delta,
        action_type=action_type,
        no_depth=no_depth,
        resize_images_size=resize_images_size,
        visualize=visualize,
        arms=arms,
        control_hz=control_hz,
    )
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.close()
