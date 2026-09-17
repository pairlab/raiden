#!/usr/bin/env python3
"""Capture the real multi-view teaser set: one locked arm pose, several scene-camera poses,
several scene states per camera pose.

1. ARM: the arm starts in gravity compensation. Pose it so the wrist camera sees the spot
   where the ChArUco board will lie, and so it looks right in the figure. ENTER locks it
   there under position control for the rest of the session; those joints are the locked
   pose. BACKSPACE unlocks it again until the first shot.
2. EXPOSURE: board and hands out of the scene view. ENTER matches the scene camera
   exposure, gain and white balance to the current auto image (about 15 s) and keeps them
   fixed for every shot.

Then, for each of --poses camera poses:

3. BOARD shot: set the scene camera, put the board where both cameras see it. ENTER takes
   one frame per camera. The locked joints give the wrist camera pose (FK x hand-eye), the
   wrist camera gives the board pose, so ``T_base_cam = T_base_board @ inv(T_scene_board)``.
4. TEASER shots, one per scene state (--states, A, B, ...): take the board away, arrange the
   objects, keep clear and do not touch the camera. ENTER takes the shot. State A must be
   the same at every pose (mark the object positions with tape).

Every shot is refused if the arm is more than --max-drift from the locked pose. A board shot
is refused if the board solve fails, a teaser shot if the scene camera still sees the board.

Camera pose bounds. While the board is in view the status bar shows the camera as SNAP
fine-tune orbit knobs about the canonical pivot, relative to the calibrated camera (the
inverse of ``orbit_base()`` in Phoenix ``probes/teaser_canon_pca.py``). OUT marks a value
outside its trained band:

    yaw      [-156.2, -20] deg   canonical pose at -22.5; [-20, +23.8] is held out
    pitch    [0, +20] deg        only raise the camera, never lower it
    dolly    [0.7, 1.2]          x 0.958 m, the calibrated distance to the pivot
    rot off  < 10 deg            angle between the camera rotation and the orbit's

The pivot lies on the calibrated camera's optical axis, so the orbit keeps it at the
principal point. The scene view draws the principal point (cyan o) and the pivot seen from
the current camera (magenta +): put the + on the o and do not roll the camera.

Every shot writes ``<out>/pose<i>_board/`` or ``<out>/pose<i>_teaser_<state>/`` (a redo
overwrites it)::

    scene_camera.png               640x480 native colour, lossless (the token input)
    scene_camera_depth_m.npy       depth aligned to colour: median over --frames (teaser), one frame (board)
    left_wrist_camera.png          wrist colour (cropped as in camera.json)
    left_wrist_camera_depth_m.npy
    meta.json                      joints, intrinsics, exposure, T_base_cam, board solve

Teaser shots also write a high-resolution copy, for print only::

    scene_camera_1080p.png         full 1920x1080 frame
    scene_camera_sq1080.png        1080x1080 crop x[420:1500]

The 640x480 colour mode is the 1920x1080 frame x[240:1680] scaled by 1/2.25, so the
1080x1080 crop has the same field of view as the policy's 480x480 centre crop.
``<out>/poses.json`` collects ``T_base_cam`` (robot base frame, OpenCV axes,
camera-to-base), the orbit knobs and the shot names per pose.

Keys in the preview window: ENTER next, BACKSPACE back (a teaser shot goes back one step;
before the first shot it unlocks the arm), ESC quit. The scene view also shows the policy
crop (yellow) and the detected board corners. On exit the arm goes back to gravity comp
and the scene camera to auto exposure.

Usage::

    sudo ip link set can_follower_l up type can bitrate 1000000   # if the link is DOWN
    uv run python scripts/teaser_capture.py
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import signal
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from raiden._config import CALIBRATION_FILE, CAMERA_CONFIG
from raiden.calibration.runner import compute_forward_kinematics
from raiden.camera_config import CameraConfig
from raiden.cameras.realsense import RealSenseCamera, rs

SCENE, WRIST = "scene_camera", "left_wrist_camera"
HIRES = (1920, 1080)
SQ1080 = (420, 0, 1080, 1080)  # x, y, w, h in the 1920x1080 frame
POLICY_CROP = (80, 0, 480, 480)  # x, y, w, h in the 640x480 frame
# Canonical orbit pivot, robot base frame, and the trained bands of the SNAP fine-tune
# (Phoenix, verified against the v2 render attributes on 2026-09-14).
PIVOT = np.array([0.29839303, -0.06934502, 0.03011114])
BANDS = {
    "yaw_deg": (-156.2, -20.0),
    "pitch_deg": (0.0, 20.0),
    "dolly": (0.7, 1.2),
    "rot_off_deg": (0.0, 10.0),
}
# Colour exposure in 100 us units, multiples of 1/120 s so 60 Hz lights do not flicker.
FLICKER_SAFE_EXPOSURES = (250, 333, 167, 83)
FREE, LOCKED = 0, 1  # arm modes
KEYS_ENTER, KEY_BACK, KEY_ESC = (10, 13), 8, 27
WINDOW = "teaser capture"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def make_T(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R)
    T[:3, 3] = np.ravel(t)
    return T


def rot_angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    cos = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def load_calibration() -> dict:
    d = json.loads(Path(CALIBRATION_FILE).read_text())
    cams = d["cameras"]
    cc = d["charuco_config"]
    board = cv2.aruco.CharucoBoard(
        (cc["squares_x"], cc["squares_y"]),
        cc["square_length"],
        cc["marker_length"],
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cc["dictionary"])),
    )
    hand_eye = cams[WRIST]["hand_eye_calibration"]
    scene = cams[SCENE]["extrinsics"]
    return {
        "K": {n: np.array(cams[n]["intrinsics"]["camera_matrix"]) for n in (SCENE, WRIST)},
        "D": {
            n: np.array(cams[n]["intrinsics"]["distortion_coeffs"], dtype=float)
            for n in (SCENE, WRIST)
        },
        # wrist camera pose in the grasp_site frame
        "X": make_T(hand_eye["rotation_matrix"], hand_eye["translation_vector"]),
        "T_base_scene": make_T(scene["rotation_matrix"], scene["translation_vector"]),
        "detector": cv2.aruco.CharucoDetector(board),
        "obj": board.getChessboardCorners(),
        "source": f"{CALIBRATION_FILE} ({d.get('timestamp')}, {d.get('source')})",
    }


# ---------------------------------------------------------------------------
# Board and pose
# ---------------------------------------------------------------------------


def board_pose(cal: dict, cam: str, color: np.ndarray):
    """Return (T_cam_board or None, corners, ids, reprojection RMS px or None)."""
    corners, ids, _, _ = cal["detector"].detectBoard(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY))
    if ids is None or len(ids) < 6:
        return None, corners, ids, None
    obj = cal["obj"][ids.ravel()]
    ok, rvec, tvec = cv2.solvePnP(obj, corners, cal["K"][cam], cal["D"][cam])
    if not ok:
        return None, corners, ids, None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, cal["K"][cam], cal["D"][cam])
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - corners.reshape(-1, 2)) ** 2, 1))))
    return make_T(cv2.Rodrigues(rvec)[0], tvec), corners, ids, rms


def solve_scene_pose(cal: dict, T_base_wrist, scene_color, wrist_color):
    """Return (T_base_cam or None, info dict, {camera: (corners, ids)})."""
    T_sb, cs, ids_s, rms_s = board_pose(cal, SCENE, scene_color)
    T_wb, cw, ids_w, rms_w = board_pose(cal, WRIST, wrist_color)
    info = {
        "scene_corners": 0 if ids_s is None else len(ids_s),
        "wrist_corners": 0 if ids_w is None else len(ids_w),
        "scene_rms_px": rms_s,
        "wrist_rms_px": rms_w,
    }
    detections = {SCENE: (cs, ids_s), WRIST: (cw, ids_w)}
    if T_sb is None or T_wb is None or T_base_wrist is None:
        return None, info, detections
    T_base_board = T_base_wrist @ T_wb
    T_base_cam = T_base_board @ np.linalg.inv(T_sb)
    info.update(
        T_scene_board=T_sb, T_wrist_board=T_wb, T_base_board=T_base_board, T_base_cam=T_base_cam
    )
    return T_base_cam, info, detections


def orbit_base(T0: np.ndarray, yaw_deg: float, pitch_deg: float = 0.0, dolly: float = 1.0):
    """Phoenix's orbit_base(): the calibrated pose T0 orbited about PIVOT."""
    v = T0[:3, 3] - PIVOT
    horiz = np.cross([0.0, 0.0, 1.0], v)
    R = (
        Rotation.from_rotvec([0.0, 0.0, np.radians(yaw_deg)]).as_matrix()
        @ Rotation.from_rotvec(horiz / np.linalg.norm(horiz) * -np.radians(pitch_deg)).as_matrix()
    )
    return make_T(R @ T0[:3, :3], PIVOT + R @ v * dolly)


def orbit(T: np.ndarray, T0: np.ndarray) -> dict:
    """Inverse of orbit_base(): yaw, pitch and dolly of T relative to T0, and how far the
    rotation of T is from the rotation the orbit would give it."""
    v, v0 = T[:3, 3] - PIVOT, T0[:3, 3] - PIVOT
    yaw = np.degrees(np.arctan2(v[1], v[0]) - np.arctan2(v0[1], v0[0]))
    yaw = (yaw + 180.0) % 360.0 - 180.0
    pitch = np.degrees(np.arcsin(v[2] / np.linalg.norm(v)) - np.arcsin(v0[2] / np.linalg.norm(v0)))
    dolly = np.linalg.norm(v) / np.linalg.norm(v0)
    return {
        "yaw_deg": float(yaw),
        "pitch_deg": float(pitch),
        "dolly": float(dolly),
        "rot_off_deg": rot_angle(orbit_base(T0, yaw, pitch)[:3, :3], T[:3, :3]),
        "dist_pivot_m": float(np.linalg.norm(v)),
    }


def orbit_text(o: dict) -> str:
    parts = []
    for key, label, fmt in (
        ("yaw_deg", "yaw", "{:+.1f}"),
        ("pitch_deg", "pitch", "{:+.1f}"),
        ("dolly", "dolly", "{:.2f}"),
        ("rot_off_deg", "rot off", "{:.1f}"),
    ):
        lo, hi = BANDS[key]
        parts.append(f"{label} {fmt.format(o[key])}" + ("" if lo <= o[key] <= hi else " OUT"))
    return "   ".join(parts)


def pivot_pixel(cal: dict, T_base_cam: np.ndarray):
    """Pixel of PIVOT in the scene image, or None when it is behind the camera."""
    p = np.linalg.inv(T_base_cam) @ np.append(PIVOT, 1.0)
    if p[2] <= 0:
        return None
    uv, _ = cv2.projectPoints(p[:3].reshape(1, 3), np.zeros(3), np.zeros(3), cal["K"][SCENE], cal["D"][SCENE])
    return tuple(int(round(v)) for v in uv.ravel())


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------


def open_camera(cfg: CameraConfig, name: str, resolution=None, depth: bool = True):
    """Open as in camera.json, or at *resolution* with no crop."""
    entry = cfg.cameras[name]
    cam = RealSenseCamera(
        name,
        str(entry["serial"]),
        resolution=resolution or entry.get("resolution"),
        crop=None if resolution else cfg.get_crop(name),
        depth=depth,
    )
    cam.open()
    return cam


def measure(cam: RealSenseCamera, settle: int = 12, n: int = 3) -> tuple[float, float]:
    """Mean brightness and blue/red ratio inside the policy crop, after *settle* frames."""
    for _ in range(settle):
        cam.grab()
    x, y, w, h = POLICY_CROP
    values = []
    for _ in range(n):
        cam.grab()
        img = cam.get_frame().color[y : y + h, x : x + w].astype(np.float32)
        values.append((img.mean(), img[..., 0].mean() / max(img[..., 2].mean(), 1e-3)))
    return tuple(float(v) for v in np.mean(values, axis=0))


def bisect(set_value, lo: float, hi: float, read, target: float, increasing: bool, iters: int) -> float:
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        set_value(mid)
        if (read() < target) == increasing:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def match_auto_exposure(cam: RealSenseCamera) -> dict:
    """Manual white balance, exposure and gain that reproduce the current auto image.

    This D435I reports no frame metadata and its option readback is not live under auto,
    so the auto values cannot be read. Measure the auto image in the policy crop instead,
    then search: white balance on the blue/red ratio (while auto exposure still holds the
    brightness), then gain at a flicker-safe exposure. Takes about 15 s.
    """
    sensor = cam._profile.get_device().first_color_sensor()
    sensor.set_option(rs.option.enable_auto_exposure, 1)
    sensor.set_option(rs.option.enable_auto_white_balance, 1)
    target_mean, target_ratio = measure(cam, settle=30, n=10)

    def set_wb(v):
        sensor.set_option(rs.option.white_balance, float(round(v / 10.0) * 10))

    def set_gain(v):
        sensor.set_option(rs.option.gain, float(round(v)))

    sensor.set_option(rs.option.enable_auto_white_balance, 0)
    wb = bisect(set_wb, 2800, 6500, lambda: measure(cam)[1], target_ratio, increasing=False, iters=9)
    set_wb(wb)

    sensor.set_option(rs.option.enable_auto_exposure, 0)
    exposure = FLICKER_SAFE_EXPOSURES[0]
    for candidate in FLICKER_SAFE_EXPOSURES:
        sensor.set_option(rs.option.exposure, candidate)
        set_gain(0)
        darkest = measure(cam)[0]
        set_gain(128)
        if darkest <= target_mean <= measure(cam)[0]:
            exposure = candidate
            break
    sensor.set_option(rs.option.exposure, exposure)
    gain = bisect(set_gain, 0, 128, lambda: measure(cam)[0], target_mean, increasing=True, iters=7)
    set_gain(gain)

    mean, ratio = measure(cam, n=5)
    return {
        "exposure": exposure,
        "gain": round(gain),
        "white_balance": round(wb / 10.0) * 10,
        "auto_target": {"mean": target_mean, "blue_red": target_ratio},
        "achieved": {"mean": mean, "blue_red": ratio},
    }


def apply_exposure(serial: str, exposure: dict | None) -> None:
    """Fix the colour sensor at *exposure*, or give it back to auto when None."""
    for dev in rs.context().query_devices():
        if dev.get_info(rs.camera_info.serial_number) != serial:
            continue
        sensor = dev.first_color_sensor()
        sensor.set_option(rs.option.enable_auto_exposure, 0 if exposure else 1)
        sensor.set_option(rs.option.enable_auto_white_balance, 0 if exposure else 1)
        for key, option in (
            ("exposure", rs.option.exposure),
            ("gain", rs.option.gain),
            ("white_balance", rs.option.white_balance),
        ):
            if exposure and key in exposure:
                sensor.set_option(option, exposure[key])


def grab_median(cams: dict, n_frames: int) -> dict:
    """Return {camera: (last colour, median depth in m aligned to colour)}."""
    colors = {}
    depths = {name: [] for name in cams}
    for cam in cams.values():
        cam._align = rs.align(rs.stream.color)
    try:
        while min(len(d) for d in depths.values()) < n_frames:
            for name, cam in cams.items():
                if len(depths[name]) < n_frames and cam.grab():
                    frame = cam.get_frame()
                    colors[name] = frame.color
                    depths[name].append(frame.depth)
    finally:
        for cam in cams.values():
            cam._align = None
    out = {}
    for name in cams:
        stack = np.stack(depths[name])
        stack[stack <= 0] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # pixels with no depth at all
            median = np.nanmedian(stack, axis=0)
        out[name] = (colors[name], np.nan_to_num(median, nan=0.0).astype(np.float32))
    return out


# ---------------------------------------------------------------------------
# Arm
# ---------------------------------------------------------------------------


def arm_process(joints, target, mode, stop) -> None:
    """Child process: run the left follower and publish its joints.

    *target* asks for FREE (gravity comp) or LOCKED (position hold where the arm is now);
    *mode* reports the mode in effect, set only after the switch and a fresh joint reading.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C goes to the parent, which lowers the arm first
    from raiden.robot.controller import RobotController

    controller = RobotController(
        use_right_leader=False,
        use_left_leader=False,
        use_right_follower=False,
        use_left_follower=True,
    )
    try:
        controller.check_can_interfaces()
        controller.initialize_robots(gravity_comp_mode=False)
        follower = controller.follower_l
        current = None
        while not stop.is_set():
            if not follower.motor_chain.running:
                print("\n*** i2rt motor loop stopped: the arm is not powered ***", flush=True)
                return
            wanted = target.value
            if wanted != current:
                if wanted == LOCKED:
                    controller.disable_gravity_compensation()
                else:
                    controller.enable_gravity_compensation()
                # i2rt stores new gains; they reach the motors with the next command.
                follower.command_joint_pos(follower.get_joint_pos())
                if wanted == LOCKED:
                    time.sleep(1.0)  # settle before the joints become the locked pose
                current = wanted
            with joints.get_lock():
                joints[:] = follower.get_joint_pos()
            mode.value = current
            time.sleep(0.005)
    finally:
        controller.close()


class Arm:
    """Left follower in its own process: gravity comp to pose it, position hold to lock it.

    i2rt computes the gravity torque in a Python thread that must hear back from each motor
    within 9 ms. pyrealsense2 holds the GIL for ~100 ms in calls such as query_devices and
    pipeline start (the 1080p grab), which stops that loop and drops the arm. A separate
    process has its own GIL.
    """

    def __init__(self):
        ctx = mp.get_context("spawn")  # not fork: the camera pipelines already run threads
        self._joints = ctx.Array("d", 7)  # 6 arm joints + gripper
        self._target = ctx.Value("i", FREE)
        self._mode = ctx.Value("i", -1)
        self._stop = ctx.Event()
        input("\n*** Hold the arm: it goes limp when gravity comp starts. Press Enter. ***")
        self._proc = ctx.Process(
            target=arm_process, args=(self._joints, self._target, self._mode, self._stop), daemon=True
        )
        self._proc.start()
        self._wait(FREE)

    def _wait(self, mode: int) -> None:
        while self._mode.value != mode:
            if not self.alive():
                raise RuntimeError("the arm process has stopped, see its output above")
            time.sleep(0.05)

    def alive(self) -> bool:
        return self._proc.is_alive()

    def joints(self) -> np.ndarray:
        if not self.alive():
            raise RuntimeError("the arm process has stopped: the arm is not powered")
        with self._joints.get_lock():
            return np.array(self._joints[:])

    def lock(self) -> np.ndarray:
        """Hold the arm where it is; return the settled joints."""
        self._target.value = LOCKED
        self._wait(LOCKED)
        return self.joints()

    def free(self) -> None:
        self._target.value = FREE
        self._wait(FREE)

    def close(self) -> None:
        self._stop.set()
        self._proc.join(timeout=15)


# ---------------------------------------------------------------------------
# Capture flow
# ---------------------------------------------------------------------------


def build_steps(n_poses: int, states: list[str]) -> list[tuple]:
    """(kind, pose, state, title, prompt) per step."""
    steps = [
        ("arm", None, None, "Arm pose",
         "Gravity comp: pose the arm so the wrist camera sees the board spot.  ENTER: lock the arm"),
        ("exposure", None, None, "Exposure",
         "Board and hands out of the scene view.  ENTER: lock exposure (15 s)   BACKSPACE: unlock the arm"),
    ]
    for i in range(n_poses):
        tag = f"Pose {i}/{n_poses - 1}"
        steps.append(("board", i, None, f"{tag}: board shot",
                      "Set the scene camera (magenta + on cyan o). Board in view of both cameras.  ENTER: board shot"))
        for s in states:
            where = " (taped positions)" if s == states[0] else ""
            steps.append(("teaser", i, s, f"{tag}: teaser shot, state {s}",
                          f"Board away, hands clear, camera untouched. Objects in state {s}{where}.  ENTER: teaser shot"))
    return steps


def compose(frames: dict, detections: dict, lines: list[tuple], target=None, pivot=None) -> np.ndarray:
    tiles = []
    for name in (SCENE, WRIST):
        img = frames.get(name)
        img = np.zeros((480, 640, 3), np.uint8) if img is None else img.copy()
        if name == SCENE:
            x, y, w, h = POLICY_CROP
            cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), (0, 255, 255), 1)
            if pivot is not None:
                cv2.circle(img, target, 8, (255, 255, 0), 2)
                cv2.drawMarker(img, pivot, (255, 0, 255), cv2.MARKER_CROSS, 24, 2)
        corners, ids = detections.get(name, (None, None))
        if ids is not None and len(ids):
            cv2.aruco.drawDetectedCornersCharuco(img, corners, ids, (0, 255, 0))
        cv2.putText(img, name, (8, 470), FONT, 0.5, (255, 255, 255), 1)
        tiles.append(img)
    view = np.hstack(tiles)
    bar = np.zeros((10 + 24 * len(lines), view.shape[1], 3), np.uint8)
    for i, (line, color) in enumerate(lines):
        cv2.putText(bar, line, (8, 24 + 24 * i), FONT, 0.55, color, 1)
    return np.vstack([bar, view])


def to_json(obj) -> str:
    return json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


class Capture:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out or f"data/teaser/real_mv/{datetime.now():%Y%m%d_%H%M%S}")
        self.out.mkdir(parents=True, exist_ok=True)
        self.cal = load_calibration()
        self.cfg = CameraConfig(CAMERA_CONFIG)
        self.scene_serial = str(self.cfg.cameras[SCENE]["serial"])
        self.states = [chr(ord("A") + s) for s in range(args.states)]
        self.cams: dict = {}
        self.device_K: dict = {}
        self.arm: Arm | None = None
        self.locked = None  # arm joints held for every shot
        self.exposure: dict | None = None
        self.exposure_match: dict | None = None
        self.poses: dict = {}

    def open(self) -> None:
        print("Opening cameras...")
        apply_exposure(self.scene_serial, None)  # start from auto, like the demos
        for name in (SCENE, WRIST):
            self.cams[name] = open_camera(self.cfg, name)
            K, dist, size = self.cams[name].get_intrinsics()
            self.device_K[name] = {"K": K, "distortion": dist, "image_size": list(size)}
        self.arm = Arm()

    def close(self) -> None:
        cv2.destroyAllWindows()
        for cam in self.cams.values():
            cam.close()
        apply_exposure(self.scene_serial, None)
        if self.arm is not None:
            if self.arm.alive():
                try:
                    self.arm.free()
                    input("\nGravity comp is on. Lower the arm to rest, then press Enter to exit.")
                except (KeyboardInterrupt, EOFError, RuntimeError):
                    pass
            else:
                print("\nThe arm process has stopped: the arm is not powered.")
            self.arm.close()
        print(f"Done. Shots in {self.out}")

    def lock_exposure(self) -> None:
        print("matching the auto exposure with manual settings (about 15 s, keep the scene still) ...")
        match = match_auto_exposure(self.cams[SCENE])
        self.exposure = {k: match[k] for k in ("exposure", "gain", "white_balance")}
        self.exposure_match = match
        print(f"scene camera locked: {match}")

    def status(self, kind: str, frames: dict):
        """Status lines, board detections and the pivot pixel for the live preview."""
        if len(frames) < 2:
            return [], {}, None
        joints = self.arm.joints()
        T_base_wrist = None
        if kind == "board":
            T_base_wrist = compute_forward_kinematics(joints[:6]) @ self.cal["X"]
        T, info, detections = solve_scene_pose(self.cal, T_base_wrist, frames[SCENE], frames[WRIST])
        line = f"board corners: scene {info['scene_corners']}, wrist {info['wrist_corners']}"
        if kind in ("exposure", "teaser") and info["scene_corners"] >= 6:
            line += "   BOARD IN THE SCENE VIEW"
        if self.locked is not None:
            line += f"   |   arm vs locked pose {np.abs(joints[:6] - self.locked[:6]).max():.4f} rad"
        lines = [line]
        pivot = None
        if T is not None:
            lines.append(orbit_text(orbit(T, self.cal["T_base_scene"])))
            pivot = pivot_pixel(self.cal, T)
        return lines, detections, pivot

    def run(self) -> None:
        steps = build_steps(self.args.poses, self.states)
        K = self.cal["K"][SCENE]
        target = (int(round(K[0, 2])), int(round(K[1, 2])))
        k = 0
        try:
            self.open()
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            print(f"\nPreview window is open. Output: {self.out}\n")
            while k < len(steps):
                kind, pose, state, title, prompt = steps[k]
                frames = {name: cam.get_frame().color for name, cam in self.cams.items() if cam.grab()}
                lines, detections, pivot = self.status(kind, frames)
                text = [(title, (0, 255, 0)), (prompt, (255, 255, 255))]
                text += [(line, (0, 220, 255)) for line in lines]
                cv2.imshow(WINDOW, compose(frames, detections, text, target, pivot))
                key = cv2.waitKey(1) & 0xFF

                if key == KEY_ESC:
                    print("ESC: quitting")
                    break
                if key == KEY_BACK:
                    if kind == "teaser":
                        k -= 1
                        print("back one step")
                    elif kind == "exposure" or (kind == "board" and not self.poses):
                        self.arm.free()
                        self.locked = None
                        k = 0
                        print("arm unlocked: gravity comp on")
                    else:
                        print("BACKSPACE: nothing to go back to here")
                elif key in KEYS_ENTER:
                    if kind == "arm":
                        self.locked = self.arm.lock()
                        print(f"arm locked at {np.array2string(self.locked, precision=4, suppress_small=True)}")
                        k += 1
                    elif kind == "exposure":
                        self.lock_exposure()
                        k += 1
                    elif self.take_shot(kind, pose, state):
                        k += 1
        except KeyboardInterrupt:
            print("\ninterrupted")
        finally:
            self.close()

    def grab_hires(self):
        """Reopen the scene camera at 1920x1080 colour, grab one frame, go back to native."""
        scene = self.cams[SCENE]
        scene.close()
        hires = open_camera(self.cfg, SCENE, resolution=HIRES, depth=False)
        try:
            apply_exposure(self.scene_serial, self.exposure)
            for _ in range(self.args.hires_warmup):
                hires.grab()
            while not hires.grab():
                pass
            color = hires.get_frame().color.copy()
            K, dist, size = hires.get_intrinsics()
        finally:
            hires.close()
            scene.open()
            apply_exposure(self.scene_serial, self.exposure)
        return color, {"K": K, "distortion": dist, "image_size": list(size)}

    def take_shot(self, kind: str, pose: int, state: str | None) -> bool:
        """Capture one board or teaser shot. Returns False if it must be redone."""
        name = f"pose{pose}_board" if kind == "board" else f"pose{pose}_teaser_{state}"
        print(f"[{name}] capturing, hold still ...", flush=True)

        # The board shot is one frame: the joints read around it match the wrist image.
        n_frames = 1 if kind == "board" else self.args.frames
        before = self.arm.joints()
        grabs = grab_median(self.cams, n_frames)
        joints = self.arm.joints()
        drift = float(max(np.abs(j[:6] - self.locked[:6]).max() for j in (before, joints)))
        if drift > self.args.max_drift:
            print(f"[{name}] the arm is {drift:.4f} rad from the locked pose (> {self.args.max_drift}). "
                  "Was it bumped? Let it settle and retry.")
            return False

        T_base_wrist = compute_forward_kinematics(joints[:6]) @ self.cal["X"]
        T, info, _ = solve_scene_pose(self.cal, T_base_wrist, grabs[SCENE][0], grabs[WRIST][0])
        if kind == "board" and T is None:
            print(f"[{name}] board solve failed ({info['scene_corners']} scene / "
                  f"{info['wrist_corners']} wrist corners, need 6 each). Move the board and retry.")
            return False
        if kind == "teaser" and info["scene_corners"] >= 6:
            print(f"[{name}] the scene camera still sees the board. Take it away and retry.")
            return False

        shot = self.out / name
        shot.mkdir(parents=True, exist_ok=True)
        for cam_name, (color, depth) in grabs.items():
            cv2.imwrite(str(shot / f"{cam_name}.png"), color)
            np.save(shot / f"{cam_name}_depth_m.npy", depth)

        entry = self.poses.setdefault(f"pose{pose}", {})
        meta = {
            "kind": kind,
            "pose": pose,
            "state": state,
            "time": datetime.now().isoformat(),
            "joints": joints,
            "locked_joints": self.locked,
            "arm_drift_rad": drift,
            "depth_median_frames": n_frames,
            "scene_exposure_locked": self.exposure_match,
            "calibration": self.cal["source"],
            "cameras": {
                n: {
                    "image": f"{n}.png",
                    "depth_m": f"{n}_depth_m.npy",
                    "K_calibrated": self.cal["K"][n],
                    "distortion_calibrated": self.cal["D"][n],
                    "K_device": self.device_K[n]["K"],
                    "distortion_device": self.device_K[n]["distortion"],
                    "image_size": self.device_K[n]["image_size"],
                    "crop_in_stream": self.cfg.get_crop(n),
                }
                for n in self.cams
            },
        }
        if kind == "board":
            o = orbit(T, self.cal["T_base_scene"])
            meta.update(T_base_cam=T, T_base_wrist_cam=T_base_wrist, orbit=o, board=info)
            entry.update(T_base_cam=T, orbit=o, board_shot=name)
            print(f"[{name}] {orbit_text(o)}; RMS scene {info['scene_rms_px']:.2f} px, "
                  f"wrist {info['wrist_rms_px']:.2f} px")
        else:
            hires, hires_K = self.grab_hires()
            x, y, w, h = SQ1080
            cv2.imwrite(str(shot / "scene_camera_1080p.png"), hires)
            cv2.imwrite(str(shot / "scene_camera_sq1080.png"), hires[y : y + h, x : x + w])
            meta.update(
                T_base_cam=entry["T_base_cam"],
                board_shot=entry["board_shot"],
                hires={**hires_K, "square_crop_xywh": SQ1080, "use": "print only, not tokens"},
            )
            entry.setdefault("teaser_shots", {})[state] = name

        (shot / "meta.json").write_text(to_json(meta))
        (self.out / "poses.json").write_text(to_json({
            "frame": "robot base (left_arm_base), OpenCV axes, camera-to-base",
            "calibration": self.cal["source"],
            "locked_joints": self.locked,
            "scene_exposure": self.exposure,
            "states": self.states,
            "poses": self.poses,
        }))
        print(f"[{name}] saved {shot}")
        return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="default: data/teaser/real_mv/<date_time>")
    ap.add_argument("--poses", type=int, default=5, help="number of scene camera poses")
    ap.add_argument("--states", type=int, default=2, help="scene states (teaser shots) per camera pose")
    ap.add_argument("--frames", type=int, default=30, help="frames in the teaser depth median")
    ap.add_argument("--hires-warmup", type=int, default=30, help="frames dropped after the 1080p switch")
    ap.add_argument("--max-drift", type=float, default=0.005,
                    help="refuse a shot if any arm joint is further than this from the locked pose (rad)")
    Capture(ap.parse_args()).run()


if __name__ == "__main__":
    sys.exit(main())
