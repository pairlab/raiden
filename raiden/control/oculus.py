"""Meta Quest controller teleoperation (clutched Cartesian control).

The headset rests on the desk with its cameras facing the operator; only the
hand controllers are used.  Button layout follows pairlab/mesa-env:

* joystick click / B (Y) .. calibrate headset->robot axes: hold A (X), move
                            ~20 cm along robot +x, release; repeat along robot +y.
* hold trigger ............ clutch: the end-effector follows the controller
                            relative to where the trigger was pressed.
* hold grip ............... gripper closed; release to open.
* A (X) ................... trigger event (start/stop recording, record pose).

Targets are tracked by mink IK on the YAM MuJoCo model (gripper-tip
``grasp_site``) in ``RobotController.start_cartesian_teleop``.
"""

import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from raiden.control.base import TeleopInterface
from raiden.robot.controller import CARTESIAN_READY_POSE, smooth_move_joints
from raiden.robot.footpedal import (
    PEDAL_LEFT,
    PEDAL_MIDDLE,
    PEDAL_RIGHT,
    try_open_footpedal,
)
from raiden.robot.oculus import OculusReader

_KEYS = {
    "r": {
        "clutch": "RTr",
        "grip": "RG",
        "grip_val": "rightGrip",
        "calib": ("RJ", "B"),
        "btn": "A",
    },
    "l": {
        "clutch": "LTr",
        "grip": "LG",
        "grip_val": "leftGrip",
        "calib": ("LJ", "Y"),
        "btn": "X",
    },
}
_GRIP_THRESHOLD = 0.2  # analog grip above this counts as pressed (mesa-env)
_MIN_CALIB_MOVE = 0.08  # m; shorter calibration moves are rejected
_STALE_AFTER = 0.5  # s without controller data -> hold the arm
_SMOOTHING_TAU = 0.05  # s low-pass on the controller pose
_MAX_LEAD_POS = 0.15  # m the IK target may lead the arm (bounds peak speed)
_MAX_LEAD_ROT = 0.5  # rad
_MAX_REACH = 0.55  # m from the shoulder; keeps the arm off full extension
_SHOULDER = np.array([0.0, 0.0, 0.067])
_DT = 0.01


class OculusInterface(TeleopInterface):
    """Quest hand controllers -> follower arm end-effector pose."""

    def __init__(
        self,
        ip_address: Optional[str] = None,
        hand_for_left_arm: str = "l",
        hand_for_right_arm: str = "r",
        pos_scale: float = 0.7,
        rot_scale: float = 0.5,
    ):
        """
        Args:
            ip_address: Quest IP for Wi-Fi ADB; None = USB.
            hand_for_left_arm / hand_for_right_arm: controller ("l"/"r") per follower.
            pos_scale: robot metres per controller metre.
            rot_scale: 0..1 gain on controller rotation (0 = translation only).
        """
        self._ip = ip_address
        self._hand = {"left": hand_for_left_arm, "right": hand_for_right_arm}
        self._pos_scale = pos_scale
        self._rot_scale = float(np.clip(rot_scale, 0.0, 1.0))
        self._alpha = 1.0 - float(np.exp(-_DT / _SMOOTHING_TAU))
        self._reader: Optional[OculusReader] = None
        self._tracking: Dict[str, str] = {}
        self._stop = threading.Event()
        self._event = threading.Event()
        self._reset_state()

    def _reset_state(self) -> None:
        hands = ("l", "r")
        self._R: Dict[str, Optional[np.ndarray]] = {
            h: None for h in hands
        }  # headset -> robot axes
        self._calib: Dict[str, Optional[dict]] = {h: None for h in hands}
        self._origin: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]] = {
            h: None for h in hands
        }
        self._smooth: Dict[str, Optional[np.ndarray]] = {h: None for h in hands}
        self._prev: Dict[str, Dict[str, bool]] = {
            h: {"calib": False, "btn": False} for h in hands
        }
        self._warned: Dict[str, float] = {h: 0.0 for h in hands}

    @property
    def name(self) -> str:
        return "oculus"

    # ------------------------------------------------------------------
    # Session-level lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        print("Connecting to Quest...")
        self._reader = OculusReader(ip_address=self._ip)
        self._reader.keep_awake(True)
        self._reader.start()
        t0 = time.monotonic()
        while self._reader.age == float("inf") and time.monotonic() - t0 < 10.0:
            time.sleep(0.1)
        if self._reader.age == float("inf"):
            print(
                "  ! No controller data yet. Are the controllers on and in view of the headset?"
            )
        else:
            print("  ✓ Quest controllers streaming")

        self._pedal_trigger = threading.Event()
        self._pedal_success = threading.Event()
        self._pedal_failure = threading.Event()
        self._footpedal = try_open_footpedal()
        if self._footpedal is not None:

            def _cb(code: int) -> None:
                if code == PEDAL_LEFT:
                    rc = getattr(self, "_recording_controller", None)
                    if rc is not None:
                        rc.soft_pause()
                    else:
                        self._pedal_trigger.set()
                elif code == PEDAL_MIDDLE:
                    self._pedal_success.set()
                elif code == PEDAL_RIGHT:
                    self._pedal_failure.set()

            self._footpedal.on_press(_cb)
            self._footpedal.start()
            print(
                "  ✓ FootPedal ready: left=trigger/pause  middle=success  right=failure"
            )

    def close(self) -> None:
        if getattr(self, "_footpedal", None) is not None:
            self._footpedal.close()
            self._footpedal = None
        if self._reader is not None:
            self._reader.stop()
            try:
                self._reader.keep_awake(False)
            except Exception:
                pass
            self._reader = None

    # ------------------------------------------------------------------
    # Episode-level lifecycle
    # ------------------------------------------------------------------

    def setup(self, robot_controller) -> None:
        # The all-zero home pose is folded against two joint limits; unfold first.
        print("  - Moving to teleop ready pose (3 s)...", flush=True)
        threads = []
        for follower in (robot_controller.follower_r, robot_controller.follower_l):
            if follower is None:
                continue
            target = np.append(CARTESIAN_READY_POSE, follower.get_joint_pos()[6])
            t = threading.Thread(
                target=smooth_move_joints,
                args=(follower, target),
                kwargs={"time_interval_s": 3.0, "steps": 300},
                daemon=True,
            )
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    def start(self, robot_controller) -> None:
        self._reset_state()
        robot_controller.start_cartesian_teleop(self._target_fn, dt=_DT)
        self._stop.clear()
        threading.Thread(
            target=self._tracking_loop, name="oculus-tracking", daemon=True
        ).start()

    def stop(self, robot_controller) -> None:
        self._stop.set()
        robot_controller.stop_cartesian_teleop()

    def _tracking_loop(self) -> None:
        """Poll the OS tracking state ~1 Hz; it gates the clutch (see _target_fn)."""
        while not self._stop.wait(1.0):
            reader = self._reader
            if reader is None:
                continue
            tracking = reader.controller_tracking()
            for hand, state in tracking.items():
                if hand not in self._hand.values() or state == self._tracking.get(hand):
                    continue
                if state == "POSITION":
                    print(f"\n  [Oculus {hand.upper()}] tracking locked", flush=True)
                elif self._tracking.get(hand) == "POSITION":
                    print(
                        f"\n  [Oculus {hand.upper()}] tracking lost ({state}); clutch ignored until it re-locks",
                        flush=True,
                    )
            if tracking:
                self._tracking = tracking

    # ------------------------------------------------------------------
    # Control law (runs on the controller's per-arm control thread)
    # ------------------------------------------------------------------

    def _target_fn(self, side: str, T_current: np.ndarray, gripper_actual: float):
        hand = self._hand[side]
        keys = _KEYS[hand]
        reader = self._reader
        if reader is None or reader.age > _STALE_AFTER:
            self._origin[hand] = None
            return None
        poses, buttons = reader.get()
        T_raw = poses.get(hand)
        if T_raw is None:
            self._origin[hand] = None
            return None
        pressed = lambda key: bool(buttons.get(key, False))

        # Joystick click / B (rising edge): start calibration.
        calib = any(pressed(k) for k in keys["calib"])
        if calib and not self._prev[hand]["calib"] and self._calib[hand] is None:
            self._calib[hand] = {"stage": "x", "start": None}
            self._origin[hand] = None
            print(
                f"\n  [Oculus {hand.upper()}] CALIBRATION 1/2: hold {keys['btn']} and move ~20 cm along the "
                "ROBOT's +x (straight out from its base), then release.",
                flush=True,
            )
        self._prev[hand]["calib"] = calib

        btn = pressed(keys["btn"])
        if self._calib[hand] is not None:
            self._calibrate(hand, btn, T_raw[:3, 3])
            self._prev[hand]["btn"] = btn
            return None  # arm holds still while calibrating
        if btn and not self._prev[hand]["btn"]:
            self._event.set()
        self._prev[hand]["btn"] = btn

        # Gripper (binary, like mesa-env): grip held -> closed (0), else open (1).
        val = buttons.get(keys["grip_val"], (0.0,))
        val = float(val[0]) if isinstance(val, tuple) and val else 0.0
        gripper = 0.0 if pressed(keys["grip"]) or val > _GRIP_THRESHOLD else 1.0
        hold = None if abs(gripper - gripper_actual) < 0.02 else (T_current, gripper)

        if not pressed(keys["clutch"]):
            self._origin[hand] = None
            self._smooth[hand] = None
            return hold
        if self._R[hand] is None:
            self._warn(
                hand, "press the joystick (or B/Y) and calibrate before moving the arm"
            )
            return hold
        # Only follow while the headset has 6-DoF (camera) tracking of the
        # controller; orientation-only means the position is gyro drift.
        track = self._tracking.get(hand)
        if track is not None and track != "POSITION":
            self._warn(
                hand,
                f"tracking is {track}, not POSITION; hold the controller in view of the headset",
            )
            self._origin[hand] = None
            self._smooth[hand] = None
            return hold

        T_vr = self._smoothed(hand, self._to_robot(hand, T_raw))
        if self._origin[hand] is None:
            self._origin[hand] = (T_current.copy(), T_vr.copy())  # clutch engaged here
            return hold
        T_robot0, T_vr0 = self._origin[hand]

        # Relative motion since clutch engage, in the robot base frame.
        d_pos = (T_vr[:3, 3] - T_vr0[:3, 3]) * self._pos_scale
        d_rot = _rotvec(T_vr[:3, :3] @ T_vr0[:3, :3].T) * self._rot_scale
        goal_pos = T_robot0[:3, 3] + d_pos
        goal_rot = _rotmat(d_rot) @ T_robot0[:3, :3]
        r_vec = goal_pos - _SHOULDER
        if np.linalg.norm(r_vec) > _MAX_REACH:
            goal_pos = _SHOULDER + r_vec * (_MAX_REACH / np.linalg.norm(r_vec))

        # Bound how far the goal may lead the arm: limits peak speed and keeps
        # a tracking glitch from flinging the arm.
        offset = goal_pos - T_current[:3, 3]
        if np.linalg.norm(offset) > _MAX_LEAD_POS:
            offset *= _MAX_LEAD_POS / np.linalg.norm(offset)
        rotvec = _rotvec(goal_rot @ T_current[:3, :3].T)
        if np.linalg.norm(rotvec) > _MAX_LEAD_ROT:
            rotvec *= _MAX_LEAD_ROT / np.linalg.norm(rotvec)
        T_target = np.eye(4)
        T_target[:3, 3] = T_current[:3, 3] + offset
        T_target[:3, :3] = _rotmat(rotvec) @ T_current[:3, :3]
        return T_target, gripper

    def _calibrate(self, hand: str, btn: bool, pos: np.ndarray) -> None:
        """mesa-env's calibrate_controller as a non-blocking state machine.

        Stage x: hold the button, move along robot +x, release.  Stage y: same
        for +y.  The dominant headset axis of each move (with sign) becomes that
        robot axis; z = x cross y.  Only the headset frame is used, so how the
        controller is held does not matter.
        """
        c = self._calib[hand]
        name = _KEYS[hand]["btn"]
        if btn and not self._prev[hand]["btn"]:
            c["start"] = pos.copy()
            return
        if btn or c["start"] is None:
            return
        delta = pos - c["start"]
        c["start"] = None
        if np.linalg.norm(delta) < _MIN_CALIB_MOVE:
            print(
                f"  [Oculus {hand.upper()}] moved only {np.linalg.norm(delta) * 100:.0f} cm; hold {name} and move further."
            )
            return
        k = int(np.argmax(np.abs(delta)))
        axis = np.zeros(3)
        axis[k] = np.sign(delta[k])
        if c["stage"] == "x":
            c.update(stage="y", x=axis, kx=k)
            print(
                f"  [Oculus {hand.upper()}] CALIBRATION 2/2: hold {name} and move ~20 cm along the ROBOT's +y "
                "(its left), then release.",
                flush=True,
            )
        elif k == c["kx"]:
            print(
                f"  [Oculus {hand.upper()}] same headset axis as +x; redo the +y move.",
                flush=True,
            )
        else:
            self._R[hand] = np.array([c["x"], axis, np.cross(c["x"], axis)])
            self._calib[hand] = None
            print(
                f"  [Oculus {hand.upper()}] calibrated. Hold the trigger to move the arm.",
                flush=True,
            )

    def _to_robot(self, hand: str, T: np.ndarray) -> np.ndarray:
        R = self._R[hand]
        out = np.eye(4)
        out[:3, :3] = R @ T[:3, :3]
        out[:3, 3] = R @ T[:3, 3]
        return out

    def _smoothed(self, hand: str, T: np.ndarray) -> np.ndarray:
        """Exponential low-pass on position and rotation."""
        prev = self._smooth[hand]
        if prev is None:
            self._smooth[hand] = T.copy()
            return T
        out = np.eye(4)
        out[:3, 3] = prev[:3, 3] + self._alpha * (T[:3, 3] - prev[:3, 3])
        out[:3, :3] = (
            _rotmat(self._alpha * _rotvec(T[:3, :3] @ prev[:3, :3].T)) @ prev[:3, :3]
        )
        self._smooth[hand] = out
        return out

    def _warn(self, hand: str, msg: str) -> None:
        now = time.monotonic()
        if now - self._warned[hand] > 2.0:
            self._warned[hand] = now
            print(f"\n  [Oculus {hand.upper()}] clutch ignored: {msg}", flush=True)

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def poll(self, robot_controller) -> bool:
        if self._event.is_set():
            self._event.clear()
            return True
        trigger = getattr(self, "_pedal_trigger", None)
        if trigger is not None and trigger.is_set():
            trigger.clear()
            return True
        return False

    def poll_success(self, robot_controller) -> bool:
        ev = getattr(self, "_pedal_success", None)
        if ev is not None and ev.is_set():
            ev.clear()
            return True
        return False

    def poll_failure(self, robot_controller) -> bool:
        ev = getattr(self, "_pedal_failure", None)
        if ev is not None and ev.is_set():
            ev.clear()
            return True
        return False

    @property
    def banner(self) -> str:
        return (
            "\n" + "=" * 60 + "\n"
            "  OCULUS TELEOPERATION ACTIVE\n" + "=" * 60 + "\n\n"
            "  1. Wait for 'tracking locked'\n"
            "  2. Press the joystick (or B/Y): hold A/X, move along robot +x, release;\n"
            "     hold A/X, move along robot +y, release\n"
            "  3. HOLD TRIGGER to move the arm; HOLD GRIP to close the gripper\n"
            "  Press Ctrl+C for EMERGENCY STOP (hold 5 s, then go home)\n\n"
            + "=" * 60
            + "\n"
        )


def _rotvec(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_rotvec()


def _rotmat(rotvec: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(rotvec).as_matrix()
