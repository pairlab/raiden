"""Meta Quest controller teleoperation (clutched Cartesian control).

The headset rests on the desk with its cameras facing the operator; only the
hand controllers are used.  Button layout follows pairlab/mesa-env:

* joystick click / B (Y) .. calibrate headset->robot axes: hold A (X), move
                            ~20 cm along robot +x, release; repeat along robot +y.
                            Press it again to cancel.  The result is saved and
                            reused; redo it only when the headset moves.
* hold trigger ............ clutch: the end-effector follows the controller
                            relative to where the trigger was pressed.
* hold grip ............... gripper closed; release to open.
* A (X) ................... trigger event (start/stop recording, record pose);
                            after a recording stops, marks it a success.
* joystick click / B (Y) .. at the success/failure prompt: mark it a failure.

Real recording (``rd record``): teleop starts at the READY prompt with the arm held
at home, so calibration happens there, before A (X) or Enter starts the recording.
During a recording the calibration buttons do nothing.

Sim collection (``auto_reset``, set by ``rd record --sim``) records from every reset:

* B (Y) ................... save the episode as a success, then reset.
* A (X) ................... discard the episode, then reset.
* joystick click .......... discard the episode and calibrate; the scene resets
                            when the calibration is done.

Calibration and the clutch use the controller only while the headset has 6-DoF
(camera) tracking of it; the app keeps streaming a pose for a controller it cannot
see.  The OS tracking state is polled once a second.

Targets are tracked by mink IK on the YAM MuJoCo model (gripper-tip
``grasp_site``) in ``RobotController.start_cartesian_teleop``.
"""

import json
import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from raiden._config import CONFIG_DIR
from raiden.control.base import TeleopInterface
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
_STALL_WARN = 2.0  # s without controller data -> tell the operator
_SMOOTHING_TAU = 0.05  # s low-pass on the controller pose
_MAX_LEAD_POS = 0.15  # m the IK target may lead the arm (bounds peak speed)
_MAX_LEAD_ROT = 0.5  # rad
# m gripper tip from the shoulder (max 0.81); keeps the arm off full extension
_MAX_REACH = 0.74
_SHOULDER = np.array([0.0, 0.0, 0.067])
_DT = 0.01
_CALIB_FILE = CONFIG_DIR / "oculus_calibration.json"
# What to do when the headset has no 6-DoF tracking of a controller (OS TrackingStatus).
_TRACKING_HELP = {
    "ORIENTATION": "the headset cameras cannot see it; hold it in front of the headset",
    "NONE": "it is asleep or off; press any button on it",
}


def _help(state: Optional[str]) -> str:
    return _TRACKING_HELP.get(state or "", "hold it in view of the headset")


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
        self._active_hands: set = set()  # controllers driving an arm; set per episode
        self._pos_scale = pos_scale
        self._rot_scale = float(np.clip(rot_scale, 0.0, 1.0))
        self._alpha = 1.0 - float(np.exp(-_DT / _SMOOTHING_TAU))
        self._reader: Optional[OculusReader] = None
        self._tracking: Dict[str, str] = {}
        self._stop = threading.Event()
        self._tracker: Optional[threading.Thread] = None
        self._event = threading.Event()
        self._success = threading.Event()
        self._failure = threading.Event()
        self._phase: Optional[str] = None  # None | "ready" | "recording" | "verdict"
        self._saved_R = self._load_calibration()
        # Last seen button states, for rising edges.  Kept across episodes: a button
        # still held from the previous episode must not fire in the next one.
        self._prev: Dict[str, Dict[str, bool]] = {
            h: {"calib": False, "face": False, "btn": False} for h in ("l", "r")
        }
        self._reset_state()

    def _reset_state(self) -> None:
        hands = ("l", "r")
        self._R: Dict[str, Optional[np.ndarray]] = {
            h: self._saved_R.get(h) for h in hands
        }  # headset -> robot axes
        self._calib: Dict[str, Optional[dict]] = {h: None for h in hands}
        self._origin: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]] = {
            h: None for h in hands
        }
        self._smooth: Dict[str, Optional[np.ndarray]] = {h: None for h in hands}
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
        if self._saved_R:
            print(
                f"  ✓ Using saved calibration ({_CALIB_FILE}); redo it ({self._calib_buttons}) only if the headset moved"
            )
        # Tracking state is session-level: one poller for the whole session, not per episode.
        self._stop.clear()
        self._tracker = threading.Thread(
            target=self._tracking_loop, name="oculus-tracking", daemon=True
        )
        self._tracker.start()

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
        self._stop.set()
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
        """Teleop starts from the home pose; nothing to do."""

    def start(self, robot_controller) -> None:
        self._begin(robot_controller, phase=None)

    def start_ready(self, robot_controller) -> bool:
        # The recorder's READY prompt: the arm holds at home until the recording starts, so
        # calibration happens here instead of inside a recording.
        self._begin(robot_controller, phase="ready")
        return True

    def _begin(self, robot_controller, phase: Optional[str]) -> None:
        self._reset_state()
        self._phase = phase
        arms = (
            ("left", robot_controller.follower_l),
            ("right", robot_controller.follower_r),
        )
        self._active_hands = {self._hand[side] for side, arm in arms if arm is not None}
        # Presses made while no episode ran (e.g. during a sim reset) belong to no episode.
        self._clear_events()
        robot_controller.start_cartesian_teleop(self._target_fn, dt=_DT)

    def stop(self, robot_controller) -> None:
        self._phase = None
        robot_controller.stop_cartesian_teleop()

    def set_active_recording(self, robot_controller=None) -> None:
        super().set_active_recording(robot_controller)
        self._phase = "recording" if robot_controller is not None else "verdict"
        # A press from the previous phase must not stop this recording or decide its verdict.
        self._clear_events()

    def _clear_events(self) -> None:
        for ev in (self._event, self._success, self._failure):
            ev.clear()

    @property
    def calibrating(self) -> bool:
        return any(c is not None for c in self._calib.values())

    def _tracking_loop(self) -> None:
        """Poll the OS tracking state ~1 Hz; it gates the clutch and calibration (see _target_fn).

        Also reports a stalled data stream, which otherwise looks like a frozen arm.
        """
        stalled = False
        while not self._stop.wait(1.0):
            reader = self._reader
            if reader is None:
                continue
            if (reader.age > _STALL_WARN) != stalled:
                stalled = not stalled
                msg = (
                    "no data from the Quest app; the arm holds until it is back"
                    if stalled
                    else "Quest data back"
                )
                print(f"\n  [Oculus] {msg}", flush=True)
            tracking = reader.controller_tracking()
            for hand, state in tracking.items():
                if hand not in self._active_hands or state == self._tracking.get(hand):
                    continue
                if state == "POSITION":
                    print(f"\n  [Oculus {hand.upper()}] tracking locked", flush=True)
                else:
                    print(
                        f"\n  [Oculus {hand.upper()}] tracking {state}: {_help(state)}. "
                        "The arm holds until it locks.",
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
        track = self._tracking.get(hand)  # None until the first OS reading: trusted

        # Joystick click / B (rising edge): failure verdict after a recording stops,
        # otherwise start or cancel calibration.  In sim collection B saves the
        # episode instead, so only the joystick calibrates.
        stick, face = (pressed(k) for k in keys["calib"])
        calib = stick or (face and not self.auto_reset)
        if calib and not self._prev[hand]["calib"]:
            self._on_calib_press(hand)
        self._prev[hand]["calib"] = calib
        if self.auto_reset and face and not self._prev[hand]["face"]:
            if self._calib[hand] is None:
                self._success.set()
        self._prev[hand]["face"] = face

        btn = pressed(keys["btn"])
        if self._calib[hand] is not None:
            self._calibrate(hand, btn, T_raw[:3, 3], track)
            self._prev[hand]["btn"] = btn
            return None  # arm holds still while calibrating
        if btn and not self._prev[hand]["btn"]:
            # Sim collection: A discards the episode and resets the scene.
            (self._failure if self.auto_reset else self._event).set()
        self._prev[hand]["btn"] = btn
        if self._phase == "ready":
            return None  # every recording starts from home: hold until it starts

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
                hand, f"press {self._calib_buttons} and calibrate before moving the arm"
            )
            return hold
        # Only follow while the headset has 6-DoF (camera) tracking of the
        # controller; orientation-only means the position is gyro drift.
        if track is not None and track != "POSITION":
            self._warn(hand, f"tracking is {track}: {_help(track)}")
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

    def _on_calib_press(self, hand: str) -> None:
        H = hand.upper()
        if self._phase == "verdict":
            self._failure.set()
        elif self._calib[hand] is not None:
            self._calib[hand] = None
            kept = (
                "; the previous calibration stays" if self._R[hand] is not None else ""
            )
            print(f"\n  [Oculus {H}] calibration cancelled{kept}.", flush=True)
        elif self._phase == "recording" and not self.auto_reset:
            print(
                f"\n  [Oculus {H}] calibrate at the READY prompt, before the next recording.",
                flush=True,
            )
        else:
            self._calib[hand] = {"stage": "x", "start": None}
            self._origin[hand] = None
            print(
                f"\n  [Oculus {H}] CALIBRATION 1/2: hold {_KEYS[hand]['btn']} and move ~20 cm along the "
                f"ROBOT's +x (straight out from its base), then release. Press {self._calib_buttons} "
                "again to cancel.",
                flush=True,
            )

    def _calibrate(
        self, hand: str, btn: bool, pos: np.ndarray, track: Optional[str]
    ) -> None:
        """mesa-env's calibrate_controller as a non-blocking state machine.

        Stage x: hold the button, move along robot +x, release.  Stage y: same
        for +y.  The dominant headset axis of each move (with sign) becomes that
        robot axis; z = x cross y.  Only the headset frame is used, so how the
        controller is held does not matter.  A move counts only if the headset
        tracked the controller (6-DoF) from press to release.
        """
        c = self._calib[hand]
        H = hand.upper()
        name = _KEYS[hand]["btn"]
        tracked = track is None or track == "POSITION"
        if btn and not self._prev[hand]["btn"]:
            if tracked:
                c.update(start=pos.copy(), lost=False)
            else:
                print(
                    f"  [Oculus {H}] tracking is {track}: {_help(track)}. Then hold {name} and move again.",
                    flush=True,
                )
            return
        if c["start"] is None:
            return
        c["lost"] = c["lost"] or not tracked
        if btn:
            return
        delta = pos - c["start"]
        c["start"] = None
        if c["lost"]:
            print(
                f"  [Oculus {H}] tracking dropped during the move; hold {name} and redo it.",
                flush=True,
            )
            return
        if np.linalg.norm(delta) < _MIN_CALIB_MOVE:
            print(
                f"  [Oculus {H}] moved only {np.linalg.norm(delta) * 100:.0f} cm; hold {name} and move further."
            )
            return
        k = int(np.argmax(np.abs(delta)))
        axis = np.zeros(3)
        axis[k] = np.sign(delta[k])
        if c["stage"] == "x":
            c.update(stage="y", x=axis, kx=k)
            print(
                f"  [Oculus {H}] CALIBRATION 2/2: hold {name} and move ~20 cm along the ROBOT's +y "
                "(its left), then release.",
                flush=True,
            )
        elif k == c["kx"]:
            print(
                f"  [Oculus {H}] same headset axis as +x; redo the +y move.",
                flush=True,
            )
        else:
            self._R[hand] = np.array([c["x"], axis, np.cross(c["x"], axis)])
            self._calib[hand] = None
            self._save_calibration(hand)
            print(
                f"  [Oculus {H}] calibrated. Hold the trigger to move the arm.",
                flush=True,
            )

    @staticmethod
    def _load_calibration() -> Dict[str, np.ndarray]:
        try:
            with open(_CALIB_FILE) as f:
                return {
                    h: np.array(R, dtype=np.float64) for h, R in json.load(f).items()
                }
        except (OSError, ValueError):
            return {}

    def _save_calibration(self, hand: str) -> None:
        self._saved_R[hand] = self._R[hand]
        _CALIB_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_CALIB_FILE, "w") as f:
            json.dump({h: R.tolist() for h, R in self._saved_R.items()}, f)

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
        if self._success.is_set():
            self._success.clear()
            return True
        if self._phase == "verdict" and self._event.is_set():
            self._event.clear()
            self._phase = None
            return True
        ev = getattr(self, "_pedal_success", None)
        if ev is not None and ev.is_set():
            ev.clear()
            return True
        return False

    def poll_failure(self, robot_controller) -> bool:
        if self._failure.is_set():
            self._failure.clear()
            self._phase = None
            return True
        ev = getattr(self, "_pedal_failure", None)
        if ev is not None and ev.is_set():
            ev.clear()
            return True
        return False

    @property
    def verdict_hint(self) -> str:
        if self.auto_reset:
            return "    B/Y → save as success   A/X → discard   joystick click → discard and calibrate"
        return "    A/X → success   joystick or B/Y → failure"

    @property
    def ready_hint(self) -> str:
        lines = [
            f"  Quest: A/X also starts and stops. Calibrate with {self._calib_buttons},"
            " only if the headset moved."
        ]
        for hand in sorted(self._active_hands):
            state = self._tracking.get(hand)
            if state == "POSITION":
                status = "tracked"
            elif state:
                status = f"tracking {state}: {_help(state)}"
            else:
                status = "tracking unknown"
            calib = "calibrated" if self._R[hand] is not None else "NOT calibrated"
            lines.append(f"  {hand.upper()} controller: {status}; {calib}")
        return "\n".join(lines)

    @property
    def _calib_buttons(self) -> str:
        return "the joystick" if self.auto_reset else "the joystick (or B/Y)"

    @property
    def banner(self) -> str:
        return (
            "\n" + "=" * 60 + "\n"
            "  OCULUS TELEOPERATION ACTIVE\n" + "=" * 60 + "\n\n"
            "  1. Wait for 'tracking locked'\n"
            "  2. Only if the headset moved (the calibration is saved): press the joystick\n"
            "     (or B/Y); hold A/X, move along robot +x, release; hold A/X, move along\n"
            "     robot +y, release\n"
            "  3. HOLD TRIGGER to move the arm; HOLD GRIP to close the gripper\n"
            "  A/X stops the recording; then A/X = success, joystick/B/Y = failure\n"
            "  Press Ctrl+C for EMERGENCY STOP (hold 5 s, then go home)\n\n"
            + "=" * 60
            + "\n"
        )


def _rotvec(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_rotvec()


def _rotmat(rotvec: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(rotvec).as_matrix()
