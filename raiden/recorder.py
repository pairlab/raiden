"""Demonstration recorder – cameras at 30 fps, robot joints at ~100 Hz.

Thread layout during a recording session::

    teleop-right     : follower_r mirrors leader_r at 100 Hz  (background, per episode)
    teleop-left      : follower_l mirrors leader_l at 100 Hz  (background, per episode)
    camera-<name>    : camera.grab() loop per camera          (active while recording)
    robot-recorder   : reads all joint observations at 100 Hz (active while recording)

Cameras are opened once at session start and stay open across episodes.
Teleop threads are started/stopped per episode.
Camera and robot threads are started/stopped per recording episode.

Output layout::

    data/raw/<task_name>/<episode_idx>/
        metadata.json
        robot_data.npz
        cameras/
            scene_camera.svo2
            left_wrist_camera.svo2
            right_wrist_camera.svo2
            ...

After conversion (``rd convert``) each camera directory is expanded into
PNG frames and depth maps.
"""

import json
import re
import select
import shutil
import signal
import sys
import termios
import threading
import time
import tty
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import boto3
import numpy as np

from raiden._config import CALIBRATION_FILE, CAMERA_CONFIG
from raiden.camera_config import CameraConfig
from raiden.cameras import Camera
from raiden.control import TeleopInterface
from raiden.db.database import get_db
from raiden.robot.controller import RobotController, _binary_gripper
from raiden.utils import fzf_select

_STATE_TIMING_DEFAULTS = {
    "state_timestamp_ns": -1,
    "state_timestamp_valid": False,
    "state_timestamp_source": "unavailable",
    "read_start_ns": -1,
    "read_end_ns": -1,
    "read_start_monotonic_ns": -1,
    "read_end_monotonic_ns": -1,
}
_COMMAND_TIMING_DEFAULTS = {
    "command_timestamp_ns": -1,
    "command_timestamp_valid": False,
    "command_timestamp_source": "unavailable",
    "command_end_ns": -1,
    "command_start_monotonic_ns": -1,
    "command_end_monotonic_ns": -1,
}

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@dataclass
class RecordingMetadata:
    task_name: str
    task_instruction: str
    timestamp: str
    duration_s: float
    robot_frames: int
    robot_hz: float
    cameras: List[str]
    camera_fps: int
    control: str = "leader"
    complete: bool = False
    converted: bool = False


# ---------------------------------------------------------------------------
# DemonstrationRecorder
# ---------------------------------------------------------------------------


class DemonstrationRecorder:
    """Manages one recording episode.

    Cameras are opened before this object is created and remain open across
    episodes (opening/closing ZED cameras is slow).  start_recording() /
    stop_recording() toggle the SVO2 writers and the robot data thread.

    stop_recording() does NOT close cameras or shut down robots — the caller
    (run_recording) is responsible for that at session end.
    """

    def __init__(
        self,
        cameras: List[Camera],
        robot_controller: RobotController,
        recording_dir: Path,
        task_name: str,
        task_instruction: str,
        interface: TeleopInterface,
        extra_metadata: Optional[Dict] = None,
        monitor=None,
    ):
        self.cameras = cameras
        self.monitor = monitor
        self.extra_metadata = dict(extra_metadata or {})
        self.robot_controller = robot_controller
        self.recording_dir = recording_dir
        self.task_name = task_name
        self.task_instruction = task_instruction
        self.interface = interface

        self.cameras_dir = recording_dir / "cameras"

        self.is_recording = False
        self._stop_event = threading.Event()
        self._threads: List[threading.Thread] = []

        # Robot data accumulated during one episode
        self._robot_frames: List[Dict] = []
        self._start_time: float = 0.0
        self._errors: List[str] = []
        self._error_lock = threading.Lock()
        self._camera_start_times_ns: Dict[str, int] = {}
        self._camera_frames: Dict[str, int] = {}
        self._camera_capture_receipts: Dict = {}

        config = dict(self.extra_metadata.get("recording_config", {}))
        config["schema_version"] = 1
        config["robot"] = robot_controller.recording_config()
        self.extra_metadata["recording_config"] = config
        self._required_followers = config["robot"]["active_followers"]
        self.extra_metadata["robot_timing"] = {
            "schema_version": 1,
            "wall_clock": "Unix epoch nanoseconds from host time.time_ns()",
            "monotonic_clock": "host time.monotonic_ns(); differences only, not epoch time",
            "timestamps": "Legacy reference-camera clock label before sequential robot reads",
            "readout_timestamp_ns/readout_end_ns": (
                "Host wall-clock row start/completion brackets around all observations and command snapshots"
            ),
            "state_timestamp_ns": (
                "Per-arm cached state timestamp, valid only with state_timestamp_valid. "
                "host_cached_feedback_read labels a host read of cached CAN feedback; "
                "derived from driver float seconds, not motor acquisition or feedback receipt time."
            ),
            "read_start_ns/read_end_ns": "Per-arm observation call brackets, including snapshot copy",
            "command_timestamp_ns/command_end_ns": (
                "Host teleop command API issue/return brackets, atomic with joint_cmd intent; "
                "not CAN delivery or motor execution. Pause hold calls are included. "
                "Home/reset calls outside teleop are not command events."
            ),
            "unknown": "-1 timestamp, False validity, unavailable source; no acquisition time inferred",
            "command_sampling": (
                "Latest completed teleop API call at each recorder read; intermediate commands "
                "may be unsampled. Initial measured-pose fallback has unknown command timing."
            ),
            "joint_cmd_api": "Per-arm teleop API input target; NaN when unknown; joint_cmd retains gripper intent",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_recording(
        self, hint: str = "  Press the button again to stop recording"
    ) -> None:
        if self.is_recording:
            return

        self.recording_dir.mkdir(parents=True, exist_ok=True)
        self.cameras_dir.mkdir(parents=True, exist_ok=True)

        self.is_recording = True
        self._start_time = time.monotonic()
        self._robot_frames = []
        self._stop_event.clear()
        self._threads = []
        self._errors = []
        self._camera_start_times_ns = {}
        self._camera_frames = {camera.name: 0 for camera in self.cameras}
        self._camera_capture_receipts = {}
        # A process crash or failed start must leave a recoverable attempt, too.
        self.extra_metadata["preserve_incomplete"] = True
        self._save_metadata(0.0, complete=False)

        names = [camera.name for camera in self.cameras]
        required = self.extra_metadata["recording_config"].get("cameras", {})
        missing = set(required) - set(names)
        if not names or len(set(names)) != len(names) or missing:
            self._record_failure(
                "camera configuration",
                RuntimeError(f"Required recording cameras unavailable: {sorted(missing)}; opened: {names}"),
            )
            self.is_recording = False
            self._save_metadata(0.0, complete=False)
            self.raise_if_failed()

        # Start all cameras in parallel so they begin recording simultaneously.
        # Sequential starts would introduce a per-camera startup offset (e.g.
        # ~50 frames for RealSense due to pipeline restart) that misaligns ZED
        # and bag frames.
        def _start_one(camera) -> None:
            try:
                path = self.cameras_dir / f"{camera.name}.{camera.recording_extension}"
                t0 = time.time_ns()
                camera.start_recording(path)
                self._camera_start_times_ns[camera.name] = t0
            except Exception as error:
                self._record_failure(f"camera {camera.name} start", error)

        start_threads = [
            threading.Thread(target=_start_one, args=(cam,), daemon=True)
            for cam in self.cameras
        ]
        for t in start_threads:
            t.start()
        for t in start_threads:
            t.join()

        if self._errors:
            # A failed SDK call may already have opened a writer. Stop every
            # attempted camera, including the one whose start raised.
            self._finalize_cameras()
            self.is_recording = False
            self._save_metadata(time.monotonic() - self._start_time, complete=False)
            self.raise_if_failed()

        # One camera grab thread per camera (naturally rate-limited by SDK)
        for camera in self.cameras:
            t = threading.Thread(
                target=self._camera_loop,
                args=(camera, self._stop_event),
                name=f"camera-{camera.name}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        # Robot data recording thread — uses the first camera as a shared clock
        # reference so that robot timestamps are on the same clock as camera
        # frame timestamps (enabling direct interpolation at conversion time).
        t = threading.Thread(
            target=self._robot_loop,
            args=(self._stop_event, self.cameras[0]),
            name="robot-recorder",
            daemon=True,
        )
        t.start()
        self._threads.append(t)

        print("\n" + "!" * 60)
        print("  RECORDING STARTED")
        print("!" * 60)
        print(hint + "\n")

    def stop_recording(
        self, complete: bool = True, shutdown_robots: bool = True,
        finalize: Callable[[Path], None] | None = None,
    ) -> Path:
        """Stop the current recording episode and persist data.

        Shuts down the robot controller (returns home + closes motor connections)
        and stops camera recording, but does NOT close cameras — they stay open
        so the next episode can start without re-initialising the camera SDK.
        The caller is responsible for calling camera.close() at session end.

        Args:
            complete: Set to True when the episode was cleanly stopped by the
                      user.  Set to False on crash / Ctrl-C so the directory
                      is retained as incomplete for recovery.
            shutdown_robots: Set to False to keep the robots up (sim collection,
                      where the next episode starts from a scene reset).
            finalize: Save additional required artifacts before marking complete.
                      A failed finalizer retains an incomplete episode for recovery.
        """
        if not self.is_recording:
            return self.recording_dir

        self.is_recording = False
        self._stop_event.set()
        duration = time.monotonic() - self._start_time

        # Stop the grab and robot-sampling threads first, so nothing reads the robot
        # while it shuts down.
        stop_error = None
        try:
            self._join_threads()
        except BaseException as error:
            stop_error = error
            self._record_failure("recording workers", error)
        finally:
            if shutdown_robots:
                # Home and close before finalising camera files, whose SDK stalls
                # can exceed the motors' 400 ms CAN watchdog. Always attempt this
                # even if joining the recording workers was interrupted.
                try:
                    self.robot_controller.shutdown()
                except BaseException as error:
                    self._record_failure("robot shutdown", error)
                    if not isinstance(error, Exception) and stop_error is None:
                        stop_error = error

        self._finalize_cameras()

        print("\n" + "!" * 60)
        print("  RECORDING STOPPED")
        print(f"  Duration : {duration:.2f} s")
        print(f"  Robot frames : {len(self._robot_frames)}")
        print("!" * 60 + "\n")

        self._save_metadata(duration, complete=False)
        try:
            self._save_robot_data()
            self._validate_cameras()
        except Exception as error:
            self._record_failure("recording integrity", error)
        if finalize is not None:
            try:
                finalize(self.recording_dir)
            except Exception as error:
                self.extra_metadata.update(
                    preserve_incomplete=True,
                    finalization_error=f"{type(error).__name__}: {error}",
                )
                self._save_metadata(duration, complete=False)
                if stop_error is not None:
                    stop_error.add_note(f"Episode finalization also failed: {error!r}")
                    raise stop_error from error
                raise
        self._save_metadata(duration, complete=complete and not self._errors)
        if stop_error is not None:
            raise stop_error
        if complete:
            self.raise_if_failed()

        return self.recording_dir

    def discard(self) -> None:
        """Stop the episode without saving it and delete its directory; robots stay up."""
        if not self.is_recording:
            return
        # A collection failure must not be mistaken for an operator's discard.
        if self._errors:
            self.stop_recording(complete=False, shutdown_robots=False)
            self.raise_if_failed()
            return
        self.is_recording = False
        self._stop_event.set()
        self._stop_streams()
        if self._errors:
            try:
                self._save_robot_data()
            except Exception as error:
                self._record_failure("recording integrity", error)
            self._save_metadata(time.monotonic() - self._start_time, complete=False)
            self.raise_if_failed()
        shutil.rmtree(self.recording_dir, ignore_errors=True)
        print("\n  Episode discarded\n")

    def _stop_streams(self) -> None:
        self._join_threads()
        self._finalize_cameras()

    def _join_threads(self) -> None:
        # Wait for threads (camera grab threads will exit within one frame period)
        for t in self._threads:
            t.join(timeout=3.0)
            if t.is_alive():
                self._record_failure(t.name, RuntimeError("recording worker did not stop"))

    def _finalize_cameras(self) -> None:
        # Finalise SVO2 / bag files but keep cameras open — run in parallel so
        # all cameras stop at the same time (avoids trailing frames on one camera).
        # Read controls before stop_recording restarts the RealSense pipeline.
        self._camera_capture_receipts = {}
        for camera in self.cameras:
            if hasattr(camera, "color_controls") and hasattr(camera, "get_camera_info"):
                try:
                    self._camera_capture_receipts[camera.name] = camera.get_camera_info()
                except Exception as error:
                    self._camera_capture_receipts[camera.name] = {"readback_error": str(error)}

        def _stop_one(camera) -> None:
            try:
                camera.stop_recording()
            except Exception as error:
                self._record_failure(f"camera {camera.name} stop", error)

        stop_threads = [
            threading.Thread(target=_stop_one, args=(camera,), daemon=True)
            for camera in self.cameras
        ]
        for t in stop_threads:
            t.start()
        for t in stop_threads:
            t.join()

    def _record_failure(self, source: str, error: BaseException) -> None:
        message = f"{source}: {type(error).__name__}: {error}"
        with self._error_lock:
            self._errors.append(message)
        self._stop_event.set()
        print(f"  Recording failed: {message}")

    def raise_if_failed(self) -> None:
        """Surface background failures to the session's foreground wait loop."""
        with self._error_lock:
            errors = list(self._errors)
        if errors:
            raise RuntimeError("Recording failed: " + "; ".join(errors))

    def _validate_cameras(self) -> None:
        for camera in self.cameras:
            if camera.name not in self._camera_start_times_ns:
                raise RuntimeError(f"Camera {camera.name} never started recording")
            if not self._camera_frames.get(camera.name, 0):
                raise RuntimeError(f"Camera {camera.name} recorded no frames")
            path = self.cameras_dir / f"{camera.name}.{camera.recording_extension}"
            if not path.exists() or (path.is_file() and not path.stat().st_size):
                raise RuntimeError(f"Camera {camera.name} recording is missing or empty: {path}")

    # ------------------------------------------------------------------
    # Background threads
    # ------------------------------------------------------------------

    def _camera_loop(self, camera: Camera, stop_event: threading.Event) -> None:
        """Grab loop – camera SDK rate-limits to its own FPS (30 Hz)."""
        try:
            while not stop_event.is_set():
                if not camera.grab():
                    raise RuntimeError("grab returned False (camera stream lost)")
                self._camera_frames[camera.name] += 1
                if self.monitor is not None:
                    # Encoding happens on the monitor thread.
                    self.monitor.log(camera.name, camera.get_frame().color)
        except Exception as error:
            self._record_failure(f"camera {camera.name} sampling", error)

    def _robot_loop(self, stop_event: threading.Event, ref_camera) -> None:
        """Read joint observations at ~100 Hz and buffer them.

        Legacy row timestamps retain the reference camera's current clock
        (host wall time for RealSense). Separate host labels, per-arm read
        brackets, cached state times and command API times preserve their
        actual provenance; none claims simultaneous motor acquisition.
        """
        target_dt = 0.01  # 100 Hz
        while not stop_event.is_set():
            loop_start = time.monotonic()

            try:
                ts_ns = ref_camera.get_current_timestamp_ns()
                readout_ns = time.time_ns()
                readout_monotonic_ns = time.monotonic_ns()
                timed_observations = getattr(self.robot_controller, "get_all_timed_observations", None)
                if callable(timed_observations):
                    obs, obs_timing = timed_observations()
                else:
                    obs = self.robot_controller.get_all_observations()
                    obs_timing = {}
                command_snapshots = getattr(self.robot_controller, "get_last_commanded_snapshots", None)
                if callable(command_snapshots):
                    cmd_timing = command_snapshots()
                    cmd = {name: snapshot["positions"] for name, snapshot in cmd_timing.items()}
                else:
                    cmd = self.robot_controller.get_last_commanded_positions()
                    cmd_timing = {}
                readout_end_ns = time.time_ns()
                readout_end_monotonic_ns = time.monotonic_ns()
                self._validate_robot_frame(obs, cmd)
                if stop_event.is_set():
                    return
                self._robot_frames.append({
                    "t": ts_ns, "obs": obs, "cmd": cmd,
                    "readout_timestamp_ns": readout_ns,
                    "readout_monotonic_ns": readout_monotonic_ns,
                    "readout_end_ns": readout_end_ns,
                    "readout_end_monotonic_ns": readout_end_monotonic_ns,
                    "obs_timing": obs_timing, "cmd_timing": cmd_timing,
                })
                if self.monitor is not None:
                    # Parks the array for the grasp guides; FK happens on the monitor thread.
                    for arm in ("follower_l", "follower_r"):
                        if arm in obs and "joint_pos" in obs[arm]:
                            self.monitor.log_joints(arm, obs[arm]["joint_pos"])
            except Exception as error:
                self._record_failure("robot sampling", error)
                return

            elapsed = time.monotonic() - loop_start
            sleep_time = target_dt - elapsed
            if sleep_time > 0:
                stop_event.wait(sleep_time)

    def _validate_robot_frame(self, obs: Dict, cmd: Dict) -> None:
        for arm in self._required_followers:
            if arm not in obs:
                raise RuntimeError(f"Missing observations for configured {arm}")
            for key, shape in (("joint_pos", (6,)), ("gripper_pos", (1,))):
                values = np.asarray(obs[arm].get(key), dtype=float)
                if key == "gripper_pos" and values.shape == ():
                    values = values.reshape(1)
                if values.shape != shape or not np.isfinite(values).all():
                    raise RuntimeError(f"Invalid {arm} {key}: expected finite shape {shape}, got {values.shape}")
            command = cmd.get(arm)
            # A short initial interval before the teleop's first command is valid;
            # _save_robot_data still requires an actual command for every arm.
            if command is not None:
                values = np.asarray(command, dtype=float)
                if values.shape != (7,) or not np.isfinite(values).all():
                    raise RuntimeError(f"Invalid {arm} command: expected finite shape (7,), got {values.shape}")

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_robot_data(self) -> None:
        """Flatten the robot frame list into per-key numpy arrays and save .npz."""
        output_file = self.recording_dir / "robot_data.npz"
        n = len(self._robot_frames)

        if n == 0:
            np.savez_compressed(output_file, timestamps=np.empty(0, dtype=np.int64))
            raise RuntimeError("No robot frames recorded")

        data: Dict[str, np.ndarray] = {}
        data["timestamps"] = np.array(
            [f["t"] for f in self._robot_frames], dtype=np.int64
        )
        for key in ("readout_timestamp_ns", "readout_monotonic_ns",
                    "readout_end_ns", "readout_end_monotonic_ns"):
            data[key] = np.array([f.get(key, -1) for f in self._robot_frames], dtype=np.int64)

        # Collect all robot names present in the first frame
        robot_names = list(self._robot_frames[0]["obs"].keys())

        for robot_name in robot_names:
            obs_keys = list(self._robot_frames[0]["obs"][robot_name].keys())
            for key in obs_keys:
                arr = np.stack([f["obs"][robot_name][key] for f in self._robot_frames])
                data[f"{robot_name}_{key}"] = arr
            timing_sources = [("obs_timing", _STATE_TIMING_DEFAULTS)]
            if robot_name in ("follower_l", "follower_r"):
                timing_sources.append(("cmd_timing", _COMMAND_TIMING_DEFAULTS))
                data[f"{robot_name}_joint_cmd_api"] = np.array([
                    frame.get("cmd_timing", {}).get(robot_name, {}).get("api_positions", np.full(7, np.nan))
                    for frame in self._robot_frames
                ], dtype=np.float32)
            for source, defaults in timing_sources:
                for key, default in defaults.items():
                    if isinstance(default, bool):
                        dtype = np.bool_
                    elif isinstance(default, str):
                        dtype = np.str_
                    else:
                        dtype = np.int64
                    data[f"{robot_name}_{key}"] = np.array([
                        frame.get(source, {}).get(robot_name, {}).get(key, default)
                        for frame in self._robot_frames
                    ], dtype=dtype)

        # Retain measured streams even if a required arm or its command stream
        # fails the checks below. A failed attempt remains useful for diagnosis.
        np.savez_compressed(output_file, **data)
        for frame in self._robot_frames:
            self._validate_robot_frame(frame["obs"], frame["cmd"])

        # Build combined 7-DOF arrays for follower arms.
        # follower_<arm>_joint_pos_7d  — executed: arm joints (6) + gripper (1)
        # follower_<arm>_joint_cmd     — commanded: 7-DOF sent to command_joint_pos
        command_errors = []
        for arm_key in ("follower_r", "follower_l"):
            jp_key = f"{arm_key}_joint_pos"
            gp_key = f"{arm_key}_gripper_pos"
            if jp_key in data and gp_key in data:
                grip = data[gp_key].reshape(n, 1).astype(np.float32)
                data[f"{arm_key}_joint_pos_7d"] = np.concatenate(
                    [data[jp_key].astype(np.float32), grip], axis=1
                )

            # Build cmd array, falling back to actual joint pos (gripper binarized)
            # for frames where no command has been issued yet (e.g. the first few
            # frames before the first teleop command is processed).
            pos7d_key = f"{arm_key}_joint_pos_7d"
            raw_cmds = [f["cmd"].get(arm_key) for f in self._robot_frames]
            if jp_key in data:
                # This arm was active — cmd data is required.
                if not any(c is not None for c in raw_cmds):
                    command_errors.append(
                        f"No commanded positions recorded for {arm_key}. "
                        "The teleop loop never issued a command to this arm."
                    )
                    continue
                fallback = None
                filled: list = []
                for i, c in enumerate(raw_cmds):
                    if c is not None:
                        fallback = c
                    if fallback is not None:
                        filled.append(fallback)
                    elif pos7d_key in data:
                        pos = data[pos7d_key][i].copy()
                        pos[6] = _binary_gripper(pos[6])
                        filled.append(pos)
                if len(filled) != n:
                    command_errors.append(
                        f"Could not build complete cmd array for {arm_key}: "
                        f"got {len(filled)}/{n} frames."
                    )
                    continue
                data[f"{arm_key}_joint_cmd"] = np.stack(filled).astype(np.float32)

        np.savez_compressed(output_file, **data)
        if command_errors:
            raise RuntimeError("; ".join(command_errors))
        print(f"  ✓ Robot data saved  ({n} frames) → {output_file}")

    def _save_metadata(self, duration: float, complete: bool = True) -> None:
        output_file = self.recording_dir / "metadata.json"
        n = len(self._robot_frames)
        hz = n / duration if duration > 0 else 0.0

        meta = RecordingMetadata(
            task_name=self.task_name,
            task_instruction=self.task_instruction,
            timestamp=datetime.now().isoformat(),
            duration_s=round(duration, 3),
            robot_frames=n,
            robot_hz=round(hz, 1),
            cameras=[c.name for c in self.cameras],
            camera_fps=30,
            control=self.interface.name,
            complete=complete,
            converted=False,
        )

        meta_dict = asdict(meta)

        if self._camera_start_times_ns:
            meta_dict["camera_start_times_ns"] = self._camera_start_times_ns

        rs_offsets = {
            cam.name: cam._clock_offset_ns
            for cam in self.cameras
            if getattr(cam, "_clock_offset_ns", None) is not None
        }
        if rs_offsets:
            meta_dict["realsense_clock_offsets"] = rs_offsets

        color_controls = {
            cam.name: cam.color_controls or "auto"
            for cam in self.cameras
            if hasattr(cam, "color_controls")
        }
        if color_controls:
            meta_dict["color_controls"] = color_controls
        if getattr(self, "_camera_capture_receipts", None):
            meta_dict["camera_capture_receipts"] = self._camera_capture_receipts
        meta_dict.update(self.extra_metadata)
        # Caller-supplied metadata cannot override the recording integrity flag.
        meta_dict["complete"] = bool(complete and not self._errors)
        if self._errors:
            meta_dict["recording_errors"] = list(self._errors)
        if meta_dict["complete"]:
            meta_dict.pop("preserve_incomplete", None)

        temporary = output_file.with_suffix(".json.tmp")
        with open(temporary, "w") as f:
            json.dump(meta_dict, f, indent=2)
        temporary.replace(output_file)

        print(f"  ✓ Metadata saved → {output_file}")


# ---------------------------------------------------------------------------
# Camera factory
# ---------------------------------------------------------------------------


def load_cameras_from_config(
    config_file: str = CAMERA_CONFIG,
) -> List[Camera]:
    """Create Camera instances for every entry in camera.json."""
    cfg = CameraConfig(config_file)
    names = cfg.list_camera_names()
    if not names:
        raise RuntimeError(f"No recording cameras configured in {config_file}")

    results: dict = {}
    errors: dict = {}

    def _open(name: str) -> None:
        cam = None
        try:
            cam = cfg.create_camera(name)
            cam.open()
            cam_type = cfg.get_camera_type(name)
            serial = cfg.get_serial_by_name(name)
            print(f"  ✓ Camera '{name}' opened ({cam_type}, serial: {serial})")
            results[name] = cam
        except Exception as exc:
            errors[name] = exc
            if cam is not None:
                try:
                    cam.close()
                except Exception:
                    pass

    threads = [
        threading.Thread(target=_open, args=(name,), daemon=True) for name in names
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        for camera in results.values():
            try:
                camera.close()
            except Exception:
                pass
        raise RuntimeError(f"Camera initialization failed: {errors}")

    # Return cameras in config order.
    return [results[name] for name in names]


def _camera_recording_config(cfg: CameraConfig, names: List[str]) -> Dict:
    """Portable semantic associations; native camera receipts retain capture details."""
    result = {}
    for name in names:
        result[name] = {
            "role": cfg.get_role(name),
            "type": cfg.get_camera_type(name),
            "policy_crop": list(cfg.get_crop(name)) if cfg.get_crop(name) else None,
        }
        recording_format = cfg.get_recording_format(name)
        if recording_format is not None:
            result[name]["recording_format"] = recording_format
    return result


# ---------------------------------------------------------------------------
# Task selection
# ---------------------------------------------------------------------------


_TASK_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


def validate_task_name(name: str) -> str | None:
    """Return an error message if *name* is invalid, otherwise None.

    Valid names contain only letters, digits, and underscores — no spaces.
    Examples: ``PickUpApple``, ``pick_up_apple``, ``task01``.
    """
    if not name:
        return "Task name cannot be empty."
    if not _TASK_NAME_RE.match(name):
        return "Task name must contain only letters, digits, and underscores (no spaces). E.g. PickUpApple or pick_up_apple."
    return None


def select_task() -> tuple[str, str]:
    """Use fzf to choose (or create) a task for recording."""
    db = get_db()
    tasks = db.get_tasks()

    _NEW = "<< Add new task >>"
    labels = {f"{t['name']}  ({t['instruction']})": t for t in reversed(tasks)}
    chosen = fzf_select(list(labels) + [_NEW], prompt="Record task> ")[0]

    if chosen == _NEW:
        name = input("  New task name: ").strip()
        err = validate_task_name(name)
        if err:
            print(f"Error: {err}")
            sys.exit(1)
        instruction = input("  Task instruction: ").strip()
        if not instruction:
            print("Error: task instruction cannot be empty")
            sys.exit(1)
        task = db.get_task_by_name(name)
        if task is None:
            db.add_task(name, instruction)
            task = db.get_task_by_name(name)
        return task["name"], task["instruction"]

    task = labels[chosen]
    return task["name"], task["instruction"]


def select_teacher() -> int:
    """Use fzf to choose (or create) a teacher; returns the DB teacher id."""
    db = get_db()
    teachers = db.get_teachers()

    _NEW = "<< Add new teacher >>"
    labels = [t["name"] for t in reversed(teachers)] + [_NEW]
    chosen = fzf_select(labels, prompt="Select teacher> ")[0]

    if chosen == _NEW:
        name = input("  New teacher name: ").strip()
        if not name:
            print("Error: teacher name cannot be empty")
            sys.exit(1)
        existing = db.get_teacher_by_name(name)
        if existing:
            return existing["id"]
        return db.add_teacher(name)

    existing = db.get_teacher_by_name(chosen)
    if existing:
        return existing["id"]
    # Shouldn't happen, but recover gracefully
    return db.add_teacher(chosen)


# ---------------------------------------------------------------------------
# Episode directory management
# ---------------------------------------------------------------------------


def _peek_next_episode(task_dir: Path) -> str:
    """The episode name ``_next_recording_dir`` will use, without creating it."""
    existing = sorted(d for d in task_dir.iterdir() if d.is_dir() and d.name.isdigit())
    if existing:
        meta_file = existing[-1] / "metadata.json"
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        if not meta.get("complete", False) and not meta.get("preserve_incomplete", False):
            return existing[-1].name
    return f"{int(existing[-1].name) + 1 if existing else 0:04d}"


def _next_recording_dir(task_dir: Path) -> Path:
    """Return the directory to record the next episode into.

    Reuse an incomplete last recording unless it was explicitly retained after a
    persistence failure. Otherwise create a new numbered directory.
    """
    existing = sorted(d for d in task_dir.iterdir() if d.is_dir() and d.name.isdigit())

    if existing:
        last_dir = existing[-1]
        meta_file = last_dir / "metadata.json"
        is_incomplete = True
        preserve_incomplete = False
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
            is_incomplete = not meta.get("complete", False)
            preserve_incomplete = meta.get("preserve_incomplete", False)

        if is_incomplete and not preserve_incomplete:
            print(f"  Overriding incomplete recording: {last_dir}")
            shutil.rmtree(last_dir)
            last_dir.mkdir()
            return last_dir

    # After the highest number, not the count: deleted episodes leave gaps.
    episode_idx = f"{int(existing[-1].name) + 1 if existing else 0:04d}"
    new_dir = task_dir / episode_idx
    new_dir.mkdir()
    return new_dir


# ---------------------------------------------------------------------------
# S3 upload
# ---------------------------------------------------------------------------


def upload_to_s3(recording_dir: Path, bucket: str, prefix: str) -> None:
    s3 = boto3.client("s3")
    for file_path in sorted(recording_dir.rglob("*")):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(recording_dir.parent)
        key = f"{prefix}/{rel}"
        print(f"  Uploading {file_path.name} → s3://{bucket}/{key}")
        s3.upload_file(str(file_path), bucket, key)

    print(f"\n✓ Upload complete → s3://{bucket}/{prefix}/")


# ---------------------------------------------------------------------------
# Keyboard helper
# ---------------------------------------------------------------------------


def _ui_ready_command(ui, interface: TeleopInterface) -> Optional[bool]:
    """The page's READY command: True to end the session, False to start, None for nothing."""
    if ui is None:
        return None
    ui.poll_interface(interface)
    cmd = ui.take("start", "end")
    if cmd == "end":
        return True
    if cmd == "start":
        if getattr(interface, "calibrating", False):
            ui.notice("finish or cancel the calibration first")
            return None
        return False
    return None


def _wait_for_enter_or_quit(
    robot_controller: RobotController,
    interface: TeleopInterface,
    ui=None,
) -> bool:
    """Wait for Enter/Space (proceed) or 'q' (quit session).

    A start is ignored while the device is calibrating: calibration stays out of recordings.

    Returns:
        True  — 'q' pressed or footpedal e-stop; caller should end the session.
        False — Enter/Space or footpedal left fired; caller should proceed.
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            if robot_controller.session_estop_requested:
                return True
            if interface.poll(robot_controller) and not interface.calibrating:
                return False
            if select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if ch.lower() == "q":
                    return True
                if ch in ("\r", "\n", " "):
                    if not interface.calibrating:
                        return False
                    print("  Finish or cancel the calibration first.")
            ui_cmd = _ui_ready_command(ui, interface)
            if ui_cmd is not None:
                return ui_cmd
            time.sleep(0.05)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _wait_for_verdict(
    robot_controller: RobotController,
    interface: TeleopInterface,
    ui=None,
    recording: Optional[DemonstrationRecorder] = None,
) -> Optional[str]:
    """After recording stops, wait for the user to mark success or failure.

    Inputs accepted:
      - Footpedal middle button → "success"
      - Footpedal right button  → "failure"
      - Leader arm top button   → "success"  (non-spacemouse)
      - Leader arm bottom button→ "failure"  (non-spacemouse)
      - Enter key               → "success"
      - 'f' key                 → "failure"
      - Any other key / timeout → None (leaves status as "pending")

    Returns:
        "success", "failure", or None.
    """
    print("\n" + "-" * 60)
    print("  Mark this demonstration:")
    lines = []
    lines += ["    Middle pedal → success", "    Right pedal  → failure"]
    if interface.supports_verdict_button:
        lines += ["    Top button    → success", "    Bottom button → failure"]
    if interface.verdict_hint:
        lines += [interface.verdict_hint]
    lines += ["    Enter → success   f → failure   other key → skip"]
    print("\n".join(lines))
    print("-" * 60 + "\n")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        deadline = time.monotonic() + 30.0  # 30 s timeout
        if ui is not None:
            ui.set_phase("verdict", verdict_deadline=time.time() + 30.0)
        while time.monotonic() < deadline:
            if recording is not None:
                recording.raise_if_failed()
            if interface.poll_success(robot_controller):
                return "success"
            if interface.poll_failure(robot_controller):
                return "failure"
            if ui is not None:
                ui.poll_interface(interface)
                cmd = ui.take("success", "failure", "skip")
                if cmd is not None:
                    return None if cmd == "skip" else cmd
            if select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if ch in ("\r", "\n"):
                    return "success"
                if ch.lower() == "f":
                    return "failure"
                return None
            if interface.supports_verdict_button:
                verdict = robot_controller.check_verdict_button()
                if verdict is not None:
                    return verdict
            time.sleep(0.05)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return None  # timed out


def _wait_for_start_or_quit(
    robot_controller: RobotController,
    interface: TeleopInterface,
    ui=None,
) -> bool:
    """Wait for a leader button press (start) or the 'q' key (quit session).

    Returns:
        True  — 'q' pressed or footpedal e-stop; caller should end the session.
        False — leader button / footpedal left pressed; caller should start recording.
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            if robot_controller.session_estop_requested:
                return True
            if interface.poll(robot_controller):
                return False
            if select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if ch.lower() == "q":
                    return True
            ui_cmd = _ui_ready_command(ui, interface)
            if ui_cmd is not None:
                return ui_cmd
            time.sleep(0.05)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# ---------------------------------------------------------------------------
# Sim collection
# ---------------------------------------------------------------------------


def _wait_for_sim_action(
    robot_controller: RobotController,
    interface: TeleopInterface,
    task,
    use_stdin: bool,
    recording: Optional[DemonstrationRecorder] = None,
) -> str:
    """Block until the operator saves ("save"), discards ("discard"), calibrates ("calibrate")
    or ends ("quit").

    Prints the sim's subtask completions meanwhile, so the operator sees when the task is done.
    """
    while True:
        if recording is not None:
            recording.raise_if_failed()
        if robot_controller.session_estop_requested:
            return "quit"
        if interface.calibrating:
            return "calibrate"
        if interface.poll_success(robot_controller):
            return "save"
        if interface.poll_failure(robot_controller):
            return "discard"
        if use_stdin and select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1).lower()
            if ch in ("\r", "\n"):
                return "save"
            if ch == "r":
                return "discard"
            if ch == "q":
                return "quit"
        if task.poll():
            for event in task.drain_events():
                print(f"  {event}")
        time.sleep(0.05)


def _run_sim_episodes(
    cameras: List[Camera],
    interface: TeleopInterface,
    sim: str,
    task_dir: Path,
    task_name: str,
    task_instruction: str,
    monitor,
    on_saved: Callable[[Path], None],
    active_ctrl: List[Optional[RobotController]],
    camera_configuration: Optional[Dict] = None,
    arms: str = "single",
) -> Optional[Path]:
    """Sim collection: record from every scene reset; only a success is saved.

    - success (Quest B/Y, middle pedal, Enter) → save the episode, reset, record the next one
    - failure (Quest A/X, right pedal, 'r')   → discard the episode, reset, record the next one
    - calibration (Quest joystick)             → discard the episode; reset once it is done
    - 'q' or a soft pause (left pedal)        → discard the episode and end the session

    Only saved episodes keep a directory, so episode numbers stay contiguous.  The controller
    stays up for the whole session (the sim has no CAN watchdog), so a reset is a snap home
    and a new object layout, not a re-initialisation.  Returns the last saved episode.
    """
    from raiden.sim import ActionLog, TaskMonitor

    robot_controller = RobotController(
        use_right_leader=False,
        use_left_leader=interface.uses_leaders,
        use_right_follower=arms == "bimanual",
        use_left_follower=True,
        sim=sim,
    )
    active_ctrl[0] = robot_controller
    task = TaskMonitor(sim)
    action_log = ActionLog(sim)
    hint = "\n".join(
        line
        for line in (
            interface.verdict_hint,
            "    Middle pedal → save   Right pedal → discard",
            "    Enter → save   r → discard   q → discard and end the session",
        )
        if line
    )

    use_stdin = sys.stdin.isatty()
    old_settings = termios.tcgetattr(sys.stdin.fileno()) if use_stdin else None
    recorder: Optional[DemonstrationRecorder] = None
    last_saved: Optional[Path] = None
    n_saved = 0
    try:
        robot_controller.setup_for_teleop_recording()
        interface.setup(robot_controller)
        robot_controller.reset_sim_episode()
        if use_stdin:
            tty.setcbreak(sys.stdin.fileno())
        while True:
            recording_dir = _next_recording_dir(task_dir)
            recorder = DemonstrationRecorder(
                cameras=cameras,
                robot_controller=robot_controller,
                recording_dir=recording_dir,
                task_name=task_name,
                task_instruction=task_instruction,
                interface=interface,
                extra_metadata={
                    "simulator": {"address": sim},
                    "recording_config": {"cameras": camera_configuration or {}},
                },
                monitor=monitor,
            )
            interface.start(robot_controller)
            interface.set_active_recording(robot_controller)
            robot_controller.enable_estop()
            action_log.start()
            recorder.start_recording(hint=f"  Episode {recording_dir.name}\n{hint}")

            action = _wait_for_sim_action(robot_controller, interface, task, use_stdin, recorder)
            if action == "save":
                saved_dir = recorder.stop_recording(
                    complete=True, shutdown_robots=False, finalize=action_log.save
                )
                on_saved(saved_dir)
                last_saved, n_saved = saved_dir, n_saved + 1
                print(f"✓ Saved as success → {saved_dir}  ({n_saved} this session)")
                task.poll(force=True)
                if task.status.get("enabled", True) and not task.status.get("success"):
                    print("  Note: the sim does not score this episode a success.")
            else:
                recorder.discard()
                action_log.discard()
            recorder = None
            if action == "quit":
                print("\nEnding session.\n")
                break
            if action == "calibrate":
                # Calibration stays out of recordings: finish it, then record from a fresh reset.
                print("  Calibrating. The scene resets when it is done.")
                while (
                    interface.calibrating
                    and not robot_controller.session_estop_requested
                ):
                    time.sleep(0.05)

            interface.stop(robot_controller)
            robot_controller.reset_sim_episode()
    except Exception as e:
        print(f"\nError during recording: {e}")
        if recorder is not None and recorder.is_recording:
            recorder.stop_recording(complete=False, shutdown_robots=False)
    finally:
        if old_settings is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)
        interface.set_active_recording(None)
        action_log.close()
        task.close()
        robot_controller.shutdown()
        active_ctrl[0] = None
    return last_saved


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_recording(
    s3_bucket: Optional[str],
    s3_prefix: str,
    interface: TeleopInterface,
    camera_config_file: str = CAMERA_CONFIG,
    calibration_file: str = CALIBRATION_FILE,
    arms: str = "bimanual",
    data_dir: str = "data",
    sim: str = "",
    monitor: bool = False,
    monitor_view: str = "",
    monitor_web: bool = False,
    sim_fps: int = 30,
    sim_record_mode: str = "frames",
    ui: bool = False,
    ui_port: int = 8765,
    ui_host: str = "127.0.0.1",
) -> None:
    """Run teleoperation with continuous demonstration recording.

    ``sim`` = address of a running MESA ``raiden_sim_server``: the left follower and both
    cameras are the digital twin; teleop, timing and files are unchanged.  The episode
    flow differs: recording runs from every scene reset and only a success is saved
    (:func:`_run_sim_episodes`).

    Cameras and robots are fully instantiated and torn down each episode so
    every demonstration starts from a clean state.

    - Press the leader button to START each episode; press again to STOP and save.
    - After each stop the system re-initializes and shows a READY prompt.
      Press the leader button to start the next episode, or 'q' to end the session.
    - Incomplete recordings (estop / Ctrl-C) are retained for recovery and
      identified by the ``complete`` flag in metadata.json.

    ``ui`` serves a browser page (:mod:`raiden.record_ui`) for the session: task and
    teacher, live views with the placement limits, start/stop/label buttons and
    per-episode QC. The buttons, pedal and keyboard keep working next to it.
    """
    print("\n" + "=" * 60)
    print("  DEMONSTRATION RECORDING")
    print("=" * 60 + "\n")

    if sim:
        from raiden.sim.client import require_arm_mode

        require_arm_mode(sim, arms)

    record_ui = None
    if ui and sim:
        print("  --ui supports real recording only; continuing in the terminal.")
    elif ui:
        from raiden.record_ui import RecordUI

        record_ui = RecordUI(port=ui_port, host=ui_host, data_dir=data_dir)
        try:
            record_ui.start()
        except RuntimeError as e:
            print(f"  {e}; continuing in the terminal.")
            record_ui = None

    # ── task and teacher selection (once per session) ────────────────────
    if record_ui is not None:
        print("  Pick the task and teacher in the browser.")
        task_name, task_instruction, teacher_id = record_ui.setup_session()
    else:
        task_name, task_instruction = select_task()
    task_dir = Path(data_dir) / "raw" / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    if record_ui is None:
        teacher_id = select_teacher()

    print(f"\n  Task       : {task_name}")
    print(f"  Instruction: {task_instruction}\n")

    # ── snapshot camera config and look up latest calibration result ─────
    db = get_db()
    try:
        with open(camera_config_file) as _f:
            _cam_cfg_data = json.load(_f)
        camera_config_id = db.snapshot_camera_config(_cam_cfg_data, camera_config_file)
    except Exception:
        camera_config_id = db.snapshot_camera_config({}, camera_config_file)

    task_record = db.get_task_by_name(task_name)
    task_id = task_record["id"] if task_record else None

    latest_calib = db.get_latest_calibration_result()
    calibration_result_id = latest_calib["id"] if latest_calib else None

    def _copy_calibration(dest_dir: Path) -> None:
        if sim:
            from raiden.sim import write_sim_files

            write_sim_files(dest_dir, sim)
            print(f"✓ Sim calibration and model written → {dest_dir}")
            return
        from raiden.sim import table_pose

        calib_src = Path(calibration_file)
        calib = json.loads(calib_src.read_text()) if calib_src.exists() else {}
        if not calib:
            print(f"  Warning: calibration file not found at {calib_src}")
        # The table pose comes from the taped rig layout, the same file the sim table is built from.
        layout = Path(data_dir) / "real2sim" / "calibration" / "layout.json"
        if "table" not in calib and layout.exists():
            calib["table"] = {
                "T_base_table": table_pose(json.loads(layout.read_text())).tolist()
            }
        if calib:
            (dest_dir / "calibration_results.json").write_text(
                json.dumps(calib, indent=2)
            )
            print(f"✓ Calibration written → {dest_dir / 'calibration_results.json'}")

    # ── optional footpedal (once per session) ────────────────────────────
    interface.auto_reset = bool(sim)
    interface.open()

    # ── one-time camera + robot initialisation ───────────────────────────
    print("Initializing cameras...")
    try:
        if sim:
            from raiden.sim import load_sim_cameras

            cameras = load_sim_cameras(sim, camera_config_file, fps=sim_fps, record_mode=sim_record_mode)
        else:
            cameras = load_cameras_from_config(camera_config_file)
        cfg = CameraConfig(camera_config_file)
        camera_configuration = _camera_recording_config(
            cfg, [camera.name for camera in cameras] if sim else cfg.list_camera_names()
        )
    except Exception as e:
        print(f"Error initialising cameras: {e}")
        interface.close()
        if record_ui is not None:
            record_ui.notice(f"camera start failed: {e}")
            record_ui.close()
        return
    print(f"✓ {len(cameras)} camera(s) ready\n")

    live_monitor = None
    if monitor or monitor_view:
        from raiden.guides import make_guides
        from raiden.monitor import make_monitor

        # The grasp guides MESA's collector draws on the gripper, projected into the views
        # the operator is actually watching.  Overlay only -- recorded frames are untouched.
        try:
            guides = make_guides(cameras, sim=sim)
        except Exception as exc:
            print(f"  grasp guides unavailable ({type(exc).__name__}: {exc})")
            guides = None
        default_view = "operator,topdown" if sim and cameras[0].backend != "isaac" else ""
        view = {"": default_view, "none": ""}.get(monitor_view, monitor_view)
        shown = [c.name for c in cameras if not view or "wrist" in c.name]
        live_monitor = make_monitor(
            shown, sim=sim, view=view, web=monitor_web, guides=guides
        )

    # The page shows the recorded frames and, in between recordings, reads the open cameras.
    recorder_monitor = live_monitor
    if record_ui is not None:
        recorder_monitor = record_ui.monitor(live_monitor)
        record_ui.attach_cameras(cameras)

    # Signal handler updated each episode to point at the current controller.
    _active_ctrl: List[Optional[RobotController]] = [None]

    def emergency_stop(signum, frame):
        if _active_ctrl[0] is not None:
            _active_ctrl[0].emergency_stop()

    signal.signal(signal.SIGTERM, emergency_stop)
    signal.signal(signal.SIGINT, emergency_stop)

    def _close_session() -> None:
        cleanup = [(f"camera {cam.name}", cam.close) for cam in cameras]
        if record_ui is not None:
            cleanup.insert(0, ("camera preview pump", record_ui.pause_pump))
        if live_monitor is not None:
            cleanup.append(("live monitor", live_monitor.close))
        cleanup.append(("teleop interface", interface.close))
        if record_ui is not None:
            cleanup.append(("recording UI", record_ui.close))
        errors = []
        for name, close in cleanup:
            try:
                close()
            except BaseException as error:
                errors.append((name, error))
        if errors:
            first_error = errors[0][1]
            for name, error in errors:
                first_error.add_note(f"Closing {name}: {error!r}")
            raise first_error

    if sim:

        def _save_demo(saved_dir: Path) -> None:
            _copy_calibration(saved_dir)
            if task_id is not None:
                demo_id = db.add_demonstration(
                    teacher_id=teacher_id,
                    task_id=task_id,
                    raw_data_path=str(saved_dir),
                    camera_config_id=camera_config_id,
                    calibration_result_id=calibration_result_id,
                )
                db.update_demonstration(demo_id, status="success", converted=False)

        try:
            last_saved_dir = _run_sim_episodes(
                cameras=cameras,
                interface=interface,
                sim=sim,
                task_dir=task_dir,
                task_name=task_name,
                task_instruction=task_instruction,
                monitor=live_monitor,
                on_saved=_save_demo,
                active_ctrl=_active_ctrl,
                camera_configuration=camera_configuration,
                arms=arms,
            )
        finally:
            _close_session()
        _report_session(last_saved_dir, s3_bucket, s3_prefix)
        return

    # ── continuous episode loop ──────────────────────────────────────────
    # Cameras stay open for the whole session (SDK init is slow).
    # Robot controller is reinited each episode — keeping motor CAN threads
    # alive while idle causes DM4310 watchdog / loss-communication errors.
    last_saved_dir: Optional[Path] = None
    recorder: Optional[DemonstrationRecorder] = None
    robot_controller: Optional[RobotController] = None
    tally: "Counter[str]" = Counter()

    use_right = arms == "bimanual" and not sim
    use_left = True

    try:
        while True:
            recorder = None
            if record_ui is not None:
                record_ui.set_phase("init")

            # ── per-episode: init robots ──────────────────────────────────
            robot_controller = RobotController(
                use_right_leader=interface.uses_leaders and use_right,
                use_left_leader=interface.uses_leaders and use_left,
                use_right_follower=use_right,
                use_left_follower=use_left,
                sim=sim or None,
            )
            _active_ctrl[0] = robot_controller

            try:
                robot_controller.setup_for_teleop_recording()
            except Exception as e:
                print(f"Error initialising robots: {e}")
                break

            # Setup interface (warmup IK, attach devices, etc.)
            interface.setup(robot_controller)
            # Quest: teleop runs from here with the arm held at home, so the operator can
            # calibrate before the recording instead of during it.
            started = interface.start_ready(robot_controller)

            # Flush stdout so any SDK log output has time to drain before
            # printing the READY banner.
            sys.stdout.flush()
            time.sleep(0.1)

            # ── wait to start ────────────────────────────────────────────
            print("\n" + "=" * 60)
            print("  READY")
            print("=" * 60)
            # The tally belongs here rather than next to the saved-recording line:
            # robot shutdown and re-init print between the two, so a total printed at
            # the end of an episode has scrolled away by the time the operator looks.
            if sum(tally.values()):
                print(_tally_line(tally))
            print(f"\n  Data dir   : {task_dir}")
            if interface.waits_for_button_start:
                print("\n  Press button on any leader arm or left pedal to START.")
            else:
                print("\n  Press Enter or left pedal to START recording.")
            if interface.ready_hint:
                print(interface.ready_hint)
            print("  Press 'q' to end session.\n")
            print("=" * 60 + "\n")

            if record_ui is not None:
                record_ui.set_phase(
                    "ready", episode=_peek_next_episode(task_dir), hint=interface.ready_hint or ""
                )
            if interface.waits_for_button_start:
                quit_session = _wait_for_start_or_quit(robot_controller, interface, record_ui)
            else:
                quit_session = _wait_for_enter_or_quit(robot_controller, interface, record_ui)
            if quit_session:
                print("\nEnding session.\n")
                break

            # ── start episode ────────────────────────────────────────────
            recording_dir = _next_recording_dir(task_dir)
            print(f"\n  Output: {recording_dir}")

            # Sim object poses, task flags and state are logged per frame by the cameras.
            extra_meta: Dict[str, object] = (
                {"simulator": {"address": sim}} if sim else {}
            )
            extra_meta["recording_config"] = {"cameras": camera_configuration}
            recorder = DemonstrationRecorder(
                cameras=cameras,
                robot_controller=robot_controller,
                recording_dir=recording_dir,
                task_name=task_name,
                task_instruction=task_instruction,
                interface=interface,
                extra_metadata=extra_meta,
                monitor=recorder_monitor,
            )
            _copy_calibration(recording_dir)
            if not started:
                interface.start(robot_controller)
            if record_ui is not None:
                # The recording restarts every camera pipeline: the page's own reads stop first.
                record_ui.pause_pump()
                record_ui.set_phase("recording", episode=recording_dir.name, rec_started=time.time())
            recorder.start_recording()
            robot_controller.enable_estop()
            interface.set_active_recording(robot_controller)

            # ── wait for stop ────────────────────────────────────────────
            forced_failure = False
            forced_success = False

            needs_stdin = not interface.waits_for_button_start
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd) if needs_stdin else None
            if needs_stdin:
                tty.setcbreak(fd)
            try:
                while True:
                    recorder.raise_if_failed()
                    if robot_controller.session_estop_requested:
                        break
                    if interface.poll_success(robot_controller):
                        forced_success = True
                        break
                    if interface.poll_failure(robot_controller):
                        forced_failure = True
                        break
                    if needs_stdin and select.select([sys.stdin], [], [], 0)[0]:
                        ch = sys.stdin.read(1)
                        if ch in ("\r", "\n", " "):
                            break
                    # Leader button, or the Quest's A/X (started at READY) stops the recording.
                    if (started or not needs_stdin) and interface.poll(
                        robot_controller
                    ):
                        break
                    if record_ui is not None:
                        record_ui.poll_interface(interface)
                        cmd = record_ui.take("stop", "success", "failure")
                        forced_success = cmd == "success"
                        forced_failure = cmd == "failure"
                        if cmd is not None:
                            break
                    time.sleep(0.05)
            finally:
                if old_settings is not None:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

            interface.set_active_recording(None)
            estop = robot_controller.session_estop_requested

            # ── verdict phase (arms still active, before shutdown) ────────
            if estop or forced_failure:
                verdict: Optional[str] = "failure"
            elif forced_success:
                verdict = "success"
            else:
                verdict = _wait_for_verdict(robot_controller, interface, record_ui, recorder)

            # ── stop episode (shuts down robots, keeps cameras open) ──────
            if record_ui is not None:
                record_ui.set_phase("saving")
            try:
                saved_dir = recorder.stop_recording(complete=not estop)
            finally:
                # stop_recording owns shutdown, including a failed shutdown.
                _active_ctrl[0] = None
                robot_controller = None
            recorder = None
            if record_ui is not None:
                record_ui.resume_pump()

            _copy_calibration(saved_dir)

            if estop:
                if task_id is not None:
                    demo_id = db.add_demonstration(
                        teacher_id=teacher_id,
                        task_id=task_id,
                        raw_data_path=str(saved_dir),
                        camera_config_id=camera_config_id,
                        calibration_result_id=calibration_result_id,
                    )
                    db.update_demonstration(demo_id, status="failure", converted=False)
                tally["failure"] += 1
                print("\nRecording aborted — marked as failure.")
                print(_tally_line(tally) + "\n")
                if record_ui is not None:
                    record_ui.episode_saved(saved_dir)
                    record_ui.notice("recording aborted (e-stop): marked failure, session ended")
                break

            last_saved_dir = saved_dir

            # ── record demonstration in DB ───────────────────────────────
            if task_id is not None:
                demo_id = db.add_demonstration(
                    teacher_id=teacher_id,
                    task_id=task_id,
                    raw_data_path=str(saved_dir),
                    camera_config_id=camera_config_id,
                    calibration_result_id=calibration_result_id,
                )
                status = verdict if verdict is not None else "pending"
                db.update_demonstration(demo_id, status=status, converted=False)
                if verdict:
                    print(f"  Demonstration marked as: {verdict}")

            tally[verdict or "pending"] += 1
            print(f"✓ Recording saved to: {saved_dir}\n")
            if record_ui is not None:
                record_ui.episode_saved(saved_dir)
            # Loop back — cameras stay open, robots reinited next iteration.
            # The running tally prints under that iteration's READY banner.

    except KeyboardInterrupt:
        print("\nCancelled by user.")
        if recorder is not None and recorder.is_recording:
            try:
                saved_dir = recorder.stop_recording(complete=False)
            finally:
                _active_ctrl[0] = None
                robot_controller = None
            _copy_calibration(saved_dir)
    except Exception as e:
        print(f"\nError during recording: {e}")
        if recorder is not None and recorder.is_recording:
            try:
                saved_dir = recorder.stop_recording(complete=False)
            finally:
                _active_ctrl[0] = None
                robot_controller = None
            _copy_calibration(saved_dir)
    finally:
        # ── session teardown ──────────────────────────────────────────────
        # Robots are shut down per-episode; only cameras and interface remain.
        try:
            if robot_controller is not None:
                _active_ctrl[0] = None
                robot_controller.shutdown()
        finally:
            _close_session()

    _report_session(last_saved_dir, s3_bucket, s3_prefix)


def _tally_line(tally: "Counter[str]") -> str:
    """Running session total, printed after every episode of a real recording.

    Success and failure always show, zeros included, so the two numbers stay in the
    same place after every episode and can be read without parsing the line.
    Unlabelled is only listed when it happens, because it means a verdict was missed
    rather than recorded.
    """
    parts = [f"{tally['success']} success", f"{tally['failure']} failure"]
    if tally["pending"]:
        parts.append(f"{tally['pending']} unlabelled")
    return f"  Session so far: {sum(tally.values())} recorded — {', '.join(parts)}"


def _report_session(
    last_saved_dir: Optional[Path], s3_bucket: Optional[str], s3_prefix: str
) -> None:
    # ── optional S3 upload ───────────────────────────────────────────────
    if s3_bucket and last_saved_dir:
        print("\nUploading to S3...")
        upload_to_s3(last_saved_dir, s3_bucket, s3_prefix)

    if last_saved_dir:
        print(f"\n✓ Done.  Last recording at: {last_saved_dir}")
        print("  Run 'rd convert' to extract PNG frames from camera files.\n")
    else:
        print("\n✓ Done (no recordings saved).\n")
