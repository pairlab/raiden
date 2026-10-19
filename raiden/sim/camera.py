"""Rig cameras with backend-specific, replayable simulation recordings."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from raiden.camera_config import CameraConfig
from raiden.cameras.base import Camera, CameraFrame
from raiden.sim.client import DEFAULT_ADDRESS, SimConnection
from raiden.sim.serialization import jsonable

RECORDING_EXTENSION = "simrec"


class SimCamera(Camera):
    """Record native camera timestamps and states, optionally retaining Isaac RGB.

    MuJoCo retains its compact state-only NPZ format. Isaac retains portable full
    scene/controller snapshots in JSONL plus exact optical camera poses. Its RGB
    is saved losslessly by default; ``record_mode="states"`` defers pixel storage
    to ``scripts/isaac_replay.py --mode states --rerender``.
    """

    def __init__(
        self, name: str, address: str = DEFAULT_ADDRESS, fps: int = 30,
        record_mode: str = "frames",
    ):
        if fps <= 0:
            raise ValueError("camera fps must be positive")
        if record_mode not in ("frames", "states"):
            raise ValueError("record_mode must be frames or states")
        self._name, self._address, self._fps = name, address, fps
        self._record_mode = record_mode
        self._c: Optional[SimConnection] = None
        self._info: Optional[dict] = None
        self._frame: Optional[CameraFrame] = None
        self._next_t: Optional[float] = None
        self._rec_dir: Optional[Path] = None
        self._timestamps: List[int] = []
        self._sim_log: List[dict] = []
        self._state_stream = None
        self._camera_poses: List[np.ndarray] = []
        self._last_frame_id = None
        self._last_grab_error = None

    # -- identity ------------------------------------------------------------
    @property
    def name(self) -> str:
        return self._name

    @property
    def backend(self) -> str:
        return self._info.get("backend", "mujoco") if self._info else ""

    @property
    def serial_number(self) -> str:
        return self._info["serial"] if self._info else f"sim-{self._name}"

    @property
    def recording_extension(self) -> str:
        return RECORDING_EXTENSION

    # -- lifecycle -----------------------------------------------------------
    def open(self) -> None:
        self._c = SimConnection(self._address)
        self._info = self._c.call("get_camera_info", name=self._name)
        if "mujoco_camera" not in self._info and self.backend != "isaac":
            raise RuntimeError(
                "the sim server does not name its MuJoCo cameras, so recorded frames could not be "
                "rendered again; restart scripts/raiden_sim_server.py (MESA)"
            )
        self._next_t = None

    def close(self) -> None:
        self.stop_recording()
        if self._c is not None:
            self._c.close()
            self._c = None

    # -- recording -----------------------------------------------------------
    def start_recording(self, path: Path) -> None:
        self._timestamps = []
        self._sim_log = []
        self._camera_poses = []
        self._last_frame_id = None
        self._next_t = None  # a recording starts its own frame grid, not the idle one's phase
        self._rec_dir = Path(path)
        self._rec_dir.mkdir(parents=True, exist_ok=True)
        if self.backend == "isaac":
            self._state_stream = (self._rec_dir / "sim_state.jsonl").open("w")
            if self._record_mode == "frames":
                (self._rec_dir / "rgb").mkdir(exist_ok=True)

    def stop_recording(self) -> None:
        if self._rec_dir is None:
            return
        rec_dir, self._rec_dir = self._rec_dir, None
        np.save(
            str(rec_dir / "timestamps.npy"), np.array(self._timestamps, dtype=np.int64)
        )
        if self._state_stream is not None:
            self._state_stream.close()
            self._state_stream = None
            poses = np.asarray(self._camera_poses, dtype=np.float64).reshape(-1, 4, 4)
            np.save(rec_dir / "camera_poses.npy", poses)
        if self._sim_log:
            self._save_sim_log(rec_dir / "sim_state.npz")
        with open(rec_dir / "camera_info.json", "w") as f:
            json.dump(self.get_camera_info(), f, indent=2)

    def _save_sim_log(self, path: Path) -> None:
        """One snapshot per recorded frame (the server's ``_snapshot``), in frame order."""
        log = self._sim_log
        objects = list(log[0]["objects"])
        np.savez_compressed(
            str(path),
            t_ns=np.array(self._timestamps, dtype=np.int64),
            state=np.stack([s["state"] for s in log]),
            warmstart=np.stack([s["warmstart"] for s in log]),
            objects=np.array(objects),
            object_poses=np.array([[s["objects"][o] for o in objects] for s in log]),
            subtasks=np.array(self._c.call("get_task_status")["subtasks"]),
            subtask_done=np.array([s["subtasks_done"] for s in log], dtype=bool),
            success=np.array([s["success"] for s in log], dtype=bool),
        )

    # -- frames --------------------------------------------------------------
    def grab(self) -> bool:
        if self._c is None:
            return False
        # Pace to a fixed 1/fps grid like a real camera: a slot missed by more than half
        # a period is dropped, not caught up with a burst of short frames.
        period = 1.0 / self._fps
        now = time.monotonic()
        if self._next_t is None:
            self._next_t = now
        elif now > self._next_t:
            self._next_t += round((now - self._next_t) / period) * period
        time.sleep(max(0.0, self._next_t - now))
        self._next_t += period
        recording = self._rec_dir is not None
        try:
            native_wait = (
                {"after_frame_id": self._last_frame_id} if self.backend == "isaac" else {}
            )
            res = self._c.call(
                "render", cameras=[self._name], depth=False, with_state=recording, **native_wait
            )[self._name]
        except (RuntimeError, EOFError, OSError) as exc:
            message = f"{type(exc).__name__}: {exc}"
            if message != self._last_grab_error:
                print(f"  Camera {self.name}: {message}")
                self._last_grab_error = message
            return False
        self._last_grab_error = None
        if self.backend == "isaac":
            frame_id = res["frame_id"]
            if frame_id == self._last_frame_id:
                return False
            self._last_frame_id = frame_id
        color = np.ascontiguousarray(
            res["rgb"][:, :, ::-1]
        )  # RGB -> BGR like RealSense
        self._frame = CameraFrame(
            color=color, depth=None, timestamp_ns=int(res["t_ns"])
        )
        if recording:
            if self.backend == "isaac":
                self._save_isaac_frame(res, color)
            else:
                self._sim_log.append(res["sim"])
            self._timestamps.append(self._frame.timestamp_ns)
        return True

    def _save_isaac_frame(self, result: dict, color: np.ndarray) -> None:
        snapshot = dict(result["sim"])
        pose = np.asarray(result["T_base_cam"], dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise RuntimeError(f"{self.name}: Isaac render lacks a finite T_base_cam")
        snapshot.update(
            t_ns=self._frame.timestamp_ns,
            frame_id=result["frame_id"],
            received_t_ns=time.time_ns(),
            received_monotonic_ns=time.monotonic_ns(),
            T_base_cam=pose,
        )
        index = len(self._timestamps)
        if self._record_mode == "frames" and not cv2.imwrite(
            str(self._rec_dir / "rgb" / f"{index:010d}.png"), color
        ):
            raise OSError(f"failed to save {self.name} frame {index}")
        self._state_stream.write(
            json.dumps(jsonable(snapshot), separators=(",", ":"), allow_nan=False) + "\n"
        )
        self._state_stream.flush()
        self._camera_poses.append(pose)

    def get_frame(self) -> CameraFrame:
        if self._frame is None:
            raise RuntimeError("No frames available. Call grab() first.")
        return self._frame

    # -- calibration ---------------------------------------------------------
    def _intrinsics(self) -> Tuple[float, float, float, float, int, int]:
        assert self._info is not None
        K = np.asarray(self._info["K"], dtype=np.float64)
        return (
            K[0, 0],
            K[1, 1],
            K[0, 2],
            K[1, 2],
            int(self._info["width"]),
            int(self._info["height"]),
        )

    def get_intrinsics(self) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        fx, fy, cx, cy, w, h = self._intrinsics()
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        distortion = np.asarray(self._info.get("distortion_coeffs", np.zeros(5)), dtype=np.float64)
        return K, distortion, (w, h)

    def get_camera_info(self) -> dict:
        fx, fy, cx, cy, w, h = self._intrinsics()
        distortion = np.asarray(self._info.get("distortion_coeffs", np.zeros(5)), dtype=np.float64)
        distortion = np.pad(distortion.ravel(), (0, max(0, 5 - distortion.size)))
        return {
            "serial_number": self.serial_number,
            "model": "MESA sim (RealSense D435 intrinsics)",
            "fps": self._fps,
            "width": w,
            "height": h,
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "k1": float(distortion[0]),
            "k2": float(distortion[1]),
            "p1": float(distortion[2]),
            "p2": float(distortion[3]),
            "k3": float(distortion[4]),
            "simulated": True,
            **{k: jsonable(v) for k, v in self._info.items()
               if k not in ("K", "width", "height", "serial")},
            "recording_mode": self._record_mode if self.backend == "isaac" else "states",
        }


def load_sim_cameras(
    address: str, camera_config_file: str, fps: int = 30, record_mode: str = "frames"
) -> List[Camera]:
    """One SimCamera per rig camera; each must render the frame its real counterpart records.

    ``fps`` defaults to the real D435 rate so sim episodes have the same temporal density as
    real ones.  Nothing in the sim requires it: raise it to collect faster, at the cost of
    sim and real datasets no longer matching frame-for-frame.
    """
    conn = SimConnection(address)
    rig = conn.call("get_rig")
    conn.close()
    cfg = CameraConfig(camera_config_file)
    cameras: List[Camera] = []
    for name, rig_cam in rig["cameras"].items():
        _check_real_frame(cfg, name, rig_cam["resolution"])
        cam = SimCamera(name, address=address, fps=fps, record_mode=record_mode)
        cam.open()
        w, h = rig_cam["resolution"]
        print(f"  ✓ Camera '{name}' opened (sim, {w}x{h})")
        cameras.append(cam)
    return cameras


def _check_real_frame(cfg: CameraConfig, name: str, rendered: Tuple[int, int]) -> None:
    """Refuse a rig camera whose render size differs from the frame the real camera records.

    real2sim_calibrate.py writes rig.json in the recorded frame (camera.json ``resolution``,
    then ``crop``); an older rig.json holds the calibration capture (848x480) instead.
    """
    stream = cfg.get_resolution(name)
    if stream is None:
        return  # no real RealSense to match
    crop = cfg.get_crop(name)
    real = tuple(crop[2:]) if crop else stream
    if tuple(rendered) != real:
        raise ValueError(
            f"Camera '{name}': the sim renders {rendered[0]}x{rendered[1]} but the real camera records "
            f"{real[0]}x{real[1]}; re-run scripts/real2sim_calibrate.py and copy rig.json to MESA"
        )
