"""Simulated D435s rendered by the sim server, recorded as states rather than pixels."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from raiden.camera_config import CameraConfig
from raiden.cameras.base import Camera, CameraFrame
from raiden.sim.client import DEFAULT_ADDRESS, SimConnection

RECORDING_EXTENSION = "simrec"


class SimCamera(Camera):
    """Renders one rig camera at ``fps`` through the sim server.

    rig.json holds each camera in the frame its real counterpart records, so frames and
    intrinsics are used as rendered (:func:`load_sim_cameras` checks this). The rendered frames
    feed the live monitor only. Recording writes no pixels: ``<path>/sim_state.npz`` holds, per
    frame, the sim state the frame was rendered from, the task object poses (base frame) and the
    subtask / success flags; ``timestamps.npy`` and ``camera_info.json`` complete it. The
    converter renders the frames again from those states and the episode's ``sim_model.xml``.
    """

    def __init__(self, name: str, address: str = DEFAULT_ADDRESS, fps: int = 30):
        self._name = name
        self._address = address
        self._fps = fps
        self._c: Optional[SimConnection] = None
        self._info: Optional[dict] = None
        self._frame: Optional[CameraFrame] = None
        self._next_t: Optional[float] = None
        self._rec_dir: Optional[Path] = None
        self._timestamps: List[int] = []
        self._sim_log: List[dict] = []

    # -- identity ------------------------------------------------------------
    @property
    def name(self) -> str:
        return self._name

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
        if "mujoco_camera" not in self._info:
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
        self._rec_dir = Path(path)
        self._rec_dir.mkdir(parents=True, exist_ok=True)

    def stop_recording(self) -> None:
        if self._rec_dir is None:
            return
        rec_dir, self._rec_dir = self._rec_dir, None
        np.save(
            str(rec_dir / "timestamps.npy"), np.array(self._timestamps, dtype=np.int64)
        )
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
        # Pace to the camera frame rate like a real SDK would.
        now = time.monotonic()
        if self._next_t is None or now > self._next_t + 1.0 / self._fps:
            self._next_t = now
        else:
            time.sleep(max(0.0, self._next_t - now))
        self._next_t += 1.0 / self._fps
        recording = (
            self._rec_dir is not None
        )  # the recording this frame belongs to, if any
        try:
            res = self._c.call(
                "render", cameras=[self._name], depth=False, with_state=recording
            )[self._name]
        except (RuntimeError, EOFError, OSError):
            return False
        color = np.ascontiguousarray(
            res["rgb"][:, :, ::-1]
        )  # RGB -> BGR like RealSense
        self._frame = CameraFrame(
            color=color, depth=None, timestamp_ns=int(res["t_ns"])
        )
        if recording:
            self._timestamps.append(self._frame.timestamp_ns)
            self._sim_log.append(res["sim"])
        return True

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
        return K, np.zeros(5, dtype=np.float64), (w, h)

    def get_camera_info(self) -> dict:
        fx, fy, cx, cy, w, h = self._intrinsics()
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
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "k3": 0.0,
            "simulated": True,
            "mujoco_camera": self._info["mujoco_camera"],
        }


def load_sim_cameras(
    address: str, camera_config_file: str, fps: int = 30
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
        cam = SimCamera(name, address=address, fps=fps)
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
