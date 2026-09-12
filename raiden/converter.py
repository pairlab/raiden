"""Convert raw camera recordings to UnifiedDataset format.

Input:  a recording directory produced by ``rd record``.
Output: UnifiedDataset layout inside the same directory.

Usage::

    rd convert ./data/test_2/test_task_2_20260223_162246

Output layout::

    <recording_dir>/
        split_all.json
        metadata_shared.json
        calibration_results.json      # copied from the first recording dir
        0000/
            metadata.json
            rgb/
                scene_camera/
                    0000000000.png      # lossless PNG
                    0000000001.png
                    ...
                left_wrist_camera/
                    ...
            depth/
                scene_camera/
                    0000000000.npz      # np.uint16, millimetres
                    ...
            lowdim/
                scene_camera/
                    0000000000.npz      # intrinsics, extrinsics, action, language
                    0000000001.npz
                    ...
                left_wrist_camera/
                    ...
                right_wrist_camera/
                    ...
        cameras/                      # raw SVO2/bag files kept
        metadata.json                 # original recording metadata (updated: converted=true)
        robot_data.npz
"""

import json
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from raiden._config import CAMERA_CONFIG
from raiden.camera_config import CameraConfig

_SEQUENCE_NAME = "0000"
_IMG_EXT = ".png"
# Per-frame camera-to-base poses, written next to a sim camera's rendered frames so that they
# follow the frames through timestamp selection and trimming.
_POSES_FILE = "camera_poses.npy"

# Cameras whose images are physically mounted upside-down and need a 180° correction.
_FLIP_CAMERAS = {"right_wrist_camera"}

# Camera role → robot_data joint key.
_ROLE_TO_JOINT_KEY: Dict[str, str] = {
    "left_wrist": "follower_l_joint_pos",
    "right_wrist": "follower_r_joint_pos",
}

# Lazily-loaded kinematics instance (MuJoCo FK for YAM arm).
_kinematics: Any = None

# Unix timestamp for 2020-01-01 in nanoseconds — anything below this is a
# hardware-relative counter, not wall-clock.
_WALL_CLOCK_MIN_NS = 1_577_836_800_000_000_000


class ConversionError(RuntimeError):
    """Raised when a recording cannot be converted into a consistent episode."""


def _apply_rs_clock_offset(
    ts_arr: np.ndarray, clock_offset: Optional[int]
) -> np.ndarray:
    """Shift RealSense frame timestamps onto the host clock if they are not already.

    With ``global_time_enabled`` the bag timestamps are wall-clock and the
    offset measured at record time is just pipeline latency (tens to hundreds
    of ms, different per camera) — applying it would skew the cameras against
    each other and against the robot data.  Only apply it to hardware-clock
    timestamps.
    """
    if clock_offset is None or len(ts_arr) == 0 or int(ts_arr[0]) > _WALL_CLOCK_MIN_NS:
        return ts_arr
    return ts_arr + int(clock_offset)


def _get_kinematics() -> Any:
    global _kinematics
    if _kinematics is None:
        from i2rt.robots.kinematics import Kinematics

        from raiden._xml_paths import get_yam_4310_linear_xml_path

        _kinematics = Kinematics(get_yam_4310_linear_xml_path(), "grasp_site")
    return _kinematics


# 4x4 homogeneous matrix for 180° rotation around the optical (Z) axis.
# Corrects both extrinsics and principal point when a camera is mounted upside-down.
_R_FLIP_180 = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Pose helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Frame extraction helpers
# ---------------------------------------------------------------------------


def _count_svo2_frames(svo_path: Path) -> int:
    """Return the total frame count of an SVO2 file without extracting any frames."""
    from raiden.cameras.zed import ZedCamera

    # Depth is never needed for counting; always use DEPTH_MODE.NONE to avoid
    # wasting GPU memory (especially important when FFS will run afterwards).
    camera = ZedCamera.from_svo(svo_path.stem, svo_path, compute_sdk_depth=False)
    n = camera.get_total_frames()
    camera.close()
    return n


def _extract_svo2_synchronized(
    svo_paths: List[Path],
    names: List[str],
    rgb_dirs: List[Path],
    depth_dirs: List[Path],
    flips: List[bool],
    max_frames: Optional[int] = None,
    sync_threshold_ns: int = 16_666_667,  # half a frame at 30 fps
    stereo_method: str = "zed",
    ffs_scale: float = 1.0,
    ffs_iters: int = 8,
    tri_stereo_variant: str = "c64",
) -> Dict[str, Tuple[np.ndarray, Optional[dict]]]:
    """Extract frames from multiple SVO2 files with cross-camera temporal alignment.

    On each output frame slot the algorithm selects the frame from every camera
    whose timestamp is closest to the latest camera in the group, skipping any
    frames that lag by more than *sync_threshold_ns*.  This guarantees that for
    any given output index N the timestamps across all cameras are within
    ``sync_threshold_ns`` of each other.

    Per-camera timestamps (int64, nanoseconds) are saved alongside images as
    ``rgb_dir/timestamps.npy`` for use during lowdim construction.

    Returns
    -------
    dict mapping camera name → (timestamps_ns np.ndarray, camera_info dict or None)
    """
    from raiden.cameras.zed import ZedCamera

    for d in rgb_dirs + depth_dirs:
        d.mkdir(parents=True, exist_ok=True)

    use_ffs = stereo_method == "ffs"
    use_tri_stereo = stereo_method == "tri_stereo"
    use_learned_stereo = use_ffs or use_tri_stereo

    # Lazily create a shared depth predictor (one instance, GPU-loaded once).
    depth_predictor = None
    if use_ffs:
        from raiden.depth.ffs import (
            FFSDepthPredictor,
            FFSOnnxDepthPredictor,
            FFSTrtDepthPredictor,
        )

        if FFSTrtDepthPredictor.engines_available():
            depth_predictor = FFSTrtDepthPredictor()
        elif FFSOnnxDepthPredictor.models_available():
            depth_predictor = FFSOnnxDepthPredictor()
        else:
            depth_predictor = FFSDepthPredictor(scale=ffs_scale, iters=ffs_iters)
    elif use_tri_stereo:
        from raiden.depth.tri_stereo import (  # noqa: PLC0415
            TRIStereoOnnxDepthPredictor,
            TRIStereoTrtDepthPredictor,
        )

        if TRIStereoTrtDepthPredictor.engine_available(variant=tri_stereo_variant):
            pred = TRIStereoTrtDepthPredictor(variant=tri_stereo_variant)
            try:
                pred._ensure_loaded()
                depth_predictor = pred
            except RuntimeError as e:
                print(f"[TRIStereo] TRT engine unusable ({e}), falling back to ONNX")
        if depth_predictor is None:
            if TRIStereoOnnxDepthPredictor.model_available(variant=tri_stereo_variant):
                depth_predictor = TRIStereoOnnxDepthPredictor(
                    variant=tri_stereo_variant
                )
            else:
                raise RuntimeError(
                    f"No TRI Stereo model found for variant '{tri_stereo_variant}'. "
                    f"Run: git lfs pull"
                )

    # Open all cameras.
    cams: Dict[str, ZedCamera] = {
        name: ZedCamera.from_svo(
            name, svo_path, compute_sdk_depth=not use_learned_stereo
        )
        for name, svo_path in zip(names, svo_paths)
    }

    # Cache per-camera stereo calibration for learned stereo backends.
    stereo_calib: Dict[str, Tuple[float, float]] = {}
    if use_learned_stereo:
        for name, cam in cams.items():
            stereo_calib[name] = cam.get_stereo_calib()
    total_frames = {name: cam.get_total_frames() for name, cam in cams.items()}
    print(f"  Frames per camera: { {n: total_frames[n] for n in names} }")

    rgb_dir_map = dict(zip(names, rgb_dirs))
    depth_dir_map = dict(zip(names, depth_dirs))
    flip_map = dict(zip(names, flips))

    timestamps: Dict[str, List[int]] = {name: [] for name in names}
    frame_idx = 0

    # Initial grab.
    active = {name: cam.grab() for name, cam in cams.items()}

    # Each ZED camera has an independent hardware clock, so absolute timestamps
    # cannot be compared across cameras.  Record each camera's first-frame
    # timestamp and use elapsed time (relative to that origin) for alignment.
    first_ts: Dict[str, int] = {
        name: (cam.get_frame_timestamp_ns() if active[name] else 0)
        for name, cam in cams.items()
    }

    ref_total = max(total_frames.values()) if total_frames else 0
    pbar = tqdm(
        total=ref_total if max_frames is None else min(ref_total, max_frames),
        unit="frame",
        desc="  extracting",
        dynamic_ncols=True,
    )

    while all(active.values()):
        if max_frames is not None and frame_idx >= max_frames:
            break

        # Elapsed nanoseconds since each camera's first frame.
        ts = {
            name: cam.get_frame_timestamp_ns() - first_ts[name]
            for name, cam in cams.items()
        }
        ref_ts = max(ts.values())

        # Advance any camera whose current frame is too far behind the latest.
        advanced = False
        for name, cam in cams.items():
            while active[name] and ref_ts - ts[name] > sync_threshold_ns:
                active[name] = cam.grab()
                if active[name]:
                    ts[name] = cam.get_frame_timestamp_ns() - first_ts[name]
                advanced = True

        if not all(active.values()):
            break
        if advanced:
            # Re-evaluate after advancing (another camera may now be the latest).
            continue

        # All cameras are within threshold — save this synchronized frame.
        for name, cam in cams.items():
            frame = cam.get_frame()
            flip = flip_map[name]

            color = cv2.rotate(frame.color, cv2.ROTATE_180) if flip else frame.color
            cv2.imwrite(str(rgb_dir_map[name] / f"{frame_idx:010d}{_IMG_EXT}"), color)

            if use_learned_stereo:
                fx, baseline = stereo_calib[name]
                # Run inference on raw (pre-rotation) images — the ZED rectifies
                # them in the sensor frame; rotating before inference only adds noise.
                depth_m = depth_predictor.predict(
                    frame.color, cam.get_right_color(), fx, baseline
                )
                if flip:
                    depth_m = cv2.rotate(depth_m, cv2.ROTATE_180)
                depth_mm = (depth_m * 1000.0).clip(0, 65535).astype(np.uint16)
            else:
                depth_mm = (frame.depth * 1000.0).clip(0, 65535).astype(np.uint16)
                if flip:
                    depth_mm = cv2.rotate(depth_mm, cv2.ROTATE_180)
            np.savez_compressed(
                str(depth_dir_map[name] / f"{frame_idx:010d}.npz"), depth=depth_mm
            )

            timestamps[name].append(frame.timestamp_ns)

        frame_idx += 1
        pbar.update(1)

        if use_learned_stereo and frame_idx % 10 == 0 and depth_predictor._n_calls > 0:
            avg_inf = depth_predictor._t_inference / depth_predictor._n_calls * 1000
            pbar.set_postfix(inf_ms=f"{avg_inf:.0f}", refresh=False)

        # Advance all cameras for the next slot.
        active = {name: cam.grab() for name, cam in cams.items()}

    pbar.close()
    print(f"    {frame_idx} synchronized frames extracted")
    if use_learned_stereo and depth_predictor._n_calls > 0:
        label = "FFS" if use_ffs else f"TRIStereo-{tri_stereo_variant.upper()}"
        print(f"  {label} timing: {depth_predictor.timing_summary()}")

    # Collect camera info and persist per-camera timestamps.
    results: Dict[str, Tuple[np.ndarray, Optional[dict]]] = {}
    for name, cam in cams.items():
        camera_info = None
        try:
            camera_info = cam.get_camera_info()
        except Exception:
            pass

        ts_arr = np.array(timestamps[name], dtype=np.int64)
        np.save(str(rgb_dir_map[name] / "timestamps.npy"), ts_arr)

        cam.close()
        results[name] = (ts_arr, camera_info)

    return results


def _extract_bag(
    bag_path: Path,
    rgb_dir: Path,
    depth_dir: Path,
    flip: bool = False,
    max_frames: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[dict]]:
    """Extract color (and depth, if recorded) frames from a RealSense .bag file.

    Returns (timestamps_ns, camera_info).  timestamps_ns is an int64 array of
    per-frame wall-clock timestamps (nanoseconds since Unix epoch) saved by the
    RealSense SDK when global_time_enabled is set.  The array is also written to
    ``rgb_dir/timestamps.npy`` for caching across re-conversions.
    """
    try:
        from raiden.cameras.realsense import RealSenseCamera
    except ImportError:
        print("  pyrealsense2 not installed – skipping .bag conversion")
        return np.array([], dtype=np.int64), None

    rgb_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Opening {bag_path.name} ...")
    crop = CameraConfig(CAMERA_CONFIG).get_crop(bag_path.stem)
    if crop is not None:
        print(f"    crop {crop}")
    camera = RealSenseCamera.from_bag(bag_path.stem, bag_path, crop=crop)

    timestamps: List[int] = []
    idx = 0

    while camera.grab():
        if max_frames is not None and idx >= max_frames:
            break
        frame = camera.get_frame()

        color = cv2.rotate(frame.color, cv2.ROTATE_180) if flip else frame.color
        cv2.imwrite(str(rgb_dir / f"{idx:010d}{_IMG_EXT}"), color)

        if frame.depth is not None:  # bags recorded with depth off have none
            depth_mm = (frame.depth * 1000.0).clip(0, 65535).astype(np.uint16)
            if flip:
                depth_mm = cv2.rotate(depth_mm, cv2.ROTATE_180)
            depth_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(depth_dir / f"{idx:010d}.npz"), depth=depth_mm)

        timestamps.append(frame.timestamp_ns)
        idx += 1

        if idx % 100 == 0:
            print(f"    {idx} frames", end="\r", flush=True)

    print(f"    {idx} frames extracted          ")

    ts_arr = np.array(timestamps, dtype=np.int64)
    if len(ts_arr) > 1:
        duration_s = (int(ts_arr[-1]) - int(ts_arr[0])) / 1e9
        actual_fps = (len(ts_arr) - 1) / duration_s if duration_s > 0 else 0.0
        print(f"    Actual FPS: {actual_fps:.1f}  duration: {duration_s:.2f}s")
        if abs(actual_fps - 30.0) > 3.0:
            print(
                f"    WARNING: FPS {actual_fps:.1f} deviates from expected 30 fps — "
                "check camera stream configuration"
            )

    camera_info = None
    try:
        camera_info = camera.get_camera_info()
    except Exception:
        pass

    camera.close()
    np.save(str(rgb_dir / "timestamps.npy"), ts_arr)
    return ts_arr, camera_info


def _extract_simrec(
    sim_dir: Path,
    rgb_dir: Path,
    model_xml: Path,
    T_world_base: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[dict]]:
    """Write a :class:`raiden.sim.SimCamera` recording into the sequence layout.

    ``sim_dir`` holds ``timestamps.npy`` (host ns), ``camera_info.json`` and ``sim_state.npz``,
    the state each frame was rendered from.  The frames are rendered again from those states in
    ``model_xml`` (the episode's ``sim_model.xml``), and each frame's camera pose is read from
    the same state into ``rgb_dir/camera_poses.npy``.
    """
    ts_path = sim_dir / "timestamps.npy"
    ts_arr = (
        np.load(str(ts_path)).astype(np.int64)
        if ts_path.exists()
        else np.array([], dtype=np.int64)
    )
    info: Optional[dict] = None
    info_path = sim_dir / "camera_info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
    rgb_dir.parent.mkdir(parents=True, exist_ok=True)
    if (sim_dir / "sim_state.npz").exists():
        from raiden.sim.render import render_states

        if not model_xml.exists() or info is None:
            raise ConversionError(
                f"{sim_dir.name}: needs {model_xml.name} and camera_info.json to render its frames"
            )
        states = np.load(str(sim_dir / "sim_state.npz"))["state"]
        poses = render_states(
            model_xml,
            info["mujoco_camera"],
            states,
            info["width"],
            info["height"],
            rgb_dir,
        )
        if T_world_base is None:
            raise ConversionError(
                f"{sim_dir.name}: the sim model is not placed in the base frame"
            )
        np.save(
            str(rgb_dir / _POSES_FILE),
            (np.linalg.inv(T_world_base) @ poses).astype(np.float32),
        )
        print(f"    {len(poses)} frames rendered from the logged sim states")
    else:
        rgb_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(rgb_dir / "timestamps.npy"), ts_arr)
    if len(ts_arr) > 1:
        duration_s = (int(ts_arr[-1]) - int(ts_arr[0])) / 1e9
        fps = (len(ts_arr) - 1) / duration_s if duration_s > 0 else 0.0
        print(f"    Actual FPS: {fps:.1f}  duration: {duration_s:.2f}s")
    return ts_arr, info


def _sim_base_frame(
    simrec_dirs: List[Path], model_xml: Path, calib: Optional[dict]
) -> Optional[np.ndarray]:
    """Place the episode's sim model in the arm base frame, or ``None`` if nothing is rendered."""
    if not model_xml.exists() or not any(
        (d / "sim_state.npz").exists() for d in simrec_dirs
    ):
        return (
            None  # nothing to render, or _extract_simrec will report the missing model
        )
    from raiden.sim.calibration import base_in_world

    cameras: Dict[str, str] = {}
    for d in simrec_dirs:
        info_path = d / "camera_info.json"
        if info_path.exists():
            with open(info_path) as f:
                mujoco_camera = json.load(f).get("mujoco_camera")
            if mujoco_camera:
                cameras[d.stem] = mujoco_camera
    return base_in_world(model_xml, cameras, calib or {})


def _load_sim_log(simrec_dirs: List[Path]) -> Optional[Dict[str, np.ndarray]]:
    """Merge every sim camera's ``sim_state.npz`` into one log sorted by time.

    Each file holds the state its camera's frames were rendered from, so the reference camera's
    frames find their own snapshot exactly.
    """
    parts = [
        dict(np.load(d / "sim_state.npz"))
        for d in simrec_dirs
        if (d / "sim_state.npz").exists()
    ]
    if not parts:
        return None
    order = np.argsort(np.concatenate([p["t_ns"] for p in parts]), kind="stable")
    log = {
        k: np.concatenate([p[k] for p in parts])[order]
        for k in ("t_ns", "state", "object_poses", "subtask_done", "success")
    }
    log["objects"], log["subtasks"] = parts[0]["objects"], parts[0]["subtasks"]
    return log


# ---------------------------------------------------------------------------
# Cross-camera temporal alignment
# ---------------------------------------------------------------------------


def _align_cameras_by_timestamp(
    seq_dir: Path,
    cam_timestamps: Dict[str, Optional[np.ndarray]],
    frame_counts: Dict[str, int],
    camera_start_times_ns: Optional[Dict[str, int]] = None,
    camera_fps: float = 30.0,
    robot_timestamps: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, Optional[np.ndarray]], Dict[str, int]]:
    """Resample cameras onto one timestamp grid inside the telemetry window.

    The first camera is the reference clock. Every other camera is mapped to
    its nearest frame rather than paired by array index. When robot timestamps
    are available, the reference grid is intersected with their range so robot
    values are never extrapolated into a trailing camera-only interval.
    """
    if not frame_counts:
        return cam_timestamps, frame_counts

    missing = []
    raw_ts: Dict[str, np.ndarray] = {}
    for name, count in frame_counts.items():
        timestamps = cam_timestamps.get(name)
        if timestamps is None or len(timestamps) != count:
            missing.append(name)
        else:
            raw_ts[name] = np.asarray(timestamps, dtype=np.int64)
    if missing:
        raise ConversionError(
            "cannot align cameras without one timestamp per frame: "
            + ", ".join(missing)
        )

    for name, ts in raw_ts.items():
        if len(ts) == 0 or np.any(np.diff(ts) < 0):
            raise ConversionError(f"{name} has empty or non-monotonic timestamps")

    wall_clock = {name: int(ts[0]) > _WALL_CLOCK_MIN_NS for name, ts in raw_ts.items()}
    if all(wall_clock.values()):
        aligned_ts = raw_ts
        clock_is_wall = True
    elif camera_start_times_ns and all(
        name in camera_start_times_ns for name in raw_ts
    ):
        # Map only hardware clocks onto the host clock; wall-clock streams are
        # already comparable and must not be shifted to camera startup time.
        aligned_ts = {
            name: (
                ts
                if wall_clock[name]
                else ts - ts[0] + int(camera_start_times_ns[name])
            )
            for name, ts in raw_ts.items()
        }
        clock_is_wall = True
    elif not any(wall_clock.values()):
        # Legacy recordings without start metadata can still align camera-to-camera.
        aligned_ts = {name: ts - ts[0] for name, ts in raw_ts.items()}
        clock_is_wall = False
    else:
        raise ConversionError(
            "camera timestamps use mixed clock domains and recording start times are unavailable"
        )

    t_start = max(int(ts[0]) for ts in aligned_ts.values())
    t_end = min(int(ts[-1]) for ts in aligned_ts.values())
    if robot_timestamps is not None and len(robot_timestamps) > 0:
        robot_ts = np.asarray(robot_timestamps, dtype=np.int64)
        if np.any(np.diff(robot_ts) < 0):
            raise ConversionError("robot timestamps are non-monotonic")
        robot_is_wall = int(robot_ts[0]) > _WALL_CLOCK_MIN_NS
        if clock_is_wall != robot_is_wall:
            raise ConversionError(
                "camera and robot timestamps use different clock domains"
            )
        t_start = max(t_start, int(robot_ts[0]))
        t_end = min(t_end, int(robot_ts[-1]))

    if t_start > t_end:
        raise ConversionError("camera and robot timestamp ranges do not overlap")

    reference = next(iter(frame_counts))
    ref_ts = aligned_ts[reference]
    ref_indices = np.flatnonzero((ref_ts >= t_start) & (ref_ts <= t_end))
    if len(ref_indices) == 0:
        raise ConversionError("no reference-camera frames overlap robot telemetry")
    grid = ref_ts[ref_indices]

    selections: Dict[str, np.ndarray] = {}
    max_allowed_ns = int(round(1e9 / camera_fps))
    for name, ts in aligned_ts.items():
        eligible = np.flatnonzero((ts >= t_start) & (ts <= t_end))
        if len(eligible) == 0:
            raise ConversionError(f"{name} has no frames inside robot telemetry")
        if name == reference:
            indices = ref_indices
        else:
            candidate_ts = ts[eligible]
            right = np.searchsorted(candidate_ts, grid, side="left")
            right = np.clip(right, 0, len(candidate_ts) - 1)
            left = np.maximum(right - 1, 0)
            choose_left = np.abs(grid - candidate_ts[left]) <= np.abs(
                candidate_ts[right] - grid
            )
            indices = eligible[np.where(choose_left, left, right)]

        residual_ns = np.abs(ts[indices] - grid)
        max_residual_ns = int(np.max(residual_ns))
        median_ms = float(np.median(residual_ns)) / 1e6
        max_ms = max_residual_ns / 1e6
        print(
            f"  Timestamp sync {name}: median {median_ms:.2f} ms, max {max_ms:.2f} ms"
        )
        if max_residual_ns > max_allowed_ns:
            raise ConversionError(
                f"{name} residual timestamp error {max_ms:.2f} ms exceeds "
                f"one {1000.0 / camera_fps:.2f} ms frame"
            )
        selections[name] = indices

    selected: Dict[str, np.ndarray] = {}
    for name, indices in selections.items():
        ts = aligned_ts[name]
        _apply_camera_selection(seq_dir, name, indices, ts[indices])
        selected[name] = ts[indices]

    print(
        f"  Timestamp grid: {reference}, {len(grid)} frames over "
        f"{(int(grid[-1]) - int(grid[0])) / 1e9:.2f}s"
    )
    return selected, {name: len(grid) for name in frame_counts}


def _apply_camera_selection(
    seq_dir: Path,
    name: str,
    indices: np.ndarray,
    timestamps: np.ndarray,
) -> None:
    """Reindex one camera stream according to nearest-timestamp selection."""
    # Reindexing rebuilds rgb/<name>, so hold the poses and write them back afterwards.
    poses_path = seq_dir / "rgb" / name / _POSES_FILE
    poses = np.load(str(poses_path))[indices] if poses_path.exists() else None
    for stream, ext in (("rgb", _IMG_EXT), ("depth", ".npz")):
        stream_dir = seq_dir / stream / name
        if not stream_dir.exists():
            continue
        temp_dir = stream_dir.with_name(f".{name}.timestamp-selection")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True)
        try:
            for new_idx, old_idx in enumerate(indices):
                src = stream_dir / f"{int(old_idx):010d}{ext}"
                if not src.exists():
                    raise ConversionError(f"missing camera frame: {src}")
                dst = temp_dir / f"{new_idx:010d}{ext}"
                try:
                    dst.hardlink_to(src)
                except OSError:
                    shutil.copy2(src, dst)
            shutil.rmtree(stream_dir)
            temp_dir.rename(stream_dir)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    np.save(str(seq_dir / "rgb" / name / "timestamps.npy"), timestamps)
    if poses is not None:
        np.save(str(poses_path), poses)


# ---------------------------------------------------------------------------
# Lowdim builder
# ---------------------------------------------------------------------------


def _build_lowdim(
    seq_dir: Path,
    cameras: List[str],
    n_frames: int,
    camera_infos: Dict[str, Optional[dict]],
    calib: Optional[dict],
    robot_data: Optional[Dict[str, np.ndarray]],
    rec_meta: dict,
    flip_cameras: set,
    right_base_to_left_base: Optional[np.ndarray],
    cam_timestamps: Dict[str, Optional[np.ndarray]],
    wrist_camera_joint_keys: Optional[Dict[str, str]] = None,
    sim_log: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Write seq_dir/lowdim.npz with all cameras' intrinsics/extrinsics plus joints, action, language.

    Output keys in lowdim.npz
    --------------------------
    Per-frame keys in each lowdim/<frame>.npz
    ------------------------------------------
    ``intrinsics``           dict[camera_name → (3, 3) float32]  camera matrix K per camera.
    ``extrinsics``           dict[camera_name → (4, 4) float32]  cam-to-left_arm_base.
                             Sim cameras: read from the model and state that rendered the frame.
                             Wrist cameras: computed per-frame via FK + hand-eye calibration.
                             Scene cameras: static calibrated extrinsics.
    ``joints``               (14,) float32  follower joint positions at this frame:
                             [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)].
    ``action``               (26,) float32  FK EE poses computed from commanded joint positions
                             (follower_*_joint_cmd), in the left_arm_base frame:
                             [l_pos(3), l_rot9(9), l_gripper(1), r_pos(3), r_rot9(9), r_gripper(1)].
    ``language_task``        str  task name.
    ``language_prompt``      str  task instruction.
    ``table_pose``           (4, 4) float32  table-to-left_arm_base, from calibration
                             ``table.T_base_table``; identity when the calibration has none.

    Sim recordings add, from the snapshot rendered nearest to the frame (see ``_load_sim_log``):
    ``sim_state``            (1 + nq + nv,) float64  MuJoCo ``[time, qpos, qvel]``; replay it in
                             the episode's ``sim_model.xml`` to re-render from any camera.
    ``object_poses``         dict[str → (4, 4) float32]  task objects in the left_arm_base frame.
    ``subtasks``             dict[str → bool]  BDDL subtask predicates, in task order.
    ``success``              bool  task goal satisfied.
    """

    # ── reference timestamp grid for interpolating robot data ─────────────
    # Prefer wall-clock timestamps; fall back to any available camera timestamps.
    ref_ts: Optional[np.ndarray] = None
    for name in cameras:
        ts = cam_timestamps.get(name)
        if ts is not None and len(ts) == n_frames and int(ts[0]) > _WALL_CLOCK_MIN_NS:
            ref_ts = ts.astype(np.int64)
            break
    if ref_ts is None:
        for name in cameras:
            ts = cam_timestamps.get(name)
            if ts is not None and len(ts) == n_frames:
                ref_ts = ts.astype(np.int64)
                break

    sim_ref_ts = ref_ts.copy() if ref_ts is not None else None

    robot_ts: Optional[np.ndarray] = None
    if robot_data is not None and n_frames > 0:
        robot_ts_raw = robot_data.get("timestamps")
        if ref_ts is not None and robot_ts_raw is not None:
            robot_ts_i64 = np.asarray(robot_ts_raw, dtype=np.int64)
            ref_ts_i64 = np.asarray(ref_ts, dtype=np.int64)
            if ref_ts_i64[0] < robot_ts_i64[0] or ref_ts_i64[-1] > robot_ts_i64[-1]:
                raise ConversionError(
                    "camera timestamp grid extends beyond robot telemetry; refusing to extrapolate"
                )
            # Subtract a shared origin before float conversion. This preserves
            # sub-millisecond precision for Unix-epoch nanosecond timestamps.
            origin = int(robot_ts_i64[0])
            robot_ts = (robot_ts_i64 - origin).astype(np.float64)
            ref_ts = (ref_ts_i64 - origin).astype(np.float64)
        else:
            # Legacy: uniform linspace over the recording duration.
            if robot_ts_raw is not None:
                robot_ts_i64 = np.asarray(robot_ts_raw, dtype=np.int64)
                robot_ts = (robot_ts_i64 - int(robot_ts_i64[0])).astype(np.float64)
                duration = float(robot_ts[-1])
            else:
                duration = float(rec_meta.get("duration_s", 1.0)) * 1e9
            ref_ts = np.linspace(0.0, duration, n_frames, endpoint=True)

    def interp_to_cam(key: str) -> Optional[np.ndarray]:
        if robot_data is None or robot_ts is None or ref_ts is None:
            return None
        arr = robot_data.get(key)
        if arr is None:
            return None
        if len(arr) != len(robot_ts):
            raise ConversionError(
                f"robot field {key} has {len(arr)} rows for {len(robot_ts)} timestamps"
            )
        if arr.ndim == 1:
            arr = arr[:, None]
        return np.stack(
            [np.interp(ref_ts, robot_ts, arr[:, d]) for d in range(arr.shape[1])],
            axis=1,
        ).astype(np.float32)

    def _fk(kin: Any, q: np.ndarray) -> np.ndarray:
        """Call FK, padding q with zeros to the model's nq if needed."""
        nq = kin._configuration.model.nq
        if len(q) < nq:
            q_full = np.zeros(nq, dtype=np.float32)
            q_full[: len(q)] = q
            q = q_full
        return kin.fk(q)

    # ── intrinsics: intrinsics_<camera> → (4,) per camera ────────────────
    intrinsics: Dict[str, np.ndarray] = {}
    for name in cameras:
        info = camera_infos.get(name)
        if not info:
            continue
        flip = name in flip_cameras
        cx, cy = info["cx"], info["cy"]
        if flip:
            cx = (info["width"] - 1) - cx
            cy = (info["height"] - 1) - cy
        intrinsics[name] = np.array(
            [[info["fx"], 0.0, cx], [0.0, info["fy"], cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    # ── extrinsics: extrinsics_<camera> → (N, 4, 4) per camera ──────────
    kin = None  # lazily loaded
    extrinsics: Dict[str, np.ndarray] = {}
    for name in cameras:
        # A rendered sim camera brought its own pose, read from the state that drew its pixels.
        poses_path = seq_dir / "rgb" / name / _POSES_FILE
        if poses_path.exists():
            poses = np.load(str(poses_path))
            if len(poses) != n_frames:
                raise ConversionError(
                    f"{name}: {len(poses)} rendered camera poses for {n_frames} frames"
                )
            extrinsics[name] = poses
            continue

        flip = name in flip_cameras
        static_ext = np.eye(4, dtype=np.float32)
        he = None
        if calib and "cameras" in calib and name in calib["cameras"]:
            cam_calib = calib["cameras"][name]
            if "hand_eye_calibration" in cam_calib:
                he = cam_calib["hand_eye_calibration"]
            elif "extrinsics" in cam_calib:
                ext = cam_calib["extrinsics"]
                if (
                    ext.get("success")
                    and "rotation_matrix" in ext
                    and "translation_vector" in ext
                ):
                    static_ext[:3, :3] = np.array(
                        ext["rotation_matrix"], dtype=np.float32
                    )
                    static_ext[:3, 3] = np.array(
                        ext["translation_vector"], dtype=np.float32
                    ).flatten()

        per_frame_ext: Optional[np.ndarray] = None
        joint_key = (wrist_camera_joint_keys or {}).get(name)
        if (
            joint_key
            and he is not None
            and he.get("success")
            and robot_data is not None
            and robot_ts is not None
            and ref_ts is not None
        ):
            T_cam2ee = np.eye(4, dtype=np.float32)
            T_cam2ee[:3, :3] = np.array(he["rotation_matrix"], dtype=np.float32)
            T_cam2ee[:3, 3] = np.array(
                he["translation_vector"], dtype=np.float32
            ).flatten()
            if flip:
                T_cam2ee = T_cam2ee @ _R_FLIP_180

            arm_joints_raw = robot_data.get(joint_key)
            if arm_joints_raw is not None:
                arm_joints = np.stack(
                    [
                        np.interp(ref_ts, robot_ts, arm_joints_raw[:, d])
                        for d in range(arm_joints_raw.shape[1])
                    ],
                    axis=1,
                ).astype(np.float32)
                if kin is None:
                    kin = _get_kinematics()
                is_right = joint_key.startswith("follower_r")
                per_frame_ext = np.stack(
                    [
                        (
                            right_base_to_left_base
                            @ _fk(kin, arm_joints[i]).astype(np.float32)
                            @ T_cam2ee
                        )
                        if (is_right and right_base_to_left_base is not None)
                        else (_fk(kin, arm_joints[i]).astype(np.float32) @ T_cam2ee)
                        for i in range(n_frames)
                    ]
                )  # (N, 4, 4)

        extrinsics[name] = (
            per_frame_ext
            if per_frame_ext is not None
            else np.tile(static_ext[None], (n_frames, 1, 1))
        )

    # ── joints: (N, 14) from follower_{l,r}_joint_pos_7d ─────────────────
    joints_parts = [
        p
        for p in (
            interp_to_cam("follower_l_joint_pos_7d"),
            interp_to_cam("follower_r_joint_pos_7d"),
        )
        if p is not None
    ]
    joints: Optional[np.ndarray] = (
        np.concatenate(joints_parts, axis=1) if joints_parts else None
    )

    # ── action: (N, 26) FK EE poses + gripper ────────────────────────────
    # Layout: [l_pos(3), l_rot9(9), l_grip(1), r_pos(3), r_rot9(9), r_grip(1)]
    action: Optional[np.ndarray] = None
    if robot_data is not None and robot_ts is not None and ref_ts is not None:
        if kin is None:
            kin = _get_kinematics()
        action_parts = []

        # follower_*_joint_cmd is 7-DOF: arm joints (6) + gripper (1)
        # Raise an error if the arm was recorded but cmd data is missing.
        for _arm_key, _cmd_key in (
            ("follower_l_joint_pos", "follower_l_joint_cmd"),
            ("follower_r_joint_pos", "follower_r_joint_cmd"),
        ):
            if _arm_key in robot_data and _cmd_key not in robot_data:
                raise ValueError(
                    f"robot_data.npz contains '{_arm_key}' but is missing "
                    f"'{_cmd_key}'. The recording has no commanded poses and "
                    "cannot be converted. Re-record the episode."
                )

        l_cmd = interp_to_cam("follower_l_joint_cmd")
        if l_cmd is not None:
            l_poses = np.stack(
                [
                    np.concatenate([T[:3, 3], T[:3, :3].flatten()]).astype(np.float32)
                    for T in (_fk(kin, l_cmd[i, :6]) for i in range(n_frames))
                ]
            )  # (N, 12)
            action_parts.append(l_poses)
            action_parts.append(l_cmd[:, 6:7])  # gripper

        r_cmd = interp_to_cam("follower_r_joint_cmd")
        if r_cmd is not None:
            r_poses = np.stack(
                [
                    np.concatenate([T[:3, 3], T[:3, :3].flatten()]).astype(np.float32)
                    for T in (_fk(kin, r_cmd[i, :6]) for i in range(n_frames))
                ]
            )  # (N, 12)  — in right-arm base frame
            action_parts.append(r_poses)
            action_parts.append(r_cmd[:, 6:7])  # gripper

        if action_parts:
            action = np.concatenate(action_parts, axis=1)

    # ── actual_poses: (N, 26) FK EE poses from actual joint positions ─────
    # Same layout as action: [l_pos(3), l_rot9(9), l_grip(1), r_pos(3), r_rot9(9), r_grip(1)]
    actual_poses: Optional[np.ndarray] = None
    if joints is not None:
        if kin is None:
            kin = _get_kinematics()
        actual_parts = []

        l_pos = interp_to_cam("follower_l_joint_pos_7d")
        if l_pos is not None:
            l_actual_poses = np.stack(
                [
                    np.concatenate([T[:3, 3], T[:3, :3].flatten()]).astype(np.float32)
                    for T in (_fk(kin, l_pos[i, :6]) for i in range(n_frames))
                ]
            )  # (N, 12)
            actual_parts.append(l_actual_poses)
            actual_parts.append(l_pos[:, 6:7])  # gripper

        r_pos = interp_to_cam("follower_r_joint_pos_7d")
        if r_pos is not None:
            r_actual_poses = np.stack(
                [
                    np.concatenate([T[:3, 3], T[:3, :3].flatten()]).astype(np.float32)
                    for T in (_fk(kin, r_pos[i, :6]) for i in range(n_frames))
                ]
            )  # (N, 12)  — in right-arm base frame
            actual_parts.append(r_actual_poses)
            actual_parts.append(r_pos[:, 6:7])  # gripper

        if actual_parts:
            actual_poses = np.concatenate(actual_parts, axis=1)

    # ── action_joints: (N, 14) commanded joint positions ─────────────────
    # Layout: [l_arm(6), l_gripper(1), r_arm(6), r_gripper(1)]
    action_joints_parts = [
        p
        for p in (
            interp_to_cam("follower_l_joint_cmd"),
            interp_to_cam("follower_r_joint_cmd"),
        )
        if p is not None
    ]
    action_joints: Optional[np.ndarray] = (
        np.concatenate(action_joints_parts, axis=1) if action_joints_parts else None
    )

    # ── table: static table-to-base pose ─────────────────────────────────
    table_pose = np.eye(4, dtype=np.float32)
    if calib and "table" in calib:
        table_pose = np.array(calib["table"]["T_base_table"], dtype=np.float32)

    # ── sim: nearest snapshot per frame (exact for the reference camera) ──
    sim_idx: Optional[np.ndarray] = None
    if (
        sim_log is not None
        and sim_ref_ts is not None
        and int(sim_ref_ts[0]) > _WALL_CLOCK_MIN_NS
    ):
        t = sim_log["t_ns"]
        j = np.clip(np.searchsorted(t, sim_ref_ts), 1, len(t) - 1)
        sim_idx = np.where(sim_ref_ts - t[j - 1] <= t[j] - sim_ref_ts, j - 1, j)
        gap_ms = np.abs(t[sim_idx] - sim_ref_ts).max() / 1e6
        print(f"  sim state: nearest snapshot at most {gap_ms:.1f} ms from its frame")

    # ── language ──────────────────────────────────────────────────────────
    language_task = np.array(rec_meta.get("task_name", ""), dtype=object)
    language_prompt = np.array(rec_meta.get("task_instruction", ""), dtype=object)

    # ── write per-frame files into lowdim/ ────────────────────────────────
    lowdim_dir = seq_dir / "lowdim"
    lowdim_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        frame_data: Dict[str, Any] = {
            "intrinsics": intrinsics,
            "extrinsics": {name: ext_arr[i] for name, ext_arr in extrinsics.items()},
            "table_pose": table_pose,
        }
        if sim_idx is not None:
            j = sim_idx[i]
            frame_data["sim_state"] = sim_log["state"][j]
            frame_data["object_poses"] = dict(
                zip(
                    sim_log["objects"].tolist(),
                    sim_log["object_poses"][j].astype(np.float32),
                )
            )
            frame_data["subtasks"] = dict(
                zip(sim_log["subtasks"].tolist(), sim_log["subtask_done"][j].tolist())
            )
            frame_data["success"] = bool(sim_log["success"][j])
        if joints is not None:
            frame_data["joints"] = joints[i]
        if action is not None:
            frame_data["action"] = action[i]
        if actual_poses is not None:
            frame_data["actual_poses"] = actual_poses[i]
        if action_joints is not None:
            frame_data["action_joints"] = action_joints[i]
        frame_data["language_task"] = language_task
        frame_data["language_prompt"] = language_prompt
        if right_base_to_left_base is not None:
            frame_data["T_left_from_right"] = right_base_to_left_base
        with open(lowdim_dir / f"{i:010d}.pkl", "wb") as f:
            pickle.dump(frame_data, f)


# ---------------------------------------------------------------------------
# Dataset-level metadata helpers
# ---------------------------------------------------------------------------


def _build_sequence_metadata(
    seq_dir: Path,
    cameras: List[str],
    frame_counts: Dict[str, int],
    rec_meta: dict,
    camera_infos: Dict[str, Optional[dict]],
) -> None:
    """Write metadata.json inside the sequence directory."""
    resolutions = {
        cam: [info.get("height"), info.get("width")]
        for cam, info in camera_infos.items()
        if info
    }
    unique_res = list({tuple(v) for v in resolutions.values()})
    if len(unique_res) == 1:
        resolution: object = list(unique_res[0])
    elif unique_res:
        resolution = {cam: v for cam, v in resolutions.items()}
    else:
        resolution = None

    has_depth = any(any((seq_dir / "depth" / cam).glob("*.npz")) for cam in cameras)
    meta = {
        "info": {
            "name": rec_meta.get("task_name", ""),
            "raw_id": rec_meta.get("timestamp", ""),
            "tags": ["robotics"],
        },
        "labels": ["rgb", *(["depth"] if has_depth else []), "action", "language"],
        "cameras": cameras,
        "resolution": resolution,
        "framerate": rec_meta.get("camera_fps", 30),
        "language": {
            "task": rec_meta.get("task_name", ""),
            "prompt": [rec_meta.get("task_instruction", "")],
        },
        "num_frames": max(frame_counts.values()) if frame_counts else 0,
        "rgb": {"extension": "png"},
        **(
            {"depth": {"extension": "npz", "sparse": False, "metric": True}}
            if has_depth
            else {}
        ),
        "intrinsics": {"model": "pinhole"},
        "extrinsics": {"transform": "cam2world", "metric": True},
        "action": {"format": "joint_cmd", "dims": 14},
        "control": rec_meta.get("control", "leader"),
    }

    with open(seq_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def _build_split(rec_path: Path, frame_counts: Dict[str, int]) -> None:
    n_frames = max(frame_counts.values()) if frame_counts else 0
    split = {
        "filters": {},
        "size": {"n_seqs": 1, "n_samples": 1, "n_frames": n_frames},
        "files": {_SEQUENCE_NAME: n_frames},
    }
    with open(rec_path / "split_all.json", "w") as f:
        json.dump(split, f, indent=2)


# ---------------------------------------------------------------------------
# Interactive task selection
# ---------------------------------------------------------------------------


def select_tasks(data_dir: str = "data") -> List[str]:
    """Use fzf to select one or more task directories (Tab to multi-select)."""
    base = Path(data_dir) / "raw"
    task_dirs = sorted(
        (
            d
            for d in base.iterdir()
            if d.is_dir()
            and any((sub / "cameras").exists() for sub in d.iterdir() if sub.is_dir())
        ),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )

    if not task_dirs:
        print(f"No tasks found in {base}")
        sys.exit(1)

    _ALL_LABEL = "*** ALL TASKS ***"
    labels = {
        f"{d.name}  ({sum(1 for s in d.iterdir() if s.is_dir() and (s / 'cameras').exists())} recording(s))": d
        for d in task_dirs
    }

    from raiden.utils import fzf_select

    choices = [_ALL_LABEL] + list(labels)
    selected = fzf_select(
        choices,
        prompt="Convert task(s)> ",
        multi=True,
        header="Tab: toggle select  |  Enter: confirm  |  Select '*** ALL TASKS ***' to convert everything",
    )

    if _ALL_LABEL in selected:
        return [str(d) for d in task_dirs]
    return [str(labels[s]) for s in selected]


# ---------------------------------------------------------------------------
# Main convert function
# ---------------------------------------------------------------------------


def convert_recording(
    recording_dir: str,
    episode_dir: Optional[str] = None,
    stereo_method: str = "zed",
    ffs_scale: float = 1.0,
    ffs_iters: int = 8,
    tri_stereo_variant: str = "c64",
    reconvert: bool = False,
) -> Dict[str, int]:
    """Convert a recording directory to UnifiedDataset format.

    When *episode_dir* is provided the sequence data is written there instead
    of ``recording_dir/0000``, and dataset-level files (split_all.json,
    metadata_shared.json, calibration_results.json) are skipped so the caller
    (e.g. :func:`convert_task`) can aggregate them.

    Returns per-camera frame counts, or an empty dict if already converted.
    """
    rec_path = Path(recording_dir)
    if not rec_path.exists():
        print(f"Error: directory not found: {rec_path}")
        sys.exit(1)

    cameras_path = rec_path / "cameras"
    if not cameras_path.exists():
        print(f"Error: no cameras/ sub-directory in {rec_path}")
        sys.exit(1)

    svo2_files = sorted(cameras_path.glob("*.svo2"))
    bag_files = sorted(cameras_path.glob("*.bag"))
    simrec_dirs = sorted(p for p in cameras_path.glob("*.simrec") if p.is_dir())

    if not svo2_files and not bag_files and not simrec_dirs:
        print("No .svo2 / .bag files or .simrec directories found in cameras/")
        sys.exit(1)

    seq_dir = Path(episode_dir) if episode_dir else rec_path / _SEQUENCE_NAME
    if (seq_dir / "rgb").exists() and not reconvert:
        print(f"Already converted: {seq_dir}")
        return {}
    elif (seq_dir / "rgb").exists() and reconvert:
        shutil.rmtree(seq_dir)
        print(f"Removed existing output: {seq_dir}")

    seq_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nConverting to UnifiedDataset: {rec_path.name}")
    print(
        f"  Found {len(svo2_files)} SVO2 file(s), {len(bag_files)} bag file(s), "
        f"{len(simrec_dirs)} sim recording(s)\n"
    )

    # ── load supporting data ──────────────────────────────────────────────
    rec_meta: dict = {}
    meta_path = rec_path / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            rec_meta = json.load(f)

    robot_data: Optional[Dict[str, np.ndarray]] = None
    robot_path = rec_path / "robot_data.npz"
    if robot_path.exists():
        npz = np.load(robot_path, allow_pickle=False)
        robot_data = {k: npz[k] for k in npz.files}

    calib: Optional[dict] = None
    calib_path = rec_path / "calibration_results.json"
    if calib_path.exists():
        with open(calib_path) as f:
            calib = json.load(f)

    # "right_base_to_left_base" maps left_arm_base → right_arm_base (despite the name).
    # Invert to get the transform that brings right_arm_base points into left_arm_base.
    T_left_base_from_right_base: Optional[np.ndarray] = None
    if calib and "bimanual_transform" in calib:
        mat = calib["bimanual_transform"].get("right_base_to_left_base")
        if mat is not None:
            T_left_base_from_right_base = np.linalg.inv(np.array(mat, dtype=np.float32))
    if T_left_base_from_right_base is None:
        print(
            "  Warning: bimanual_transform not found in calibration; right wrist extrinsics will be in right_arm_base frame"
        )

    # ── extract frames ────────────────────────────────────────────────────
    frame_counts: Dict[str, int] = {}
    camera_infos: Dict[str, Optional[dict]] = {}
    cam_timestamps: Dict[str, Optional[np.ndarray]] = {}

    # SVO2 cameras are extracted together with cross-camera temporal alignment.
    # Skip only when every SVO2 camera already has its rgb dir AND timestamps.npy
    # (the latter is written by _extract_svo2_synchronized).
    svo2_names = [p.stem for p in svo2_files]
    svo2_all_done = svo2_names and all(
        (seq_dir / "rgb" / name).exists()
        and (seq_dir / "rgb" / name / "timestamps.npy").exists()
        for name in svo2_names
    )

    if svo2_all_done:
        for name in svo2_names:
            rgb_dir = seq_dir / "rgb" / name
            n = len(list(rgb_dir.glob("*.png")))
            print(f"  Skipping {name} (already extracted, {n} frames)")
            frame_counts[name] = n
            camera_infos[name] = None
            cam_timestamps[name] = np.load(str(rgb_dir / "timestamps.npy"))
    elif svo2_files:
        # Pre-scan to determine the min frame count cap.
        pre_counts: Dict[str, int] = {}
        for svo_path in svo2_files:
            name = svo_path.stem
            rgb_dir_check = seq_dir / "rgb" / name
            if rgb_dir_check.exists():
                pre_counts[name] = len(list(rgb_dir_check.glob("*.png")))
            else:
                print(f"  Pre-scanning {svo_path.name} ...")
                pre_counts[name] = _count_svo2_frames(svo_path)
                print(f"    {pre_counts[name]} frames")
        max_frames_svo2 = min(pre_counts.values()) if pre_counts else None
        if max_frames_svo2 is not None:
            print(
                f"  Capping SVO2 extraction at {max_frames_svo2} frames (min across cameras)\n"
            )

        print(f"  Extracting {len(svo2_files)} SVO2 file(s) in sync ...")
        if stereo_method != "zed":
            extra = ""
            if stereo_method == "ffs" and ffs_scale != 1.0:
                extra = f" (scale={ffs_scale})"
            elif stereo_method == "tri_stereo":
                extra = f" (variant={tri_stereo_variant})"
            print(f"  Stereo method: {stereo_method}{extra}")
        sync_results = _extract_svo2_synchronized(
            svo_paths=svo2_files,
            names=svo2_names,
            rgb_dirs=[seq_dir / "rgb" / n for n in svo2_names],
            depth_dirs=[seq_dir / "depth" / n for n in svo2_names],
            flips=[n in _FLIP_CAMERAS for n in svo2_names],
            max_frames=max_frames_svo2,
            stereo_method=stereo_method,
            ffs_scale=ffs_scale,
            ffs_iters=ffs_iters,
            tri_stereo_variant=tri_stereo_variant,
        )
        for name, (ts_arr, info) in sync_results.items():
            frame_counts[name] = len(ts_arr)
            camera_infos[name] = info
            cam_timestamps[name] = ts_arr
            print(f"  ✓ {name}: {len(ts_arr)} frames")

    # Bag files are extracted independently then aligned to ZED by timestamp.
    bag_max = min(frame_counts.values()) if frame_counts else None
    rs_offsets = rec_meta.get("realsense_clock_offsets", {})
    for bag_path in bag_files:
        name = bag_path.stem
        rgb_dir = seq_dir / "rgb" / name
        depth_dir = seq_dir / "depth" / name

        if rgb_dir.exists():
            n = len(list(rgb_dir.glob("*.png")))
            print(f"  Skipping {name} (already extracted, {n} frames)")
            frame_counts[name] = n
            camera_infos[name] = None
            ts_path = rgb_dir / "timestamps.npy"
            if ts_path.exists():
                ts_arr = np.load(str(ts_path))
                cam_timestamps[name] = _apply_rs_clock_offset(
                    ts_arr, rs_offsets.get(name)
                )
            else:
                cam_timestamps[name] = None
            continue

        flip = name in _FLIP_CAMERAS
        print(f"  Extracting {bag_path.name}" + (" (flipped)" if flip else ""))
        ts_arr, info = _extract_bag(
            bag_path, rgb_dir, depth_dir, flip=flip, max_frames=bag_max
        )
        frame_counts[name] = len(ts_arr)
        camera_infos[name] = info

        # With global_time_enabled, RealSense timestamps are wall-clock (same as
        # ZED).  Old recordings without it carry a clock offset in metadata;
        # _apply_rs_clock_offset applies it only when the timestamps need it.
        if len(ts_arr) > 0:
            cam_timestamps[name] = _apply_rs_clock_offset(ts_arr, rs_offsets.get(name))
        else:
            cam_timestamps[name] = None
        print(f"  ✓ {name}: {len(ts_arr)} frames\n")

    # Simulated cameras store the state of each frame, not its pixels; render them.
    sim_base = _sim_base_frame(simrec_dirs, rec_path / "sim_model.xml", calib)
    for sim_dir in simrec_dirs:
        name = sim_dir.stem
        rgb_dir = seq_dir / "rgb" / name
        if rgb_dir.exists():
            n = len(list(rgb_dir.glob("*.png")))
            print(f"  Skipping {name} (already extracted, {n} frames)")
            frame_counts[name] = n
            camera_infos[name] = None
            ts_path = rgb_dir / "timestamps.npy"
            cam_timestamps[name] = np.load(str(ts_path)) if ts_path.exists() else None
            continue
        print(f"  Extracting {sim_dir.name} (sim)")
        ts_arr, info = _extract_simrec(
            sim_dir, rgb_dir, rec_path / "sim_model.xml", sim_base
        )
        frame_counts[name] = len(ts_arr)
        camera_infos[name] = info
        cam_timestamps[name] = ts_arr if len(ts_arr) > 0 else None
        print(f"  ✓ {name}: {len(ts_arr)} frames\n")

    # Resample every camera onto one grid bounded by the camera overlap and
    # robot telemetry, then verify the residual cross-camera timestamp error.
    cam_timestamps, frame_counts = _align_cameras_by_timestamp(
        seq_dir,
        cam_timestamps,
        frame_counts,
        camera_start_times_ns=rec_meta.get("camera_start_times_ns"),
        camera_fps=float(rec_meta.get("camera_fps", 30)),
        robot_timestamps=(
            robot_data.get("timestamps") if robot_data is not None else None
        ),
    )

    # Trim all cameras to the same (minimum) frame count.  Frames are indexed
    # by position after alignment, so a camera that dropped most of its frames
    # (e.g. RealSense on a USB 2 link) would silently pair frame i of one
    # camera with a different moment in time on the other — refuse instead.
    if frame_counts:
        n_max = max(frame_counts.values())
        n_min = min(frame_counts.values())
        if n_max > 0 and n_min < 0.9 * n_max:
            counts = ", ".join(f"{k}={v}" for k, v in frame_counts.items())
            raise ConversionError(
                f"camera frame counts differ by more than 10% ({counts}); "
                "a camera dropped frames — check the USB link and re-record"
            )
        frame_counts = {k: n_min for k in frame_counts}
        cam_timestamps = {
            k: (ts[:n_min] if ts is not None else None)
            for k, ts in cam_timestamps.items()
        }
        # Delete any trailing image/depth files beyond n_min and update timestamps.npy
        # so that rgb, depth, and lowdim always have exactly the same frame count.
        for name, ts in cam_timestamps.items():
            rgb_dir = seq_dir / "rgb" / name
            depth_dir = seq_dir / "depth" / name
            for idx in range(n_min, n_min + 100):  # clean up any trailing extras
                for d, ext in ((rgb_dir, _IMG_EXT), (depth_dir, ".npz")):
                    f = d / f"{idx:010d}{ext}"
                    if f.exists():
                        f.unlink()
                    else:
                        break
            poses_path = rgb_dir / _POSES_FILE
            if poses_path.exists():
                np.save(str(poses_path), np.load(str(poses_path))[:n_min])
            if ts is not None:
                np.save(str(rgb_dir / "timestamps.npy"), ts)

    cameras = list(frame_counts.keys())

    # ── wrist camera → joint key mapping from camera config roles ─────────
    cam_cfg = CameraConfig(CAMERA_CONFIG)
    wrist_camera_joint_keys: Dict[str, str] = {
        name: _ROLE_TO_JOINT_KEY[role]
        for name in cameras
        if (role := cam_cfg.get_role(name)) in _ROLE_TO_JOINT_KEY
    }

    # ── lowdim ────────────────────────────────────────────────────────────
    print("\nBuilding lowdim...")
    _build_lowdim(
        seq_dir=seq_dir,
        cameras=cameras,
        n_frames=n_min,
        camera_infos=camera_infos,
        calib=calib,
        robot_data=robot_data,
        rec_meta=rec_meta,
        flip_cameras=_FLIP_CAMERAS,
        right_base_to_left_base=T_left_base_from_right_base,
        cam_timestamps=cam_timestamps,
        wrist_camera_joint_keys=wrist_camera_joint_keys,
        sim_log=_load_sim_log(simrec_dirs),
    )
    print(f"  ✓ lowdim/ ({n_min} frames)")
    if (rec_path / "sim_model.xml").exists():
        shutil.copy(rec_path / "sim_model.xml", seq_dir / "sim_model.xml")
        print("  ✓ sim_model.xml")

    # ── sequence metadata ─────────────────────────────────────────────────
    _build_sequence_metadata(seq_dir, cameras, frame_counts, rec_meta, camera_infos)
    print("  ✓ metadata.json")

    if episode_dir is None:
        # ── dataset-level files ───────────────────────────────────────────
        _build_split(rec_path, frame_counts)
        print("  ✓ split_all.json")

        shutil.copy(seq_dir / "metadata.json", rec_path / "metadata_shared.json")
        print("  ✓ metadata_shared.json")

        if calib_path.exists():
            shutil.copy(calib_path, rec_path / "calibration_results.json")
            print(f"  ✓ calibration_results.json (copied from {calib_path})")
        else:
            print(f"  Warning: calibration_results.json not found at {calib_path}")

    # ── update original metadata ──────────────────────────────────────────
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        meta["converted"] = True
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    print(f"\n✓ UnifiedDataset ready: {seq_dir if episode_dir else rec_path}")
    return frame_counts


def convert_task(
    task_dir: str,
    output_dir: Optional[str] = None,
    stereo_method: str = "zed",
    ffs_scale: float = 1.0,
    ffs_iters: int = 8,
    reconvert: bool = False,
    processed_base: Optional[str] = None,
    tri_stereo_variant: str = "c64",
) -> None:
    """Convert all recordings in a task directory into a single UnifiedDataset.

    Each recording becomes a numbered episode (0000, 0001, …) under *out_base*::

        <output_dir>/<task_name>/
            0000/
            0001/
            ...
            split_all.json
            metadata_shared.json
            calibration_results.json

    *output_dir* defaults to ``<task_parent>/processed_data``.
    """
    task_path = Path(task_dir)
    if not task_path.exists():
        print(f"Error: directory not found: {task_path}")
        sys.exit(1)

    recording_dirs = sorted(
        d for d in task_path.iterdir() if d.is_dir() and (d / "cameras").exists()
    )

    if not recording_dirs:
        print(f"Error: no recording directories found in {task_path}")
        sys.exit(1)

    if output_dir:
        out_base = Path(output_dir) / task_path.name
    elif processed_base:
        out_base = Path(processed_base) / task_path.name
    else:
        out_base = Path("data") / "processed" / task_path.name
    out_base.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(recording_dirs)} recording(s) in {task_path.name}")
    print(f"Output → {out_base}\n")

    episode_frame_counts: Dict[str, int] = {}

    # Lazy import to avoid hard dependency when DB is not set up
    try:
        from raiden.db.database import get_db as _get_db

        _db = _get_db()
    except Exception:
        _db = None

    # Filter recordings: skip failures/pending, and (unless reconvert) already
    # converted ones.  Recordings with no DB entry are treated as unknown and
    # included so directories recorded before the DB was set up are not dropped.
    success_dirs = []
    skipped = 0
    for rec_dir in recording_dirs:
        status = "unknown"
        already_converted = False
        if _db is not None:
            try:
                demo = _db.get_demonstration_by_raw_path(str(rec_dir))
                if demo is not None:
                    status = demo.get("status", "pending")
                    already_converted = bool(demo.get("converted", False))
            except Exception:
                pass
        if status not in ("success", "unknown"):
            print(f"  Skipping {rec_dir.name} (status={status})")
            skipped += 1
        elif already_converted and not reconvert:
            print(
                f"  Skipping {rec_dir.name} (already converted, use --reconvert to force)"
            )
            skipped += 1
        else:
            success_dirs.append(rec_dir)

    if skipped:
        print(f"\nSkipped {skipped} non-success recording(s).")
    if not success_dirs:
        print("No successful recordings to convert.")
        return

    print(f"Converting {len(success_dirs)} successful recording(s)\n")

    failed: List[str] = []
    for i, rec_dir in enumerate(success_dirs):
        episode_name = f"{len(episode_frame_counts):04d}"
        ep_dir = out_base / episode_name
        print(f"[{i + 1}/{len(success_dirs)}] {rec_dir.name} → {episode_name}/")
        try:
            counts = convert_recording(
                str(rec_dir),
                episode_dir=str(ep_dir),
                stereo_method=stereo_method,
                ffs_scale=ffs_scale,
                ffs_iters=ffs_iters,
                tri_stereo_variant=tri_stereo_variant,
                reconvert=reconvert,
            )
        except ConversionError as e:
            print(f"\n  ✗ Skipping {rec_dir.name}: {e}\n")
            shutil.rmtree(ep_dir, ignore_errors=True)
            failed.append(rec_dir.name)
            continue

        if counts:
            episode_frame_counts[episode_name] = max(counts.values())
        else:
            # Already converted — read frame count from existing metadata.
            meta_file = ep_dir / "metadata.json"
            if meta_file.exists():
                with open(meta_file) as f:
                    ep_meta = json.load(f)
                episode_frame_counts[episode_name] = ep_meta.get("num_frames", 0)
            else:
                episode_frame_counts[episode_name] = 0

        # Update Demonstration status in DB
        if _db is not None:
            try:
                demo = _db.get_demonstration_by_raw_path(str(rec_dir))
                if demo is not None:
                    _db.update_demonstration(
                        demo["id"],
                        converted=True,
                        converted_data_path=str(ep_dir),
                    )
            except Exception:
                pass

    if failed:
        print(
            f"\n✗ {len(failed)} recording(s) could not be converted: {', '.join(failed)}"
        )
        print("  Mark them as failures in `rd console` so they are not retried.")
    if not episode_frame_counts:
        return

    # ── combined split_all.json ────────────────────────────────────────────
    total_frames = sum(episode_frame_counts.values())
    n_eps = len(episode_frame_counts)
    split = {
        "filters": {},
        "size": {"n_seqs": n_eps, "n_samples": n_eps, "n_frames": total_frames},
        "files": episode_frame_counts,
    }
    with open(out_base / "split_all.json", "w") as f:
        json.dump(split, f, indent=2)
    print(f"\n✓ split_all.json ({n_eps} episodes, {total_frames} total frames)")

    # ── shared metadata & calibration ─────────────────────────────────────
    first_ep = next(iter(episode_frame_counts))
    first_meta = out_base / first_ep / "metadata.json"
    if first_meta.exists():
        shutil.copy(first_meta, out_base / "metadata_shared.json")
        print("✓ metadata_shared.json")

    first_calib = recording_dirs[0] / "calibration_results.json"
    if first_calib.exists():
        shutil.copy(first_calib, out_base / "calibration_results.json")
        print("✓ calibration_results.json")

    print(f"\n✓ Task dataset ready: {out_base}")
