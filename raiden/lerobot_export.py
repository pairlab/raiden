"""Export converted UnifiedDataset episodes to the LeRobot dataset format.

Input:  ``data/processed/<task>/<episode>/`` directories produced by ``rd convert``.
Output: a LeRobotDataset (codebase v3) at ``<output_dir>/<task>``::

    observation.images.<camera>   video, (H, W, 3) — one stream per camera
    observation.state             float32, measured joint positions + gripper
    observation.ee_pose           float32, measured end-effector pose (FK) + gripper
    action                        float32, commanded joint positions + gripper
    action.ee_pose                float32, commanded end-effector pose (FK) + gripper
    camera.<camera>.intrinsics    float32 (4,), fx, fy, cx, cy of that camera's image
    camera.<camera>.extrinsics    float32 (12,), camera-to-base pose of that camera
    table.pose                    float32 (12,), table-to-base pose (static per episode)
    depth.<camera>                uint16 (H, W), millimetres, 0 = no return (only when recorded)
    task                          language instruction from the recording

Both the joint-space and the end-effector-space versions are stored so the
policy action space can be chosen (or changed) at training time.

Joint layout is ``[l_arm(6), l_gripper(1)]`` for single-arm recordings and
``[l_arm(6), l_gripper(1), r_arm(6), r_gripper(1)]`` for bimanual ones — the
same layout as ``joints`` / ``action_joints`` in the lowdim pickles.
EE-pose layout is ``[l_xyz(3), l_rot9(9), l_gripper(1)]`` (plus the same for
the right arm), i.e. the position and row-major 3×3 rotation of the gripper
site in the (left) arm base frame — the same layout as ``actual_poses`` /
``action`` in the lowdim pickles.

Camera calibration is stored per frame: a wrist camera moves every frame and a
scene camera can move between sessions. ``intrinsics`` is the pinhole matrix of
the exported (cropped) image. ``extrinsics`` is the pose of the camera's OpenCV
optical frame (x right, y down, z forward) in the left arm base frame, in the
EE-pose layout without the gripper, ``[x, y, z, r00..r22]``: a base-frame point
``p`` lands at pixel ``K @ R.T @ (p - t)``. Both come from the lowdim
``intrinsics`` / ``extrinsics`` dicts; a camera that had no calibration at
conversion time carries the identity pose, which the export reports.
``table.pose`` is the table frame (origin at the centre of the table top, axes
along the base axes) in the same layout, from the lowdim ``table_pose``; it too
is the identity when the recording's calibration had none.
The camera features sit outside ``observation.*`` on purpose: LeRobot turns
every ``observation.*`` float feature into a policy state input, and camera
poses should reach a policy only when it asks for them by key.

Depth is exported when the converted episodes have it (``depth/<camera>/*.npz``),
for point-cloud policies such as Adapt3R. It is a plain uint16 array, not an
image or video feature: LeRobot images are 3-channel uint8 and its videos are
lossy, and either would destroy the metric values. It sits outside
``observation.*`` for the same reason as the camera poses. LeRobot's per-feature
stats are skipped for it: they would be computed per pixel, which is slow and
bloats ``stats.json``, and depth is not normalised with them.

Requires the ``lerobot`` optional extra (``uv sync --extra lerobot``).
"""

import json
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

_DEPTH_PREFIX = "depth."

_ARM_JOINT_NAMES = [f"joint_{i}" for i in range(6)] + ["gripper"]
_ARM_EE_NAMES = (
    ["x", "y", "z"] + [f"r{i}{j}" for i in range(3) for j in range(3)] + ["gripper"]
)
_CAMERA_INTRINSICS_NAMES = ["fx", "fy", "cx", "cy"]
_POSE_NAMES = _ARM_EE_NAMES[:-1]  # EE-pose layout without the gripper
# What the converter writes for a camera or table pose it has no calibration for.
_IDENTITY_POSE = np.concatenate([np.zeros(3), np.eye(3).reshape(-1)]).astype(np.float32)

# lowdim key -> (feature key, per-arm names)
_LOWDIM_FEATURES = {
    "joints": ("observation.state", _ARM_JOINT_NAMES),
    "actual_poses": ("observation.ee_pose", _ARM_EE_NAMES),
    "action_joints": ("action", _ARM_JOINT_NAMES),
    "action": ("action.ee_pose", _ARM_EE_NAMES),
}


def _dim_names(dim: int, per_arm: List[str]) -> List[str]:
    n = len(per_arm)
    if dim == n:
        return [f"l_{x}" for x in per_arm]
    if dim == 2 * n:
        return [f"l_{x}" for x in per_arm] + [f"r_{x}" for x in per_arm]
    return [f"dim_{i}" for i in range(dim)]


def _pose_row(T) -> np.ndarray:
    """(4, 4) pose -> ``[x, y, z, r00..r22]``."""
    T = np.asarray(T, dtype=np.float32)
    return np.concatenate([T[:3, 3], T[:3, :3].reshape(-1)])


def _load_episode_lowdim(
    ep_dir: Path, n_frames: int, cameras: List[str]
) -> Dict[str, np.ndarray]:
    """Return {feature key: (n_frames, dim) float32} for every entry of _LOWDIM_FEATURES
    plus ``camera.<camera>.intrinsics`` / ``.extrinsics`` for every camera and ``table.pose``."""
    rows: Dict[str, list] = {feat: [] for feat, _ in _LOWDIM_FEATURES.values()}
    for cam in cameras:
        rows[f"camera.{cam}.intrinsics"] = []
        rows[f"camera.{cam}.extrinsics"] = []
    rows["table.pose"] = []
    for i in range(n_frames):
        with open(ep_dir / "lowdim" / f"{i:010d}.pkl", "rb") as f:
            fd = pickle.load(f)
        for key, (feat, _) in _LOWDIM_FEATURES.items():
            if key not in fd:
                raise ValueError(
                    f"{ep_dir.name}: lowdim frame {i} lacks '{key}' — "
                    "re-run `rd convert --reconvert`"
                )
            rows[feat].append(np.asarray(fd[key], dtype=np.float32).reshape(-1))
        for cam in cameras:
            K = fd.get("intrinsics", {}).get(cam)
            T = fd.get("extrinsics", {}).get(cam)
            if K is None or T is None:
                raise ValueError(
                    f"{ep_dir.name}: lowdim frame {i} lacks the calibration of '{cam}' — "
                    "re-run `rd convert --reconvert`"
                )
            K = np.asarray(K, dtype=np.float32)
            rows[f"camera.{cam}.intrinsics"].append(
                np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32)
            )
            rows[f"camera.{cam}.extrinsics"].append(_pose_row(T))
        if "table_pose" not in fd:
            raise ValueError(
                f"{ep_dir.name}: lowdim frame {i} lacks 'table_pose' — re-run `rd convert --reconvert`"
            )
        rows["table.pose"].append(_pose_row(fd["table_pose"]))
    return {feat: np.stack(v) for feat, v in rows.items()}


def _placeholder_poses(lowdim: Dict[str, np.ndarray], keys: List[str]) -> List[str]:
    """Pose features that are the converter's identity placeholder in every frame."""
    return [k for k in keys if np.allclose(lowdim[k], _IDENTITY_POSE)]


def _rotation_skew(extrinsics: np.ndarray) -> float:
    """max |R^T R - I| over an episode's extrinsics rows: ~1e-7 for proper rotations."""
    R = extrinsics[:, 3:].reshape(-1, 3, 3)
    return float(np.abs(R.transpose(0, 2, 1) @ R - np.eye(3)).max())


def _skip_depth_stats():
    """Make LeRobot compute episode stats without the depth features.

    Returns a callable that restores the original function.
    """
    import lerobot.datasets.lerobot_dataset as lerobot_dataset

    original = lerobot_dataset.compute_episode_stats

    def without_depth(episode_data, features):
        return original(
            {k: v for k, v in episode_data.items() if not k.startswith(_DEPTH_PREFIX)},
            {k: v for k, v in features.items() if not k.startswith(_DEPTH_PREFIX)},
        )

    lerobot_dataset.compute_episode_stats = without_depth
    return lambda: setattr(lerobot_dataset, "compute_episode_stats", original)


def export_task_to_lerobot(
    task_dir: Path,
    episode_dirs: List[Path],
    output_dir: Path,
    repo_id: Optional[str] = None,
    vcodec: str = "libsvtav1",
    image_writer_threads: int = 4,
    overwrite: bool = False,
) -> Path:
    """Write every episode in *episode_dirs* into one LeRobotDataset.

    Returns the dataset root directory.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    task_name = task_dir.name
    repo_id = repo_id or f"raiden/{task_name}"
    root = Path(output_dir) / task_name

    if root.exists():
        if not overwrite:
            raise FileExistsError(
                f"{root} already exists (use --overwrite to replace it)"
            )
        shutil.rmtree(root)

    episode_dirs = sorted(d for d in episode_dirs if (d / "metadata.json").exists())
    if not episode_dirs:
        raise FileNotFoundError(f"No converted episodes in {task_dir}")

    # ── infer features from the first episode ─────────────────────────────
    with open(episode_dirs[0] / "metadata.json") as f:
        meta0 = json.load(f)
    cameras: List[str] = meta0["cameras"]
    fps = int(meta0.get("framerate", 30))

    features: Dict[str, dict] = {}
    for cam in cameras:
        img = cv2.imread(str(episode_dirs[0] / "rgb" / cam / "0000000000.png"))
        if img is None:
            raise FileNotFoundError(f"No frames for camera {cam} in {episode_dirs[0]}")
        h, w = img.shape[:2]
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        }
    lowdim0 = _load_episode_lowdim(episode_dirs[0], 1, cameras)
    for _key, (feat, per_arm) in _LOWDIM_FEATURES.items():
        dim = int(lowdim0[feat].shape[1])
        features[feat] = {
            "dtype": "float32",
            "shape": (dim,),
            "names": _dim_names(dim, per_arm),
        }
    for cam in cameras:
        features[f"camera.{cam}.intrinsics"] = {
            "dtype": "float32",
            "shape": (len(_CAMERA_INTRINSICS_NAMES),),
            "names": _CAMERA_INTRINSICS_NAMES,
        }
        features[f"camera.{cam}.extrinsics"] = {
            "dtype": "float32",
            "shape": (len(_POSE_NAMES),),
            "names": _POSE_NAMES,
        }
    features["table.pose"] = {
        "dtype": "float32",
        "shape": (len(_POSE_NAMES),),
        "names": _POSE_NAMES,
    }
    depth_cameras = [
        cam
        for cam in cameras
        if (episode_dirs[0] / "depth" / cam / "0000000000.npz").exists()
    ]
    for cam in depth_cameras:
        depth0 = np.load(episode_dirs[0] / "depth" / cam / "0000000000.npz")["depth"]
        features[f"{_DEPTH_PREFIX}{cam}"] = {
            "dtype": "uint16",
            "shape": depth0.shape,
            "names": ["height", "width"],
        }
    pose_keys = [f"camera.{cam}.extrinsics" for cam in cameras] + ["table.pose"]

    print(f"Exporting {len(episode_dirs)} episode(s) of '{task_name}' → {root}")
    print(f"  repo_id={repo_id}  fps={fps}  cameras={cameras}")
    dims = ", ".join(
        f"{k}={v['shape'][0]}" for k, v in features.items() if v["dtype"] == "float32"
    )
    print(f"  {dims}  vcodec={vcodec}")
    print(f"  depth: {depth_cameras or 'none'}\n")

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=root,
        robot_type="yam",
        use_videos=True,
        image_writer_threads=image_writer_threads,
        vcodec=vcodec,
    )
    restore_stats = _skip_depth_stats() if depth_cameras else None

    placeholders: List[str] = []
    skewed: List[str] = []
    for ep_dir in episode_dirs:
        with open(ep_dir / "metadata.json") as f:
            meta = json.load(f)
        n_frames = int(meta["num_frames"])
        prompts = meta.get("language", {}).get("prompt") or [""]
        task_text = prompts[0] or meta.get("language", {}).get("task", task_name)

        lowdim = _load_episode_lowdim(ep_dir, n_frames, cameras)
        uncalibrated = _placeholder_poses(lowdim, pose_keys)
        if uncalibrated:
            placeholders.append(f"{ep_dir.name}: {', '.join(uncalibrated)}")
        for cam in cameras:
            skew = _rotation_skew(lowdim[f"camera.{cam}.extrinsics"])
            if skew > 1e-3:
                skewed.append(f"{ep_dir.name}: {cam} (max |RᵀR - I| = {skew:.3f})")

        for i in tqdm(range(n_frames), desc=f"  {ep_dir.name}", leave=False):
            frame: Dict[str, object] = {feat: arr[i] for feat, arr in lowdim.items()}
            frame["task"] = task_text
            for cam in cameras:
                img_bgr = cv2.imread(str(ep_dir / "rgb" / cam / f"{i:010d}.png"))
                if img_bgr is None:
                    raise FileNotFoundError(f"{ep_dir}/rgb/{cam}/{i:010d}.png")
                frame[f"observation.images.{cam}"] = cv2.cvtColor(
                    img_bgr, cv2.COLOR_BGR2RGB
                )
            for cam in depth_cameras:
                depth_path = ep_dir / "depth" / cam / f"{i:010d}.npz"
                if not depth_path.exists():
                    raise FileNotFoundError(
                        f"{depth_path} (the first episode has depth for {cam}; "
                        "every exported episode must)"
                    )
                depth = np.load(depth_path)["depth"]
                # 65535 is the camera's saturated value, not a measurement (the
                # wrist D435 sends a whole frame of it when recording starts).
                depth[depth == 65535] = 0
                frame[f"{_DEPTH_PREFIX}{cam}"] = depth
            dataset.add_frame(frame)

        dataset.save_episode()
        print(f"  ✓ {ep_dir.name}: {n_frames} frames")

    dataset.finalize()
    if restore_stats is not None:
        restore_stats()
    print(f"\n✓ LeRobot dataset ready: {root}")
    print(
        f"  {dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames"
    )
    print(
        f"  View: uv run lerobot-dataset-viz --repo-id {repo_id} --root {root} --episode-index 0"
    )
    if placeholders:
        print(
            "\n⚠ Identity camera extrinsics / table pose (no calibration when converted). These poses are\n"
            "  placeholders, not measurements: put a calibration_results.json in the raw\n"
            "  recording and re-run `rd convert --reconvert` before relying on them."
        )
        for line in placeholders:
            print(f"    {line}")
    if skewed:
        print(
            "\n⚠ Camera rotations that are not rotation matrices: the calibration behind them is\n"
            "  skewed, so projecting with R.T is off. Fix the calibration and reconvert:"
        )
        for line in skewed:
            print(f"    {line}")
    return root
