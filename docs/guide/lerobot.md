# Exporting to LeRobot

`rd lerobot` writes converted Raiden episodes into a
[LeRobot](https://github.com/huggingface/lerobot) dataset (codebase v3), so
they can be trained on, visualized, or pushed to the Hugging Face Hub with the
LeRobot tooling.

## Installation

LeRobot is an optional extra:

```bash
uv sync --extra lerobot
```

## Usage

```bash
rd lerobot
```

As with `rd shardify`, an fzf selector lists the converted tasks under
`./data/processed/`. The dataset is written to `data/lerobot/<task_name>`.

```bash
rd lerobot --output-dir /mnt/storage/lerobot --repo-id myorg/cube_stack
rd lerobot --overwrite            # replace an existing export
rd lerobot --vcodec libx264       # H.264 instead of the AV1 default
```

## What it produces

| Feature | Type | Content |
|---|---|---|
| `observation.images.<camera>` | video | one stream per camera, in recording order (scene first, then wrist) |
| `observation.state` | float32 | measured joint positions + gripper (`l_joint_0..5, l_gripper[, r_...]`) |
| `observation.ee_pose` | float32 | measured gripper pose from FK: `l_x, l_y, l_z, l_r00..l_r22, l_gripper[, r_...]` (position + row-major 3×3 rotation in the left arm base frame) |
| `action` | float32 | commanded joint positions + gripper, same layout as `observation.state` |
| `action.ee_pose` | float32 | commanded gripper pose from FK of the commanded joints, same layout as `observation.ee_pose` |
| `camera.<camera>.intrinsics` | float32 (4) | `fx, fy, cx, cy` of that camera's exported image (pinhole) |
| `camera.<camera>.extrinsics` | float32 (12) | camera-to-base pose `x, y, z, r00..r22` (see [Camera calibration](#camera-calibration)) |
| `table.pose` | float32 (12) | table-to-base pose, same layout, static per episode (see [Table pose](#table-pose)) |
| `task` | string | the task instruction from the recording |

Both joint-space and end-effector-space versions are stored so the policy
action space can be chosen at training time (e.g. absolute joints, or
delta-EE derived from consecutive `action.ee_pose` rows).

Sim-only per-frame data (`sim_state`, `object_poses`, `subtasks`, `success`;
see [Conversion](conversion.md)) stays in the converted episodes and is not
exported, so sim and real datasets have the same features.

Frames are the aligned `rgb/<camera>/*.png` files of each converted episode,
so the export matches what `rd shardify` sees. Timestamps are
`frame_index / fps`.

## Camera calibration

Every camera gets two features, stored per frame because a wrist camera moves
every frame and a scene camera can move between sessions. Both are the
converter's `intrinsics` / `extrinsics` for that frame (see
[Conversion](conversion.md)), reshaped:

- `camera.<camera>.intrinsics` = `[fx, fy, cx, cy]` of the image in
  `observation.images.<camera>`, after any crop; the image size is that
  feature's shape. Pinhole: no distortion is stored, and the D435 colour
  streams report none.
- `camera.<camera>.extrinsics` = pose of the camera's OpenCV optical frame
  (x right, y down, z forward) in the left arm base frame, in the
  `observation.ee_pose` layout without the gripper: optical centre `x, y, z`,
  then the row-major rotation `r00..r22`. Scene cameras carry their static
  calibration; wrist cameras carry FK of the measured joints times the
  hand-eye calibration.

Projecting a point `p` given in the arm base frame:

```python
fx, fy, cx, cy = sample["camera.scene_camera.intrinsics"].numpy()
e = sample["camera.scene_camera.extrinsics"].numpy()
t, R = e[:3], e[3:].reshape(3, 3)
x, y, z = R.T @ (p - t)
u, v = fx * x / z + cx, fy * y / z + cy
```

The camera features are deliberately not under `observation.*`: LeRobot hands
every `observation.*` float feature to a policy as a state input, so a policy
(a pixel baseline, say) sees camera poses only if it reads them by key.

The export warns about two calibration problems, without stopping:

- **Identity poses.** The recording's `calibration_results.json` had no
  calibration for a camera (or no table pose), so the converter wrote the
  identity. Those poses are placeholders: put a calibration into the raw
  recording and run `rd convert --reconvert`.
- **Skewed rotations.** An extrinsics rotation that is not a rotation matrix
  (max |RᵀR − I| > 1e-3), so projecting with `R.T` is wrong. Fix the
  calibration it came from and reconvert.

## Table pose

`table.pose` is the pose of the table frame in the left arm base frame, in
the camera-extrinsics layout: a table-frame point `q` is `R @ q + t` in the
base frame. The table frame has its origin at the centre of the table top and
axes parallel to the base axes. It comes from `table.T_base_table` in the
recording's `calibration_results.json`: sim recordings take it from the rig,
real ones from `data/real2sim/calibration/layout.json` (the taped table
extent), so both use the same frame, and moving the scene camera does not
change it.

## Viewing an episode

LeRobot's viewer shows all cameras side by side together with the state and
action plots:

```bash
uv run lerobot-dataset-viz --repo-id raiden/cube_stack --root data/lerobot/cube_stack --episode-index 0
```

For a plain MP4 with the cameras side by side (no LeRobot needed), render
straight from a converted episode:

```bash
uv run python scripts/rollout_video.py data/processed/cube_stack/0003
```

## Loading in Python

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("raiden/cube_stack", root="data/lerobot/cube_stack")
sample = ds[0]
sample["observation.images.scene_camera"].shape   # (3, H, W) float tensor
sample["observation.state"], sample["action"]
sample["camera.left_wrist_camera.extrinsics"]      # (12,) wrist camera-to-base at this frame
```
