# Running the policy server

The `rd serve` command starts a live inference server that streams camera
observations and robot proprioception to a remote policy over WebSocket using
the [chiral](https://github.com/TRI-ML/chiral) protocol.

## Usage

```bash
rd serve
```

The server binds on `0.0.0.0:8765` by default.  Connect a chiral policy client
to `ws://<host>:8765`.

## Arms

By default the server drives both follower arms.  For a single-arm rig
(recorded with `rd record --arms single`), drive the left follower only:

```bash
rd serve --arms single
```

## Action space

By default the server operates in **EE pose** mode.  The policy receives
observations and returns a flat action vector:

```
[l_xyz(3), r_xyz(3), l_rot6d(6), r_rot6d(6), l_grip(1), r_grip(1)]    # bimanual, 20-D
[l_xyz(3), l_rot6d(6), l_grip(1)]                                     # single, 10-D
```

Poses are in each arm's own base frame (left arm in left-base frame, right arm
in right-base frame) — the same convention used in `lowdim.npz` during
shardification.

The server solves IK on-the-fly to convert the policy's EE pose target into
joint commands, seeding each solve from the last commanded joint positions for
a warm-started, smooth solution.

To operate in **joint** mode instead (absolute joint positions, left arm first):

```bash
rd serve --action-type joint
```

| `--action-type` | `--arms` | Action dims | Layout |
|---|---|---|---|
| `ee_pose` (default) | `bimanual` | 20 | `l_xyz(3) + r_xyz(3) + l_rot6d(6) + r_rot6d(6) + l_grip(1) + r_grip(1)` |
| `ee_pose` | `single` | 10 | `l_xyz(3) + l_rot6d(6) + l_grip(1)` |
| `joint` | `bimanual` | 14 | `l_joints(7) + r_joints(7)` |
| `joint` | `single` | 7 | `l_joints(7)` |

Each arm's 7 joints are the 6 arm joints followed by the gripper opening
(normalized 0–1).  The layout is also reported by the `metadata` request.

## Control rate

Set `--control-hz` to the rate the client sends actions at (default 10).  Each
action is reached one period after it arrives and held until the next one; the
command is sent to the followers in 100 Hz sub-steps, the rate of the teleop
loops that record the training data.  An action that arrives before the
previous one is reached replaces it, so commands never queue up behind the
robot.

```bash
# Single-arm joint-space policy executing at 30 Hz
rd serve --arms single --action-type joint --control-hz 30
```

## Cameras

Images are served at the resolution they are recorded at, as RGB.  RealSense
cameras use the `resolution`, `fps`, `crop` and `depth` settings from
`camera.json`, the same settings `rd record` and `rd convert` use, and the
intrinsics are shifted by the crop.  Pass `--resize-images HxW` (e.g.
`384x384`) to resize on the server instead; intrinsics are scaled to match.

`--no-depth` turns depth off for every camera: ZED NEURAL_LIGHT inference and
the RealSense depth stream and its alignment to color.

## Safety

The server enforces a per-step joint delta limit.  If any joint command deviates
from the current measured position by more than `--max-joint-delta` radians, the
command is dropped and the emergency stop below is triggered.

```bash
# Tighter limit (default is 0.8 rad)
rd serve --max-joint-delta 0.3
```

The gripper command goes through the same limit as teleoperation: it never
closes past the minimum opening.  Squeeze force on a blocked grasp is bounded
by the motor controller's own force limiter.

A footpedal e-stop is attached automatically if present.  Pressing the left
pedal, or a client disconnecting, immediately triggers an emergency stop: arms
are held at their current positions for 5 s, then moved to home, and the server
exits.

## Options

| Option | Default | Description |
|---|---|---|
| `--host` | `0.0.0.0` | WebSocket bind address |
| `--port` | `8765` | WebSocket port |
| `--arms` | `bimanual` | Followers to drive: `bimanual` or `single` (left only) |
| `--action-type` | `ee_pose` | Action space: `ee_pose` (IK) or `joint` (direct) |
| `--control-hz` | `10` | Rate the client sends actions at |
| `--max-joint-delta` | `0.8` | Safety limit in radians per step |
| `--no-depth` | `false` | Disable depth on all cameras (ZED NEURAL_LIGHT, RealSense depth stream) |
| `--resize-images` | none | Resize images to `HxW` before sending to the policy; native resolution when unset |
| `--camera-config-file` | `~/.config/raiden/camera.json` | Path to camera config |
| `--calibration-file` | `~/.config/raiden/calibration_results.json` | Path to calibration file |

Run `rd serve --help` for the full list.
