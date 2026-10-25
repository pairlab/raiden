# Recording demonstrations

## Overview

The `rd record` command records a full teleoperation demonstration:

- **Camera data** at 30 fps - ZED cameras write `.svo2` files; RealSense
  cameras write `.bag` or `.db3` files. Native formats store the raw sensor stream
  (stereo pair + depth) and are converted to a structured dataset afterwards
  with `rd convert`.
- **Robot joint data** at ~100 Hz - leader and follower joint positions and
  velocities are saved to `robot_data.npz`. Before reading each row, Raiden
  samples the first camera's `get_current_timestamp_ns()`:
  - **ZED** returns current host time through `sl.TIME_REFERENCE.CURRENT`, on
    the same clock as its SVO2 frame timestamps.
  - **RealSense** uses `time.time_ns()` for robot rows. A frame's SDK timestamp
    domain identifies whether it uses global/system time or a device-relative
    hardware clock. For device-relative timestamps, the recorded
    `realsense_clock_offsets` in `metadata.json` provides an approximate host
    mapping from one received frame. Transport delay and clock drift remain
    unmeasured; enabling `global_time_enabled` alone does not prove exposure
    synchronization.

Arm observations and command snapshots are read sequentially after that
timestamp. A row shares one time label; it does not establish simultaneous
left/right measurements or hardware synchronization with a camera exposure.
New recordings also retain each arm's own host read bracket, available cached
feedback timestamp, and latest completed command API issue/return times.

All data for one recording episode lands in a single timestamped directory
under the output folder.

RealSense camera settings accept `"recording_format": "bag"` (the compatibility
default) or `"recording_format": "db3"`. Choose the format supported by the
installed SDK build: ROS2-enabled builds write `.db3`, while legacy builds write
`.bag`. This depends on build options, so Raiden does not guess from a version
number or retry with a different filename. A path that contradicts the configured
format is rejected before restarting a stream. Both formats remain readable
when supported by the playback SDK. The SDK's
[writer factory](https://github.com/realsenseai/librealsense/blob/v2.58.1/src/media/ros_factory.cpp)
documents this build-dependent choice.

## Single-arm vs bimanual

Pass `--arms single` to record with the left arm only:

```bash
rd record --arms single
```

In single-arm mode the active arm is always named **left** for consistency.
Bimanual robot vectors are ordered left then right. Each end-effector pose is
expressed in its own arm's base frame; calibrated camera extrinsics use the
left-arm base frame. Configure the physical CAN mapping as described in
[Hardware setup](hardware.md#optional-channel-configuration).

## Thread layout

| Thread | Purpose |
|---|---|
| `teleop-right` | Follower-right mirrors leader-right at 100 Hz (runs for whole session) |
| `teleop-left` | Follower-left mirrors leader-left at 100 Hz (runs for whole session) |
| `camera-<name>` | One per camera; calls `camera.grab()` in a tight loop (SDK limits to 30 fps); active only during an episode |
| `robot-recorder` | Samples reference-camera current time, then reads arm observations and commands sequentially at ~100 Hz; active only during an episode |

## Output layout

```
<output_dir>/<task>_<timestamp>/
    metadata.json        # task name, duration, fps statistics
    robot_data.npz       # joint positions and velocities for all arms
    cameras/
        scene_camera.svo2
        left_wrist_camera.svo2
        right_wrist_camera.svo2
```

### `robot_data.npz` keys

Each robot arm contributes keys of the form `<arm>_<field>`, e.g.:

| Key | Shape | Description |
|---|---|---|
| `timestamps` | `(N,)` int64 | Legacy reference-camera clock label sampled before each sequential robot row; retained for compatibility. |
| `readout_timestamp_ns`, `readout_monotonic_ns` | `(N,)` int64 | Host wall-clock and monotonic labels sampled before the row's sequential reads. Monotonic values support elapsed-time comparisons only. |
| `readout_end_ns`, `readout_end_monotonic_ns` | `(N,)` int64 | Host labels sampled after all observation and command snapshots return; command age is measured at this capture completion. |
| `leader_r_joint_pos` | `(N, 6)` float32 | Leader-right joint positions (rad) |
| `leader_r_joint_vel` | `(N, 6)` float32 | Leader-right joint velocities (rad/s) |
| `follower_r_joint_pos` | `(N, 6)` float32 | Follower-right arm joint positions (rad) |
| `follower_r_joint_vel` | `(N, 6)` float32 | Follower-right arm joint velocities (rad/s) |
| `follower_l_joint_pos` | `(N, 6)` float32 | Follower-left arm joint positions (rad) |
| `follower_{l,r}_joint_pos_7d` | `(N, 7)` float32 | Six measured arm joints followed by normalized measured gripper aperture (0 closed, 1 open) |
| `follower_{l,r}_joint_cmd` | `(N, 7)` float32 | Six commanded arm joints followed by binary operator gripper intent (0 closed, 1 open) |
| `<arm>_joint_cmd_api` | `(N, 7)` float32 | Exact latest teleop API input target, including physical gripper target; NaN when no API command is known. This is distinct from binary gripper intent and the driver's later motor command. |
| `<arm>_state_timestamp_ns` | `(N,)` int64 | Available cached state timestamp. Source `host_cached_feedback_read` labels a host read of cached CAN feedback, derived from driver float seconds; it is not motor acquisition or CAN receipt time. |
| `<arm>_state_timestamp_valid`, `<arm>_state_timestamp_source` | `(N,)` bool / string | Whether the state timestamp is known and what it measures. |
| `<arm>_read_start_ns`, `<arm>_read_end_ns` | `(N,)` int64 | Host wall-clock bracket around this arm's observation call and snapshot copy. |
| `<arm>_read_start_monotonic_ns`, `<arm>_read_end_monotonic_ns` | `(N,)` int64 | The corresponding monotonic bracket; use its difference for call duration. |
| `<arm>_command_timestamp_ns`, `<arm>_command_end_ns` | `(N,)` int64 | Host teleop command API issue/return times, saved atomically with the latest completed command snapshot; these do not measure CAN delivery or motor execution. |
| `<arm>_command_timestamp_valid`, `<arm>_command_timestamp_source` | `(N,)` bool / string | Whether the command issue time is known; current native source is `host_command_api_issue`. |
| `<arm>_command_start_monotonic_ns`, `<arm>_command_end_monotonic_ns` | `(N,)` int64 | Monotonic bracket around the same command API call. |

Only enabled arms contribute streams. Episode `metadata.json` stores the
resolved robot channels, gripper settings, vector ordering, and camera roles
and settings in `recording_config` for later conversion.
Timing semantics are also saved in `metadata.json` under `robot_timing`.
Unknown timestamps are `-1`, with validity `False` and source `unavailable`.
The initial measured-pose command fallback has unknown command timing. Only
the latest completed command at each recorder sample is retained; intermediate
API calls can be absent. An unchanged target does not prove stale feedback or
that no new command was issued.

## Auditing a saved episode's timing

Run this read-only command from the directory containing the `raiden` checkout:

```bash
./raiden/.venv/bin/python -m raiden.timing_audit /path/to/raw/episode \
  --reference-camera scene_camera \
  --alignment-report /path/to/converted/sequence/camera_alignment.json \
  --json-out /tmp/episode-timing.json
```

Omit `--alignment-report` to find `camera_alignment.json` in the episode or its
standard converted `0000` subdirectory, and omit `--reference-camera` to
use the first camera name in sorted order. Console output is the default;
`--json-out -` prints just JSON. A named JSON output must be a new file. The
command never changes the saved episode, opens a physical camera, resets a
device, or changes camera settings. It reads native RealSense metadata with
non-realtime `.bag`/`.db3` file playback, separately for color and depth, without decoding
images or aligning depth. The SDK bag format already retains per-frame
timestamps, timestamp domains, counters, and supported native metadata, so
recording does not add a second per-frame log. See the SDK's
[legacy bag format documentation](https://github.com/realsenseai/librealsense/blob/master/doc/record-and-playback-legacy-ros1.md).

The summary includes frame counts, counter gaps and duplicates, timestamp
coverage and interval percentiles, native metadata availability, nearest
RGB/depth timestamp skew within each device clock domain, and nearest
cross-camera timestamp residuals on the mapped host clock. Gaps reveal missing
recorded frames but cannot identify where they were lost. Comparisons report
frames outside the other stream's time coverage separately. Unknown domains
and hardware clocks without recorded host offsets remain explicitly unmapped.
Domain changes have separate coverage and transition counts; interval summaries
use adjacent frames in the same domain. An offset measured from a known global
timestamp is not applied to hardware-relative frames.
`camera_capture_receipts.<camera>.clock_mapping_receipt` records the one-frame
host-receipt mapping provenance in new recordings; its wait bracket is not a
bound on exposure time or clock error. Legacy offsets lack that provenance.

Robot metrics include per-arm observation-call duration, the age of the cached
feedback timestamp at observation-read end, command API duration, and command
issue age at completed capture of the row's observation and command snapshots.
Command age requires `readout_end_ns`; older recordings lacking this completion
label report it as unavailable. Reused command issue
times and unchanged command vectors are counted separately. Missing timing
fields in older episodes are reported as unavailable; row labels cannot recover
feedback freshness or left/right simultaneity.

Conversion saves `camera_alignment.json` with selected source-frame indices,
actual selected capture timestamps, signed capture-minus-grid residuals,
held/unused frame counts, and clock mapping provenance. The audit summarizes
that report without copying its per-frame arrays. In converted data, each wrist
image's forward-kinematics pose uses that camera's own selected capture time,
including when an image is held across grid slots. Available per-arm feedback
timestamps drive state interpolation; available command issue timestamps select
the latest command at or before the grid time. Legacy recordings retain row-time
fallbacks. Relative-camera-start alignment and receipt-based host mappings have
different limits, both visible in the report.

Playback is bounded by `--max-seconds` (120 wall seconds per file) and
`--max-frames` (1,000,000 combined color/depth frames per file). A limit, corrupt
metadata, or playback failure produces a labelled partial result and nonzero
exit status. Timing magnitudes have no automatic pass/fail threshold. Timestamp
agreement does not establish hardware exposure synchronization, flash-event
precision, CAN acquisition time, or command-to-motion latency. SVO2 timing is
available through the conversion report; this native metadata reader handles
RealSense recordings.

## Usage

```bash
# Leader-follower control (default)
rd record

# SpaceMouse EE-velocity control (paths loaded from ~/.config/raiden/spacemouse.json)
rd record --control spacemouse

# Single arm with SpaceMouse
rd record --control spacemouse --arms single

# Upload to S3 after each episode
rd record --s3-bucket my-robot-data --s3-prefix demonstrations

# Store data in a custom root directory (default: ./data)
rd record --data-dir /mnt/storage/robot_data
```

Episodes are written to `<data-dir>/raw/<task_name>/`.

Recorded gripper commands preserve binary operator intent: **0 closed, 1 open**.
The motor command is separately clamped to a minimum normalized opening of
`20/95` and passed through the driver's force limiter. Measured gripper aperture
remains continuous; it is a separate state field.

During the session:

- Press the **button on any leader arm** to start an episode; press it again
  to stop and save.
- After each episode, press the **leader button** again to start the next one,
  or press **`q`** to end the session.
- Press the **left foot pedal** (if connected) to trigger a soft e-stop
  **during recording**: all arms freeze for 5 seconds, then return to home and
  the session exits. The current episode is saved as incomplete. The e-stop is
  only active while an episode is recording — pressing the left pedal before
  recording starts will begin the episode instead. See
  [Soft E-Stop](hardware.md#soft-e-stop) for hardware details.
- Press **Ctrl-C** for an immediate emergency stop.

## Bimanual Quest control

With both follower channels and camera roles configured for the rig:

```bash
rd record --control oculus --arms bimanual --oculus-hand l
```

The left Quest controller drives the left follower, and the right controller
drives the right follower. `--oculus-hand l` is the default and is required for
this bimanual CLI mapping; `--oculus-hand r` would assign both followers to the
right controller and is rejected. For a single left-labelled follower,
`--arms single --oculus-hand r` still selects the right controller.

Calibrate **both controllers separately** at READY before recording: each
joystick click or B/Y starts that controller's +x/+y motion calibration, measured
along its own arm base axes. The rotation fit supports a securely mounted
headset facing outward at your neck/chest, including a fixed tilt. Poses remain
relative to the headset: **release both triggers before moving your torso or
adjusting the mount**, and recalibrate if the headset orientation changes.
There is no room-fixed torso compensation. This motion-axis calibration is
independent of camera calibration; see [Quest setup and calibration](oculus_teleop.md).

Hold each controller's trigger to move its arm relative to the clutch pose;
hold its grip to close that arm's gripper and release to open it. Either active
hand can start or stop an episode.

| Phase | Real recording buttons |
|---|---|
| READY | A/X starts; joystick click or B/Y starts or cancels calibration |
| Recording | A/X stops; calibration buttons are ignored |
| Verdict | Release the stop button, then press A/X again for success; joystick click or B/Y marks failure |

With `--sim`, collection is single-arm and starts recording at every scene
reset: **B/Y saves as success**, **A/X discards**, and **joystick click discards
and starts calibration**. These simulator button meanings differ from the real
READY/recording/verdict sequence. See [Local Isaac collection](#local-isaac-collection).

## Marking demonstrations

At the end of each episode, mark it as success or failure using the teaching
hardware. Only successful demonstrations are included when you run `rd convert`.

| Input | Action |
|---|---|
| Leader button / left foot pedal | Start or stop recording |
| Top leader button / middle foot pedal | Mark as **Success** |
| Bottom leader button / right foot pedal | Mark as **Failure** |

If no mark is given, the status stays `pending`. You can correct it later in
the [console](console.md).

After recording, convert the raw camera files with [rd convert](conversion.md).

## Recording from the browser

`rd record --ui` runs the session from a page at `http://localhost:8765` (it
opens by itself; `--ui-port` and `--ui-host` change the address) instead of
the terminal prompts. Real recording only.

- **Session:** pick the task and teacher, or add new ones.
- **Live views:** the scene camera with the task's placement limits projected
  on the table (white: the whole croissant and oven stay inside; blue: the
  croissant's centre of mass; orange: the oven), and a marker that turns green
  once the croissant is placed correctly. The wrist camera with
  `camera_preview --grid`'s lines and the fingertip reference, flagged if the
  mount has slipped or the view is dark.
- **Controls:** Start, Stop, Success, Failure and Skip, with the keys
  <kbd>Space</kbd>, <kbd>S</kbd>, <kbd>F</kbd> and <kbd>K</kbd>. The leader,
  Quest and pedal buttons keep working. A countdown shows how long the verdict
  prompt stays open.
- **Episodes:** each saved episode is checked (`raiden/qc.py`: colour lock,
  wrist aim, pauses, croissant start, gripper cycles, duration) and listed with
  PASS or the reasons. Pending episodes can be labelled there, and episodes
  deleted, files and DB row together, while nothing is recording.
- **Coverage:** where the croissant started in the successful episodes so far
  (every session), on a grid over the blue box, and where the oven stood, over
  the sim's range of oven positions; the current placements are outlined.
- **Dataset:** the header and a side panel count every episode of the task
  across sessions (successes, their minutes, failures, unlabelled, sessions,
  successes passing QC) next to this session's count.
- **Oven:** found from its three knob faces (a knob-face template, matched where
  the oven model puts them), drawn on the table in green or red, and checked
  against the sim: x within 1.5 cm of the sim's, y in the sim's range, the whole
  oven in the white box.
- **Terminal and Quest:** a panel shows everything `rd record` prints, with the
  Quest calibration prompts highlighted, and a box shows each controller's
  tracking and calibration state and the button help, for when the terminal is
  out of sight.

The limits and the camera pose come from the vla-benchmark twin
(`~/robot/vla-benchmark`: rig.json, layout.json and the croissant task JSON).
`scripts/qc_episodes.py <task> --watch` runs the same checks without the page.

Each task gets a summary, `data/raw/<task>/README.md`, rebuilt whenever an
episode is checked, relabelled or deleted: totals, success rate, the coverage
grids, a table per session, QC flag counts, every episode with its placements
and flags, and the setup (placement limits, colour lock, wrist reference).
Write notes between the markers in its Notes section; the rest is overwritten.
Sessions come from `data/qc/<task>/sessions.jsonl` (one line per
`rd record --ui` start), and a 30 min break also starts one. To rebuild it
without recording, run `scripts/qc_episodes.py <task>` once.

## Uploading to S3

Pass `--s3-bucket` to automatically upload each episode to S3 immediately
after it is saved:

```bash
rd record --s3-bucket my-robot-data
```

By default episodes are uploaded under the `demonstrations/` prefix. Override
with `--s3-prefix`:

```bash
rd record --s3-bucket my-robot-data --s3-prefix pick_purrito/session_01
```

Each episode directory is uploaded recursively. The S3 key for each file is:

```
<prefix>/<episode_dir_name>/<file>
```

For example, with `--s3-prefix demonstrations` and episode directory
`pick_purrito_20260312_220000`:

```
demonstrations/pick_purrito_20260312_220000/metadata.json
demonstrations/pick_purrito_20260312_220000/robot_data.npz
demonstrations/pick_purrito_20260312_220000/cameras/scene_camera.svo2
demonstrations/pick_purrito_20260312_220000/cameras/left_wrist_camera.svo2
demonstrations/pick_purrito_20260312_220000/cameras/right_wrist_camera.svo2
```

**Credentials** — Raiden uses the standard AWS credential chain via `boto3`.
Configure credentials with any of the usual methods before recording:

```bash
# Option A — environment variables
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1

# Option B — AWS CLI profile
aws configure
```

## RealSense bag file size

> **Warning:** RealSense `.bag` files are large. A 10-second recording at 30 fps (640×480 BGR8 color + 640×480 depth) typically produces **~500 MB–1 GB per camera**. For longer demonstrations or setups with multiple RealSense cameras, disk space can fill up quickly.
>
> **Mitigations:**
>
> - Reduce recording duration where possible.
> - Ensure the recording disk has sufficient free space before starting a session.

## Local Isaac collection

Launch `mesa_isaac/scripts/raiden_sim_server.py` in the Isaac environment with
its accepted calibration and placement ID (see `mesa_isaac/README.md`). Then
run Raiden in its own environment:

```bash
rd record --control oculus --arms single --sim 127.0.0.1:5599 --data-dir data/isaac --monitor
```

The standard Quest path uses six absolute joint targets and normalized aperture
(1 = open). Each simulation reset starts an episode. Quest B/Y, the middle pedal,
or Enter saves it and resets; Quest A/X, the right pedal, or `r` discards it and
resets. `q` discards the active episode and ends the session. Quest calibration
discards the active episode and starts a fresh reset after calibration finishes.
Saving is the operator's label; the Isaac task checker is disabled. The monitor
shows the calibrated scene and wrist cameras. `--ui` falls back to the terminal
workflow for simulated collection.

Isaac recordings retain native frames and full scene/controller state. Their
raw layout includes:

```text
<episode>/
    robot_data.npz                 # ~100 Hz sampled robot observations/commands
    calibration_results.json       # episode rig calibration, not the live hardware file
    sim_rig.json                   # camera intrinsics and optical transforms
    sim_metadata.json              # backend/package versions, scene/twin hashes, settings
    sim_initial_state.json         # physical/controller state before the first logged tick
    sim_action_log.npz             # every physics tick, requested/effective targets and clocks
    sim_action_events.jsonl         # native incoming command events, timing and clamp results
    sim_action_states.jsonl         # initial state, then one state after every native tick
    sim_action_metadata.json
    cameras/<name>.simrec/
        timestamps.npy             # original native capture timestamps
        camera_info.json
        camera_poses.npy           # matching T_base_cam, OpenCV optical convention
        sim_state.jsonl            # full scene/controller snapshot behind each frame
        rgb/0000000000.png          # lossless BGR PNGs, frames mode only
```

The native action log has one command row per physics tick and one more state
row than commands, including the initial and terminal states. Camera JSONL
records the state at each image capture, with shared server frame IDs and
timestamps for matching views; individual camera streams can drop frames.
An episode is marked complete only after its native action bundle is saved.
If finalization fails, the incomplete directory is retained for recovery and is
not reused by the next recording.

To defer storing pixels, add `--sim-record-mode states`. Live monitoring still
uses the server's camera frames. With the matching Isaac server running,
materialize the pixels afterwards, then convert the episode normally:

```bash
python scripts/isaac_replay.py data/isaac/raw/<task>/0000 --mode states --rerender
rd convert --data-dir data/isaac
```

`rd convert` selects successful demonstrations from the database under that
root. To convert one explicit episode directly, use the Python API (replace
`<task>` with the recorded task name):

```bash
python -c "from raiden.converter import convert_recording; convert_recording('data/isaac/raw/<task>/0000')"
```

Rerender restores each recorded physical state and checks its optical camera
pose against the capture. Live reset poses and sample indices may differ; replay
restores those from the recording. It rejects scene, calibration, twin, physics
backend, step size, or controller metadata changes. For a deliberate calibration/scene
experiment, use `--allow-scene-change --out /tmp/rerender_experiment`; this keeps
the original captures and poses intact. Rerendering does not replay dynamics.
RTX lighting/temporal accumulation can produce small RGB differences even when
camera poses and restored physical states agree.

To test dynamics, replay the original native per-tick target log instead:

```bash
python scripts/isaac_replay.py data/isaac/raw/<task>/0000 --mode commands --repeat 2 --out /tmp/isaac_replay
```

This checks that tick times are contiguous from the saved initial state and
applies every target with the original hold interval. Its optional traces retain
all portable scene fields for calibration analysis; the per-frame training
export remains separate. Isaac's portable state restores physical state,
controller latches and gains, but does not include solver/contact warmstarts.
The replay reports variation across repeats rather than promising bit-identical
physics. Replay restores the prior server state and leaves the server paused;
resume it explicitly through the simulator before starting another collection.
Use `scripts/sim_replay.py` for MuJoCo episodes. These two replay scripts only
address simulator RPC servers; `rd replay` is the separate hardware workflow.

For a recording smoke test without a headset, use a temporary data root:

```bash
python scripts/sim_scripted_episode.py --sim 127.0.0.1:5599 --seconds 3 --out /tmp/isaac_record_smoke
```
