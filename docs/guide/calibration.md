# Camera calibration

## Overview

Raiden uses ChArUco boards to calibrate cameras. Two types of calibration are
supported:

- **Hand-eye calibration** (wrist cameras) - determines the rigid transform
  from the camera frame to the gripper frame (`T_camera→gripper`).
- **Scene camera calibration** - determines the transform from the camera
  frame to the robot base frame (`T_camera→base`).

![Calibration setup](../assets/calibration.jpg)

## Step 1 - Print a ChArUco board

Print a ChArUco board and mount it rigidly (e.g. on a flat aluminium plate).
For best results, consider an industry-grade printed target from
[calib.io](https://calib.io/products/charuco-targets). These are manufactured
on dimensionally stable material and give lower reprojection errors than
paper prints.

## Step 2 - Record calibration poses

Move the robot to a series of diverse poses while the board is visible. The
pose recorder defaults to a minimum target of **5 poses**. For the bimanual rig,
record **12–16 distinct stationary configurations per wrist**, varying depth,
position, and rotation about at least two axes. The solver's three valid
observations per wrist is a numerical minimum, not a quality criterion; the
runner also requires at least three board detections per target camera.

```bash
rd record_calibration_poses
```

The recorder stops before robot initialization if the camera configuration is
empty or any configured camera fails to open or provide warmup frames.

A live Rerun preview opens automatically in your browser, with each configured
camera shown side by side so you can check that the board is in frame before
recording each pose. Keep the recorder terminal focused for keyboard commands.
If the browser does not open (for example, over SSH), open the printed URL;
the terminal also prints an SSH tunnel command:

```
Rerun viewer:    http://localhost:9877?url=...
SSH tunnel:      ssh -L 9877:localhost:9877 -L 9878:localhost:9878 <host>
```

Images are refreshed at up to **10 Hz** in each camera's native orientation.

Keyboard controls:

- **R** - record current pose
- **D** - delete last recorded pose
- **L** - list all recorded poses
- **Q** - quit (saves if enough poses recorded)

Poses are saved to `~/.config/raiden/calibration_poses.json`.

### Bimanual Quest capture with the previous 4×5 board

The previous target was **4×5 squares, 50 mm squares, 37 mm markers,
`DICT_6X6_250`**. Dimensions below are in metres. Check the actual print scale
if the board has been reprinted. Secure the board flat and keep it fixed
throughout pose recording and calibration, including while switching arms.

On the robot host, `camera.json` must assign `left_wrist`, `right_wrist`, and
`scene` roles to the intended cameras. The wrist roles enable both followers;
Quest uses the left controller for the left arm and the right controller for
the right arm. The same driver reads the optional
`~/.config/raiden/robot.json` CAN mapping:

```json
{"follower_l": "can_left", "follower_r": "can_right"}
```

Without this file the historical `can_follower_l` / `can_follower_r` names are
used. These instructions do not install or change a live robot profile.

From the Raiden checkout, using its installed environment:

```bash
rd record_calibration_poses \
    --control oculus \
    --min-poses 12 \
    --camera-config-file ~/.config/raiden/camera.json \
    --output-file ~/.config/raiden/bimanual/calibration_poses.json \
    --squares-x 4 --squares-y 5 \
    --square-length 0.05 --marker-length 0.037 \
    --dictionary DICT_6X6_250
```

**This command initializes and homes both enabled arms**, signals readiness
with the grippers, and starts Quest teleoperation. Calibrate each controller's
headset-to-robot axes if needed (joystick click or B/Y, then the prompted
A/X-held +x and +y gestures). Hold that hand's trigger to move its arm;
release it and let the arm settle before saving a pose. A/X or keyboard **R**
saves the measured joints of **both arms** in one entry. **D** removes the last
entry, **L** lists entries, and **Q** saves and returns the arms home.

You may position one arm at a time. Give each wrist its own diverse set of
views; twelve entries with one arm stationary provide no useful diversity for
that wrist. Both wrists may observe the fixed board at different poses or
times. Keep the scene camera's view of the board clear as well. The Rerun
preview preserves native image orientation; calibration uses native pixels
and matching factory intrinsics.

Pose recording saves joint configurations and board metadata. It does **not**
save calibration image pairs or solve camera transforms.

Then run the capture and solve separately:

Repeat the board flags: the calibration CLI's board options override the
metadata stored by the pose recorder, including when its defaults are used.

```bash
rd calibrate \
    --camera-config-file ~/.config/raiden/camera.json \
    --poses-file ~/.config/raiden/bimanual/calibration_poses.json \
    --output-file ~/.config/raiden/bimanual/calibration_results.candidate.json \
    --squares-x 4 --squares-y 5 \
    --square-length 0.05 --marker-length 0.037 \
    --dictionary DICT_6X6_250
```

**This command moves both arms automatically**: it homes them, replays the
saved joint configurations with simultaneous arm moves, waits for settling,
and captures images and measured joints. Manual teleop paths are not saved,
so replay interpolates directly between the saved configurations. Choose
successive configurations with clear paths for both arms.

Native images, intrinsics, stable pose IDs, and both measured joint vectors
are retained beside the candidate under
`bimanual/calibration_captures/<timestamp>/captures.json`. Failed detections
and failed pose estimates are excluded with their corresponding joint poses.
The solve rejects missing or invalid extrinsics before writing the candidate.
Inspect reprojection consistency on withheld wrist poses and confirm the
mount stays rigid before using the result as measured ground truth.

The explicit candidate output leaves the active
`~/.config/raiden/calibration_results.json` untouched. Loading accepted results
into Isaac is a separate step: compose each hand–eye with the same YAM
`grasp_site` frame used by FK, then convert OpenCV optical axes into the
renderer camera convention. This workflow reads factory intrinsics and
distortion; it estimates wrist hand–eye transforms, the inter-base transform,
and scene-camera extrinsics rather than refitting intrinsics.

## Step 3 - Run calibration

```bash
rd calibrate
```

If your board has different dimensions or a different dictionary, pass them here:

```bash
rd calibrate \
    --squares-x 9 \
    --squares-y 9 \
    --square-length 0.03 \
    --marker-length 0.023 \
    --dictionary DICT_6X6_250
```

| Parameter | Default | Description |
|---|---|---|
| `--squares-x` | `9` | Number of squares along X |
| `--squares-y` | `9` | Number of squares along Y |
| `--square-length` | `0.03` | Checker square side length in metres |
| `--marker-length` | `0.023` | ArUco marker side length in metres |
| `--dictionary` | `DICT_6X6_250` | ArUco dictionary |

Raiden reads `~/.config/raiden/camera.json` and calibrates every camera whose
`role` is `"left_wrist"`, `"right_wrist"`, or `"scene"` automatically - no
explicit camera IDs are required.

The runner will:

1. Load the recorded poses from `~/.config/raiden/calibration_poses.json`
2. Move the robot to each pose
3. Capture frames from all target cameras
4. Detect the ChArUco board in each frame
5. Run hand-eye / scene calibration
6. Save results to `~/.config/raiden/calibration_results.json`

## Output

`~/.config/raiden/calibration_results.json` contains one entry per camera:

```json
{
  "version": "1.0",
  "timestamp": "2026-03-12T22:00:00.000000",
  "coordinate_frame": "left_arm_base",
  "cameras": {
    "left_wrist_camera": {
      "type": "hand_eye",
      "serial_number": 16522755,
      "intrinsics": {
        "camera_matrix": [[526.7, 0, 319.4], [0, 526.7, 178.5], [0, 0, 1]],
        "distortion_coeffs": [-0.041, 0.013, 0.0, 0.0, 0.0],
        "image_size": [672, 376],
        "source": "zed_sdk_factory_calibration"
      },
      "num_poses_used": 8,
      "hand_eye_calibration": {
        "success": true,
        "rotation_matrix": [[0.999, -0.012, 0.008], [0.012, 0.999, -0.003], [-0.008, 0.003, 0.999]],
        "translation_vector": [0.012, -0.034, 0.061],
        "rotation_vector": [-0.003, 0.008, 0.012],
        "method": "Tsai"
      }
    },
    "scene_camera": {
      "type": "scene",
      "serial_number": 37038161,
      "intrinsics": {
        "camera_matrix": [[700.1, 0, 659.8], [0, 700.1, 371.2], [0, 0, 1]],
        "distortion_coeffs": [-0.052, 0.021, 0.0, 0.0, 0.0],
        "image_size": [1280, 720],
        "source": "zed_sdk_factory_calibration"
      },
      "num_poses_used": 8
    }
  },
  "bimanual_transform": {
    "T_left_from_right": [[1, 0, 0, 0.612], [0, 1, 0, 0.0], [0, 0, 1, 0.0], [0, 0, 0, 1]],
    "computed_from_calibration": true,
    "direction": "T_left_from_right maps right_arm_base points into left_arm_base"
  }
}
```

New files also retain the historical `right_base_to_left_base` field for
compatibility. Despite its name, that legacy field maps **left-base points
into right-base coordinates**, so it is the inverse of `T_left_from_right`.
Conversion and serving prefer the explicit canonical field and invert the
legacy field when reading older artifacts. Scene-only calibration, without a
calibrated left wrist, reports scene extrinsics in the board frame.

