# Native recordings to policy HDF5

`python -m raiden.native_hdf5` exports explicitly selected, completed RealSense
episodes directly from their native `.bag` or `.db3` files. It supports single
arm and bimanual recordings without a camera SDK, connected devices, or a local
Raiden database. The original camera files, robot telemetry and metadata remain
unchanged. Install the optional dependencies with `uv sync --extra hdf5`.

After uploading a raw task directory to the processing machine, run:

```bash
python -m raiden.native_hdf5 \
  --raw-root /path/to/data/raw/stand_bottle \
  --output /path/to/processed/stand_bottle.hdf5 \
  --episodes 0002 0003 0005 \
  --validation-episodes 0003
```

The episode IDs are the reviewed success list. This command does not infer
success from `complete: true`, or consult the desktop's separate database.
Incomplete recordings are rejected. Choose an explicit validation holdout for
actual training; the example above only illustrates the option. Existing output
files are never overwritten, and failed exports do not publish a partial HDF5.

Each episode becomes `data/demo_<id>`. `mask/train` and `mask/valid` hold the
selected split. RGB images are stored at 128 and 256 pixels by default; `--sizes`
changes these. Unsuffixed `<camera>_image` keys point to the first requested size,
so use `--sizes 256 128` when the training configuration requests 256 pixels.

The RGB pipeline applies the episode's recorded policy crop, a centered square
crop based on the actual recorded dimensions, then `cv2.INTER_AREA` resize.
All cameras retain native orientation. BGR source pixels become RGB. Intrinsics
come from capture receipts and follow those same crops and sizes; no current
camera configuration is consulted. Camera calibration, distortion metadata,
hand–eye transforms and the measured inter-base transform are retained. A table
transform is exported only when it is actually present in the source calibration.

`actions_joint_pos` is seven values per arm, ordered left then right: six absolute
joint targets in radians followed by binary operator gripper intent (0 closed,
1 open). `obs/robot0_joint_pos` and `obs/robot0_gripper_jaw_width` contain measured
left-arm state; `robot1_*` is the right arm. Measured aperture stays continuous.
The `actions` alias uses the same joint action contract; it is not an EE pose.

Policy timestamps form a uniform 30 Hz grid inside overlapping camera/state
coverage. New recordings use each arm's own feedback timestamp for state
interpolation and its completed command API issue timestamp for causal command
selection. Older recordings without those fields retain row-time state and arm
target interpolation plus causal binary gripper hold. Source command indices,
timestamps, image frame indices and actual image capture timestamps are saved.

ROS2 `.db3` Image headers are recording-relative. The exporter joins every color
image to its native metadata and uses the recorded SDK timestamp/domain instead.
Hardware-clock frames require a matching recorded host mapping. Unknown clocks,
missing metadata, or image selection skew above one policy period fail export.
`--max-image-skew-ms` makes that freshness bound explicit when another is needed;
it does not certify hardware exposure synchronization.

Diagnostics report camera skew, source reuse, frame counter gaps, robot cadence
gaps and per-arm timing sources. Each policy anchor also retains state
interpolation bracket timestamps and interval lengths. The boolean
`robot_state_gap_over_50ms` marks anchors interpolating across a state interval
over 50 ms. Such intervals produce a warning and remain visible for review;
episodes and frames are never silently dropped or relabeled as measured samples.

This exporter produces **RGB and joint policy inputs**, matching the existing
single-arm RGB training contract. It does not export depth images. Native depth
remains available for the corrected SDK conversion path and future geometry
processing. A 512-value zero `task_emb` is retained for older single-task readers;
use a single-task training configuration or generate actual task embeddings when
task conditioning is required. Recompute normalization statistics and any visual
feature caches for the new task, arm count, cameras and image preprocessing.
