# Simulation (MESA digital twin)

Raiden can drive a MuJoCo twin of the rig instead of the robot. The same teleop,
recorder, converter and export run unchanged: the left follower and both cameras
are simulated, and the episodes land in the same schema as real ones.

The twin itself lives in a second repository, [MESA](https://github.com/pairlab/vla-benchmark).
Raiden talks to it over a local RPC (default `127.0.0.1:5599`), so the two
repositories stay separate and only the address is shared.

## What you need

- Two clones: this repository and `pairlab/vla-benchmark` (branch `3d_simtoreal`).
- `uv`, a GPU that can render with EGL, and about 15 GB of disk for the MESA assets.
- A Quest headset over USB (see [Quest Teleop](oculus_teleop.md)) or a SpaceMouse.
  There is no keyboard teleop; `scripts/sim_scripted_episode.py` exercises the
  pipeline without a controller.
- No robot, no CAN interfaces and no cameras. Sim skips the CAN check, and the rig
  cameras are rendered.

## MESA (the twin)

```bash
git clone -b 3d_simtoreal git@github.com:pairlab/vla-benchmark.git
cd vla-benchmark
uv sync                                                              # Python 3.10
uv pip install gymnasium lxml
uv pip install --no-deps git+https://github.com/robocasa/robocasa.git@v0.2
# then comment out the mujoco and numpy version asserts in
# .venv/lib/python3.10/site-packages/robocasa/__init__.py
./scripts/setup.sh                                                   # ~12 GB of assets
```

Everything the rig needs is in that branch: the calibrated rig (`rig.json`,
`layout.json`, `table_atlas.png`), the `raiden_lab` arena, the `RaidenYam` robot with
the linear_4310 gripper, the joint-space controller, the task suites, and the server.

Start the server in its own terminal and leave it running:

```bash
MUJOCO_GL=egl uv run python scripts/raiden_sim_server.py
```

It runs physics at 100 Hz, listens on port 5599, and loads
`stack_white_cubes` unless `--task-json` says otherwise. Restart it after changing a
task or updating the repository — it reads the task file once, at startup.

## Raiden

```bash
git clone --recurse-submodules -b quest-teleop git@github.com:pairlab/raiden.git
cd raiden
uv sync --extra lerobot
uv pip install --force-reinstall --no-deps "opencv-contrib-python==4.13.0.92"
```

The last line restores `cv2.imshow`, which the `--monitor` window needs: `lerobot`
installs `opencv-python-headless` over `opencv-contrib-python`. Re-run it after any
`uv sync`.

No `camera.json` is needed. The sim cameras come from the server's rig, and the
resolution check only applies to a camera that is also configured as a real
RealSense, so a machine with no cameras configured is fine.

## Teleoperate and record

```bash
uv run python scripts/quest_teleop.py --sim                                   # teleop only
uv run rd record --control oculus --arms single --sim 127.0.0.1:5599 --monitor
uv run python scripts/sim_scripted_episode.py --out /tmp/smoke --monitor      # no controller
```

Sim recording starts at every scene reset and only saves successes; the buttons and
the monitor views are described under
[Sim collection](oculus_teleop.md#sim-collection). Then convert and export as usual:

```bash
uv run rd convert
uv run rd lerobot
```

Sim recordings store per-frame states rather than pixels, and `rd convert` renders the
frames from them — see [Conversion](conversion.md). Without a display, raiden selects
`MUJOCO_GL=egl` itself.

`table.pose` in the exported dataset comes from the rig layout, which lives in MESA.
Copy it once so sim and real episodes carry the same table frame, otherwise the pose
is exported as the identity:

```bash
mkdir -p data/real2sim/calibration
cp ../vla-benchmark/mesa/task_suites/rigs/raiden_lab/layout.json data/real2sim/calibration/
```

## Tasks

**`stack_white_cubes`** works end to end and is what the server loads by default.

**`croissant_oven_heating_region`** (put the croissant in the oven) does not load yet.
`BDDLBaseDomain` picks its object format from the robot name, so `RaidenYam` selects
`bimesa`, whose registry is built from `assets/objects/objaverse` and
`assets/objects/aigen_objs`. Neither directory is in the asset download, so the
registry is empty and the croissant — which exists only in the robocasa tree — cannot
be resolved. The oven asset, its 81 texture variants and the task file are all in
MESA; only the object format stands in the way, and choosing between pulling those
assets in, registering the croissant into the flat registry, and not deriving the
format from the robot name is an open decision.

Reach has also not been checked for that task: the open door adds 158 mm in front of
the oven, so the arm has to reach about 378 mm front to back to a rack 102 mm above
the table, and that has never been tested against the 0.74 m teleop clamp. The sim
walls have no collision, so a sim-only check would not catch a real collision either.
Check reach before collecting anything on the oven.

## Init states and evaluation

Teleoperation does not need init states — every reset re-samples the scene. Fixed
init states are for evaluation, and MESA ignores `task_suites/init_states/`, so
regenerate them in the MESA clone:

```bash
MUJOCO_GL=egl uv run python scripts/raiden_lab/gen_init_states.py
cp -r /tmp/raiden_lab_init_states/mesa-eval/stack_white_cubes \
  mesa/task_suites/init_states/raiden_lab/
MUJOCO_GL=egl uv run python scripts/raiden_lab/region_check.py 100
```

`region_check.py` verifies that every sampled and saved layout is inside its region,
visible to the scene camera, clear of the rail and partitions, inside the teleop
clamp, and graspable within 30 degrees of tilt. `reset_dist.py` in the same directory
reports the reset distribution.
