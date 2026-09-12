# Oven task: real reference and sim renders

Reference material for the `croissant_oven_heating_region` task (put the croissant in
the oven). Checked in because `data/` is gitignored, and a fresh clone on another
machine needs these to work on the oven asset at all.

The oven model itself lives in MESA, not here: `mesa/sim/assets/objects/articulated_objects/oven.xml`
on branch `3d_simtoreal`. See [Simulation](../../docs/guide/sim.md) for the clone and setup.

| Folder | What | Regenerable? |
|---|---|---|
| `captures/` | Photos of the real oven, from the rig cameras | **No.** Needs the physical oven. |
| `renders/` | Sim review sheets and scene renders | Yes, from the MESA scripts. |

## `captures/` — the real oven

Four shots, taken 2026-09-11 with `scripts/gravity_comp_capture.py`. The arm is held at
zero stiffness so the operator can aim the wrist camera by hand while an assistant fires
the trigger. Each shot is one 640x480 PNG per camera plus `meta.json`.

`meta.json` holds the measured joints, the FK of `grasp_site` in the arm base frame, and
per camera the on-device intrinsics, distortion and image size. Only the wrist camera
carries `pose_in_base` (FK x hand-eye from `calibration_results.json`); the scene camera
is static and its pose comes from the rig.

| Shot | View | Note |
|---|---|---|
| `0000_oven_01` | Front, door closed | Handle, control panel, feet, top plate |
| `0001_oven_open_02` | Front, door open | The wire rack and the crumb tray below it |
| `0002_oven_open_left` | — | **Missed.** Aimed up over the oven, caught the top plate and the wall. Kept as a record of the failed pose. |
| `0003_oven_open_right` | Right side | Vent grille, feet, dark side panel. The door is out of frame, but the *scene* camera in the same shot caught it in profile. |

Two things worth knowing before taking more. The wrist lens is about 55 degrees across
640 px, so the camera has to be ~40 cm back to fit the oven and its open door in one
frame — two of these first four framed the wrong thing. And always check the scene camera
in the same shot; it has caught detail the wrist camera missed.

```bash
uv run python scripts/gravity_comp_capture.py --out reference/oven/captures
echo oven_front_closed > reference/oven/captures/TRIGGER    # from another terminal
```

The shot index counts directories in the output folder, so do not put anything else in
there.

## `renders/` — the model

| File | What |
|---|---|
| `oven_states_angles.png` | Closed / open / open + croissant, each from 6 angles |
| `oven_closed_vs_open.png` | The two door states side by side, one row per angle |
| `oven_sim_vs_real.png` | Sim next to the matching wrist photo, 4 pairs |
| `oven_scene_cameras.png` | The task in the raiden_lab arena, start and croissant-on-rack, 4 cameras |
| `scene/<state>_<camera>.png` | The same arena renders full size, 1280x720 |

`scene_camera` is the rig's real D435 pose. `sideview` is blocked by a partition and the
wrist camera sees only the gripper at the home pose, so neither is rendered. The goal
state is posed directly rather than by the task file, because an `in` initial-state
predicate does not place the object (O9).

In the *asset* sheets, MuJoCo azimuth 90 is the front, 0 the left side and 180 the knob
side — the opening faces the model's local -y. This is unrelated to the arena rotation,
where the class rotation is -pi/2.

These were produced by `scripts/raiden_lab/oven/` in MESA, which reads and writes this
folder and defaults to finding it at `../raiden`:

```bash
cd ~/robot/vla-benchmark
MUJOCO_GL=egl uv run python scripts/raiden_lab/oven/make_oven_review.py   # the 3 sheets
MUJOCO_GL=egl uv run python scripts/raiden_lab/oven/make_scene_review.py  # the arena renders
MUJOCO_GL=egl uv run python scripts/raiden_lab/oven/check_oven.py         # door + rack physics
```

Regenerating is not byte-identical: the arena re-samples the oven and croissant
placement on every reset, so a rerun gives a different valid layout. Only commit a
regenerated sheet if you meant to replace the reviewed set.

## The task does not run yet

`croissant_oven_heating_region` does not load. `BDDLBaseDomain` picks its object format
from the robot name, so `RaidenYam` selects `bimesa`, whose registry is built from
`assets/objects/objaverse` and `assets/objects/aigen_objs` — neither is in the asset
download. The croissant exists only in the robocasa (mesa) tree, so it cannot resolve.
The oven asset, its 81 texture variants and the task file are all in MESA; only the
format lookup stands in the way. See [Simulation](../../docs/guide/sim.md#tasks).

Reach is also unchecked: the open door adds 158 mm in front of the oven, so the arm has
to reach about 378 mm front to back to a rack 102 mm above the table, against a 0.74 m
teleop clamp. The sim walls have no collision, so a sim-only check would not catch a real
collision. Check reach before collecting anything.
