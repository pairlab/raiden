# Isaac scenes with single-arm or bimanual Quest control

`scripts/isaac_teleop.py` launches an explicitly mapped Isaac scene, then runs
Raiden's existing Quest teleop or recorder with the same arm mode. The server
uses the pinned Isaac Python environment; Raiden uses its own Python environment.
The launcher owns and stops its server when the client exits or startup fails.

Use the existing configured Isaac environment, including any vendor license
setup required by that installation. The launcher inherits those settings; it
does not answer vendor license prompts.

From the Raiden checkout on the pod:

```bash
export MESA_ISAAC_PROJECT=/home/frankchang/s2r/work/vla-benchmark-scene-teleop/mesa_isaac
export MESA_ISAAC_PYTHON=/home/frankchang/s2r/work/vla-benchmark/mesa_isaac/.venv/bin/python

.venv/bin/python scripts/isaac_teleop.py \
  --scene /home/frankchang/s2r/scenes/bimanual_bottles_1025_v01 \
  --task-config "$MESA_ISAAC_PROJECT/config/teleop/stand_bottle.json" \
  --arms bimanual --check
```

Use the same command without `--check` to record. Add
`--data-dir /path/to/data/isaac` for the dataset destination and
`--oculus-ip <headset-IP>` for the existing Wi-Fi ADB Quest connection. The pod
must be able to reach and authorize that headset; use the existing
[Quest setup](oculus_teleop.md) first. Add `--mode teleop` to control without
recording. `--arms single` activates the left arm and its wrist camera; bimanual
activates both arms and both wrist cameras. The fixed scene camera is always
recorded. A mismatched server/client mode fails before opening the input device.

On a machine with an interactive display, `--visualizer kit` enables the Isaac
window. The default is offscreen RTX. Raiden's `--monitor --monitor-web` options
can be passed through for a browser monitor. Camera FPS is a target, and actual
native frame timestamps are retained; inspect throughput on the collection pod.

The simulator can also run separately, with Raiden on the Quest-connected machine:

```bash
# In mesa_isaac, with its pinned interpreter:
PYTHONPATH=src HEADLESS=1 "$MESA_ISAAC_PYTHON" scripts/raiden_sim_server.py \
  --scene /path/to/scene/runtime_scene.json \
  --task-config config/teleop/stand_bottle.json --arms bimanual

# In Raiden, with RPC reachable at localhost (an SSH tunnel works):
.venv/bin/rd record --control oculus --arms bimanual --sim 127.0.0.1:5599
```

The server loads a scene JSON, its package directory, or a current-scene selector.
A raw USD alone does not define arm joints, tool parents, recording cameras, or
task reset semantics. Those mappings are explicit in the descriptor and task
configuration. Existing croissant twin launches without `--scene` remain supported.

## Bottle reset and success rules

All three bottles start on their sides. Cap headings range from -85 to +85 degrees
relative to the left base's +X direction, the away-facing half-plane of the arm
bank. The sampler uses each USD's full local bounds to place the object above the
measured tabletop, keeps its full footprint within the table, and conservatively
rejects overlapping bounding disks with 2 cm clearance.

The center region is x = 0.33–0.49 m in the left base. The lateral bands are
[-0.05, 0.03], [-0.31, -0.23], and [-0.55, -0.47] m. These bands come from
approximate bottle-center annotations of the initial frames of real
`stand_bottle/0000,0002,0003,0004,0005`, projected through the measured scene
calibration onto a depth-fitted table plane. They are an informed reset
distribution, not object ground truth. The evidence is tracked beside the task
configuration. The source table dimensions are unchanged.

The fixed camera uses the scene optical calibration and factory intrinsics saved
with those demos, expressed through the scene's authored left base. Wrist cameras
retain their explicit scene tool mounts and factory intrinsics. Images use native
orientation, with no 180-degree correction. Per-frame camera poses come from the
native renderer. RTX uses an undistorted pinhole model; real lens distortion and
camera response are not simulated.

Task success requires all three bottles upright within 15 degrees, supported
within 15 mm of the tabletop, inside its footprint, with less than 1 cm translation and 5 degrees rotation
over a continuous 0.5 seconds of simulation time. The checker provides
feedback; the existing save/discard controls remain the episode label.

`--seed` controls the reset sequence. Portable states include the realized reset,
RNG state, task hold timer, both articulations, objects, and commanded targets.
State replay restores these fields. Solver warmstarts are not serialized, so
command replay measures divergence instead of promising identical contacts.

## Recording and verification

Raw and converted recordings use the existing Raiden layout. Bimanual robot rows
contain both followers; native action logs contain 14 values in left/right order,
six absolute joint radians plus normalized aperture (1=open) per arm. Single-arm
logs contain seven values. Each native tick retains physical state, and each RGB
frame retains the native optical pose. The scene metadata records asset and code
hashes, versions, task configuration, and controller settings. These scene servos
are inherited generic position servos, not the calibrated single-arm motor model.

The three canonical bimanual camera names are `scene_camera`,
`left_wrist_camera`, and `right_wrist_camera`. Their RGB resolution is 640×360.
Isaac RGB recordings do not provide RealSense depth or force sensing.

For a controller-free integration check, start the server, then run:

```bash
.venv/bin/python scripts/sim_scripted_episode.py \
  --sim 127.0.0.1:5599 --arms bimanual --seconds 4 --out /tmp/bottle-smoke
.venv/bin/python scripts/isaac_replay.py \
  /tmp/bottle-smoke/raw/sim_smoke/0000 --mode commands --repeat 2
```

The existing converter accepts these recordings, followed by the existing
converted-dataset export paths. The separate native HDF5 exporter reads real
RealSense recordings; it does not read Isaac `.simrec` directories.
Training configurations must still select the intended arm count and views;
seven-action single-arm checkpoints do not become fourteen-action bimanual
policies without changing their model/data configuration.

If CUDA works but RTX produces black images, check `vulkaninfo --summary` for
the assigned NVIDIA GPU. The pod needs the standard Vulkan/X11 runtime libraries
and NVIDIA EGL vendor registration for its mounted driver. Do not replace the
host driver to resolve a missing container library. The scene server rejects
constant camera output during startup.

On the pod in a four-second scripted recording on 2026-10-25, single-arm
collection measured 29.7 native FPS across two cameras; bimanual collection
measured 20.1 FPS across three cameras. Robot sampling was 99 Hz. A separate
steady bimanual check sustained 120 Hz physics at real time and 27.9 FPS without
recording. The 30 FPS target is not a guaranteed bimanual recording rate.
