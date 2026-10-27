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

Both bottles start on their sides. Cap headings range from -50 to +30 degrees
relative to the left base's +X direction, the away-facing half-plane of the arm
bank. The sampler uses each USD's full local bounds to place the object above the
measured tabletop, keeps its full footprint within the table, and conservatively
rejects overlapping bounding disks with 2 cm clearance.

The center region is x = 0.33–0.52 m in the left base. The lateral bands are
[-0.15, 0.04] m for the left bottle and [-0.63, -0.37] m for the right bottle.
These collection defaults come from approximate center/axis annotations of
initial frames in nine new two-bottle `stand_bottle` demos spanning 0000–0050.
Saved scene-camera calibration projects rays to a nominal bottle-center plane
34 mm above the fitted tabletop. XY ranges include 20 mm padding; five partially
occluded centers are estimated. This is a practical reset distribution, not
object ground truth. `stand_bottle_region_evidence.json` stores the annotations,
projection assumptions and source identities. Table containment and bottle
separation remain enforced at every reset.

The fixed camera uses the scene optical calibration and factory intrinsics saved
with those demos, expressed through the scene's authored left base. Wrist cameras
retain their explicit scene tool mounts and factory intrinsics. Images use native
orientation, with no 180-degree correction. Per-frame camera poses come from the
native renderer. RTX uses an undistorted pinhole model; real lens distortion and
camera response are not simulated.

Task success requires both bottles upright within 15 degrees, supported
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
hashes, versions, task configuration, and controller settings. The bottle task uses generic position servos with source-CAD gravity feedforward,
240 Hz integration, stronger finger coupling and decomposed vendor finger colliders.
These remain independent of the calibrated single-arm motor model.

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

On the RTX 3090 desktop, the October 27 two-bottle collection profile uses a
60 Hz host tick and four solver substeps, retaining 240 Hz contact integration.
Targets and gravity feedforward refresh at 60 Hz. Batched cached gravity evaluation,
a shared tiled RTX product, cached camera remapping and corrected late-frame
scheduling improve the prior live 6–7 FPS to about 25 FPS at real time across three
views. The selected profile renders internally at half resolution with DLSS
performance mode and returns calibrated 640×360 images. Geometry, optical rays
and transparent materials are preserved; image detail is softer. Frame generation
is disabled. Sustained 30 FPS has not been demonstrated. Use a separate config
with `render_scale = 1.0` for full internal resolution. Evidence is under
`artifacts/bimanual_fps/`; recorded metadata includes the render and substep settings.

## Real-action bottle replay on the desktop

The current two-bottle replay uses real `data/raw/stand_bottle/0050`. Evidence and the reproduction script are under `artifacts/two_bottle_pickup/` at the workspace root. Native replay keeps the sampled API joint targets and motor aperture (close is about 0.21), with causal per-arm command completion timing. It does not substitute measured joint states or binary operator intent for commands.

The dry replay supplies grasp-aligned initial poses for two scanned bottles, with cap direction matched to the real demonstration. The pickup replay subsequently changes only actuator commands; it uses dynamic contacts throughout. The 10 ms finger/bottle contact response supports pickup without deformable bottle modeling. The `aligned_0050` replay lifts both bottles, releases them upright and passes the settled task check. This is a bounded recorded-action check, not validation of every interactive grasp. Native camera targets remain independent of the 240 Hz physics step; the saved replay video uses eight frames per simulation second. Historical three-bottle trials remain in `artifacts/bottle_replay_latest/`.

## Package selected simulation episodes

Use the shared local packager after selecting and checking completed episodes. For USD dependency verification,
run it with the pinned Isaac interpreter and this checkout on `PYTHONPATH`:

```bash
python scripts/package_sim_dataset.py \
  --episodes-file /path/to/selected_episodes.txt \
  --asset-root /path/to/isaac-scenes \
  --out /path/to/new-release \
  --verify-assets \
  --include sim/environment/uv.lock=/path/to/mesa_isaac/uv.lock
```

The selection is one episode-directory path per line, absolute or relative to
the selection file. A bundle contains one task and recorded runtime profile.
Each episode is stored in its own tar shard under `sim/teleop/train/`, preserving
its original identifier and recording bytes. The packager checks the recorded
asset/scene hashes and bundles those dependencies with their relative paths,
resolved task/contact configuration, environment metadata and per-file SHA-256
inventory. Additional configurations or documents are explicit `--include`
attachments. Source code, installed binaries, hidden files and unexpected
recording file types are rejected. Existing output directories are never
overwritten, and packaging never uploads anything. Extract the scene archive
into the environment's `assets/` directory and use `entrypoints.json` to locate
the portable scene descriptor and task config. `--verify-assets` extracts a
temporary copy and verifies that every resolved USD dependency stays inside the
bundle, catching missing files and absolute layer references.
