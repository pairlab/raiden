# Quest controller teleoperation

`rd teleop --control oculus` and `rd record --control oculus` drive the follower
arms from Meta Quest hand controllers. `scripts/quest_teleop.py` is a
single-arm shortcut with no recording or prompts. Place the headset on a desk
facing you, or secure it at your neck/chest with its cameras facing outward
toward both controllers. Keep the cameras unobstructed and prevent the headset
from swinging or slipping.

## Neck/chest mounting and reference frame

The bundled app sends each controller pose **relative to the headset**:
`T_head_controller = inverse(T_world_head) @ T_world_controller`.
This is explicit in the [upstream app source](https://github.com/rail-berkeley/oculus_reader/blob/main/app_source/Src/OculusTeleop.cpp#L1140-L1149).
Moving or turning the headset while a trigger is held therefore changes the
robot's target, even if your hand stays still in the room. A chest mount supports
operation with a steady torso; there is no compensation for torso motion and
no room-fixed control mode in this APK.

**Release both triggers before shifting your torso, turning, or adjusting the
mount.** Re-engage only after the headset is steady. Redo axis calibration at
READY if its orientation relative to the robot changed, including when switching
between desk and chest placement. Translation alone does not change the axis
calibration. The fit supports arbitrary fixed yaw, pitch, and roll; it does not
measure headset movement during operation.

## One-time setup

1. Enable **Developer Mode** on the headset (Meta phone app → headset → Developer Mode).
2. `sudo apt install android-tools-adb`, plug the headset in over USB, and accept
   *Allow USB debugging* on the headset. `adb devices` must list it as `device`.
3. Set up a Guardian boundary once so tracking is allowed. The teleop app
   (`third_party/oculus_reader/teleop-debug.apk`) is installed automatically on first run.

Raiden already sends the Quest's keep-awake `prox_close` broadcast when connecting.
If the headset still sleeps away from your face, use **Meta Quest Developer Hub →
Device Actions → Proximity Sensor → off**, as described in the
[MQDH device-actions guide](https://developers.meta.com/vr/documentation/unity/ts-mqdh-basic-usage/).
With a chest mount, check that both controllers remain visible to the headset
throughout the intended workspace, including low reaches over the table.
Neck/chest operation has been tested offline and still needs validation on the rig.

## Controls

The layout matches [mesa-env](https://github.com/pairlab/mesa-env). Left
controller buttons in parentheses.

| Input | Action |
|---|---|
| Joystick click or B (Y) | Start or cancel axis calibration for that controller; follow the +x/+y motion prompts below. |
| Hold trigger | Clutch: the end-effector follows the controller relative to where you pressed. Release to re-position your hand. |
| Hold grip | Gripper closed; release to open. |
| A (X) | Trigger event (start/stop recording, record calibration pose). After a recording stops, marks it a success. |
| Joystick click or B (Y) at the success/failure prompt | Mark the recording a failure. |
| Ctrl-C | Emergency stop: hold 5 s, then return home. |

The arm starts from its home pose and holds until the clutch is engaged
with 6-DoF tracking. Detected tracking loss, missing or nonfinite controller
poses, or a stale stream discard the clutch and smoothing reference. When data
and tracking return, a held trigger establishes a new reference at the current arm pose;
motion made during the gap is not replayed. OS tracking state is polled about
once a second. Release the triggers when tracking is unreliable.

## Axis calibration

At the READY prompt, keep the headset and torso still and calibrate **each
active controller separately**. For bimanual control, do both left and right;
one controller's calibration does not calibrate the other.

1. On the controller being calibrated, click its joystick or press Y (left) / B
   (right).
2. Hold X (left) / A (right), move about 20 cm along that arm base's **+x**
   (straight out from its base), then release the button.
3. Hold the same button, move along that arm base's **+y** (its left), then release.

The arm holds while calibrating. Full measured motion vectors are fitted to a
right-handed rotation, allowing a tilted mount without snapping to headset axes.
Moves shorter than 8 cm, invalid poses, interrupted tracking/data, and +x/+y
directions outside 60°–120° are rejected; repeat the requested move. Press the
joystick or Y/B again to cancel and retain the previous calibration.

The per-controller 3×3 rotations are saved to
`~/.config/raiden/oculus_calibration.json` and reused. Existing valid calibrations
retain their format and load normally; recalibrate after remounting or changing
headset orientation. To check buttons, tracking, and calibration without connecting
to either robot, run `python scripts/quest_ready_check.py --hand l`, then repeat
with `--hand r`. This check connects to the Quest and saves completed calibration.

This axis calibration maps controller motion to each arm's base frame. It is
independent of [camera calibration](calibration.md): it neither changes camera
intrinsics nor estimates camera-to-robot or camera-to-tool transforms.

The terminal prints each controller's tracking state when it changes:

- `POSITION`: tracked (6-DoF).
- `ORIENTATION`: the headset cameras cannot see the controller. Hold it in front of the headset.
- `NONE`: the controller is asleep or off. Press any button on it.

It also says when no data arrives from the Quest app. The reader reopens a
dropped adb stream on its own.

## Real recording

`rd record --control oculus` starts teleop at the READY prompt with the arm
held at home. The prompt shows each controller's tracking state and whether
it is calibrated. Calibrate there if needed; a start is ignored until the
calibration is done or cancelled. A (X) or Enter starts the recording, and
A (X) or Enter stops it. Then A (X) marks a success and the joystick or
B (Y) a failure. During a recording the calibration buttons do nothing.

## Sim collection

`rd record --control oculus --arms single --sim 127.0.0.1:5599` records from every
scene reset. There is no start button and no success/failure prompt; only a success
is saved.

| Input | Action |
|---|---|
| B (Y) | Save the episode as a success, reset the scene, record the next one. |
| A (X) | Discard the episode, reset the scene, record the next one. |
| Joystick click | Discard the episode and calibrate (B/Y is taken). The scene resets when the calibration is done or cancelled. |
| Enter / `r` / `q` | Save / discard / discard and end the session. |
| Middle / right pedal | Save / discard. |

A reset snaps the arm home and re-samples the objects. Only saved episodes keep a
directory, so episode numbers stay contiguous.

With `--monitor`, the window shows the `operator` view (behind the arm), the
`topdown` view (above the table, angled down) and the wrist camera, with the
partitions hidden. The views are for the operator only and are never recorded;
`--monitor-view` picks other operator views (comma-separated, e.g.
`--monitor-view leftshoulder,topdown`), and `--monitor-view none` shows the
recorded scene camera. The terminal prints each subtask as the sim completes it,
and a note if you save an episode the sim does not score a success.

## Options

- `--oculus-hand l|r`: controller driving the left arm (default `l`). The right
  arm always uses the right controller; use `l` for bimanual operation so the
  followers have distinct controllers. See [Bimanual Quest control](recording.md#bimanual-quest-control).
- `--oculus-pos-scale 0.7`: robot metres per controller metre.
- `--oculus-rot-scale 0.5`: rotation gain 0..1; `0` keeps the orientation fixed.
- `--oculus-ip <ip>`: Wi-Fi ADB instead of USB (`adb tcpip 5555` once over USB,
  then `adb shell ip route` shows the IP).
