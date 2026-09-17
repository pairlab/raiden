# Quest controller teleoperation

`rd teleop --control oculus` and `rd record --control oculus` drive the follower
arms from Meta Quest hand controllers. `scripts/quest_teleop.py` is a
single-arm shortcut with no recording or prompts. The headset is not worn:
place it on the desk with its cameras facing you so it can track the controllers.

## One-time setup

1. Enable **Developer Mode** on the headset (Meta phone app → headset → Developer Mode).
2. `sudo apt install android-tools-adb`, plug the headset in over USB, and accept
   *Allow USB debugging* on the headset. `adb devices` must list it as `device`.
3. Set up a Guardian boundary once so tracking is allowed. The teleop app
   (`third_party/oculus_reader/teleop-debug.apk`) is installed automatically on first run.

## Controls

The layout matches [mesa-env](https://github.com/pairlab/mesa-env). Left
controller buttons in parentheses.

| Input | Action |
|---|---|
| Joystick click or B (Y) | Calibrate headset→robot axes: hold A (X) and move ~20 cm along the robot's +x, release; hold A (X) and move along the robot's +y, release. Press it again to cancel. Saved to `~/.config/raiden/oculus_calibration.json` and reused in every session; redo it only if the headset moves. |
| Hold trigger | Clutch: the end-effector follows the controller relative to where you pressed. Release to re-position your hand. |
| Hold grip | Gripper closed; release to open. |
| A (X) | Trigger event (start/stop recording, record calibration pose). After a recording stops, marks it a success. |
| Joystick click or B (Y) at the success/failure prompt | Mark the recording a failure. |
| Ctrl-C | Emergency stop: hold 5 s, then return home. |

The arm starts from its home pose and holds until the clutch is engaged
with 6-DoF tracking. Tracking loss freezes the arm until it re-locks.
A calibration move counts only if the headset tracks the controller from
press to release.

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
  arm always uses the right controller.
- `--oculus-pos-scale 0.7`: robot metres per controller metre.
- `--oculus-rot-scale 0.5`: rotation gain 0..1; `0` keeps the orientation fixed.
- `--oculus-ip <ip>`: Wi-Fi ADB instead of USB (`adb tcpip 5555` once over USB,
  then `adb shell ip route` shows the IP).
