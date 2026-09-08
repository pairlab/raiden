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
| Joystick click or B (Y) | Calibrate headset→robot axes: hold A (X) and move ~20 cm along the robot's +x, release; hold A (X) and move along the robot's +y, release. Saved to `~/.config/raiden/oculus_calibration.json` and reused; redo it if the headset moves. |
| Hold trigger | Clutch: the end-effector follows the controller relative to where you pressed. Release to re-position your hand. |
| Hold grip | Gripper closed; release to open. |
| A (X) | Trigger event (start/stop recording, record calibration pose). After a recording stops, marks it a success. |
| Joystick click or B (Y) while recording | Stop and mark the recording a failure (also answers the success/failure prompt). |
| Ctrl-C | Emergency stop: hold 5 s, then return home. |

The arm starts from its home pose and holds until the clutch is engaged
with 6-DoF tracking. Tracking loss freezes the arm until it re-locks.

## Options

- `--oculus-hand l|r`: controller driving the left arm (default `l`). The right
  arm always uses the right controller.
- `--oculus-pos-scale 0.7`: robot metres per controller metre.
- `--oculus-rot-scale 0.5`: rotation gain 0..1; `0` keeps the orientation fixed.
- `--oculus-ip <ip>`: Wi-Fi ADB instead of USB (`adb tcpip 5555` once over USB,
  then `adb shell ip route` shows the IP).
