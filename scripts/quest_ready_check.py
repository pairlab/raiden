#!/usr/bin/env python3
"""Run the Quest interface as it runs at `rd record`'s READY prompt, without the robot.

Prints every button edge with the controller's tracking state, plus the interface's own
calibration messages, for `--seconds` (default 40).  A finished calibration is saved to
~/.config/raiden/oculus_calibration.json, as in `rd record`.

    uv run python scripts/quest_ready_check.py            # left controller
    uv run python scripts/quest_ready_check.py --hand r
"""

import argparse
import threading
import time

import numpy as np

from raiden.control.oculus import _KEYS, OculusInterface


class _NoRobot:
    """Stands in for RobotController: calls the control law at 100 Hz, moves nothing."""

    def __init__(self, side: str):
        self.follower_l = object() if side == "left" else None
        self.follower_r = object() if side == "right" else None
        self._side = side
        self._stop = threading.Event()

    def start_cartesian_teleop(self, target_fn, dt: float = 0.01) -> None:
        T = np.eye(4)
        T[:3, 3] = [0.3, 0.0, 0.2]

        def loop() -> None:
            while not self._stop.is_set():
                target_fn(self._side, T, 1.0)
                time.sleep(dt)

        threading.Thread(target=loop, daemon=True).start()

    def stop_cartesian_teleop(self) -> None:
        self._stop.set()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--hand", default="l", choices=["l", "r"])
    ap.add_argument("--seconds", type=float, default=40.0)
    args = ap.parse_args()

    it = OculusInterface(hand_for_left_arm=args.hand)
    it.open()
    robot = _NoRobot("left")
    it.start_ready(robot)
    time.sleep(1.5)  # first OS tracking reading
    print(it.ready_hint)
    k = _KEYS[args.hand]
    watch = (k["btn"], *k["calib"], k["clutch"], k["grip"])
    print(
        f"\nPress {k['calib'][1]} (or the joystick), then hold {k['btn']} and move along robot +x, "
        f"release; then +y. Watching {', '.join(watch)} for {args.seconds:.0f} s.\n",
        flush=True,
    )
    prev, calibrating, t0 = {}, False, time.monotonic()
    while time.monotonic() - t0 < args.seconds:
        _, buttons = it._reader.get()
        for key in watch:
            down = bool(buttons.get(key, False))
            if down != prev.get(key, False):
                print(
                    f"  {time.monotonic() - t0:5.1f} s  {key} {'down' if down else 'up'}"
                    f"   tracking {it._tracking.get(args.hand)}, data age {it._reader.age:.2f} s",
                    flush=True,
                )
            prev[key] = down
        if it.calibrating != calibrating:
            calibrating = it.calibrating
            print(f"  calibrating: {calibrating}", flush=True)
        if it.poll(robot):
            print(f"  {k['btn']}: start event (would start the recording)", flush=True)
        time.sleep(0.005)
    robot.stop_cartesian_teleop()
    it.close()


if __name__ == "__main__":
    main()
