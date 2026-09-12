#!/usr/bin/env python3
"""Teleoperate a single YAM arm with one Meta Quest controller (no recording).

    uv run python scripts/quest_teleop.py                 # left controller, USB
    uv run python scripts/quest_teleop.py --hand r        # right controller
    uv run python scripts/quest_teleop.py --ip 10.0.0.42  # Wi-Fi ADB
    uv run python scripts/quest_teleop.py --sim           # MESA digital twin (start raiden_sim_server first)

See docs/guide/oculus_teleop.md for setup and controls.
"""

import argparse

from raiden.robot.teleop import run_bimanual_teleop


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--hand",
        default="l",
        choices=["l", "r"],
        help="controller to use (default: left)",
    )
    ap.add_argument(
        "--pos-scale", type=float, default=0.7, help="robot metres per hand metre"
    )
    ap.add_argument(
        "--rot-scale",
        type=float,
        default=0.5,
        help="rotation gain 0..1 (0 = translation only)",
    )
    ap.add_argument("--ip", default="", help="Quest IP for Wi-Fi ADB (default: USB)")
    ap.add_argument(
        "--sim",
        nargs="?",
        const="127.0.0.1:5599",
        default="",
        help="drive the MESA digital twin (address of a running raiden_sim_server)",
    )
    args = ap.parse_args()

    run_bimanual_teleop(
        control="oculus",
        arms="single",
        oculus_hand=args.hand,
        oculus_pos_scale=args.pos_scale,
        oculus_rot_scale=args.rot_scale,
        oculus_ip=args.ip,
        sim=args.sim,
    )


if __name__ == "__main__":
    main()
