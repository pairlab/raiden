#!/usr/bin/env python3
"""Measure the table height by resting the fingertips on it under Quest teleop.

``real2sim_table.py``'s automatic touch-off does not work on the real arm: at its slow descent
the joints stick, the fingertips stop following the command in mid-air, and its lag trigger
reads that as contact. Here the operator drives the fingertips onto the table with the Quest
and presses Enter while they rest on it. The measured joints (not the command) then place the
table: pressed onto it, the fingertips can be neither below it nor hanging above it.

Each touch is the lowest fingertip point of the i2rt vendor chain -- the chain the twin
simulates -- at the measured joints and gripper opening, averaged over half a second.

    uv run python scripts/real2sim_table_teleop.py                  # 4 touches, real arm
    uv run python scripts/real2sim_table_teleop.py --sim 127.0.0.1:5599
    uv run python scripts/real2sim_table_teleop.py --fit data/real2sim/table/teleop_<stamp>.json
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from raiden.real2sim.kinematics import fit_plane, vendor_chain

RIG = Path("data/real2sim/calibration/rig.json")
SAMPLE_S = 0.5  # the measured joints are averaged over this window after Enter
STILL_MM = 0.3  # a fingertip moving more than this over the window is not resting


def tool_tilt_deg(chain, q6: np.ndarray, opening: float) -> float:
    """Angle of the gripper's approach axis off straight down."""
    import mujoco

    chain.set_joints(q6, opening)
    site = mujoco.mj_name2id(chain.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    approach_z = chain.data.site_xmat[site].reshape(3, 3)[2, 2]
    return float(np.degrees(np.arccos(np.clip(-approach_z, -1.0, 1.0))))


def evaluate(chain, q6: np.ndarray, opening: float) -> dict:
    p = chain.probe_point(q6, opening)
    return {
        "q": [float(v) for v in q6],
        "gripper": float(opening),
        "probe": [float(v) for v in p],
        "tilt_deg": tool_tilt_deg(chain, q6, opening),
        "clearance_mm": 1000 * chain.fingertip_clearance(q6, opening),
    }


def measure(robot, chain) -> tuple[dict, float]:
    """Median of the measured joints over SAMPLE_S, and how far the fingertip moved meanwhile."""
    samples = []
    t_end = time.monotonic() + SAMPLE_S
    while time.monotonic() < t_end:
        samples.append(np.asarray(robot.get_joint_pos(), dtype=np.float64))
        time.sleep(0.01)
    S = np.array(samples)
    z = [chain.probe_z(s[:6], s[6]) for s in S]
    touch = evaluate(chain, np.median(S[:, :6], axis=0), float(np.median(S[:, 6])))
    return touch, 1000 * (max(z) - min(z))


def report(touches: list, rig_z: float) -> None:
    P = np.array([t["probe"] for t in touches])
    print(f"\n{len(P)} touches, i2rt vendor chain (the twin's), base frame:")
    for i, t in enumerate(touches):
        x, y, z = t["probe"]
        print(
            f"  {i + 1}: x {x:+.3f}  y {y:+.3f}  z {1000 * z:+7.2f} mm   "
            f"tool tilt {t['tilt_deg']:4.1f} deg   gripper {t['gripper']:.2f}"
        )
    z = P[:, 2]
    print(
        f"\ntable z {1000 * z.mean():+.2f} mm (mean; spread {1000 * (z.max() - z.min()):.2f} mm, "
        f"std {1000 * z.std():.2f} mm)   rig.json {1000 * rig_z:+.2f} mm   "
        f"difference {1000 * (z.mean() - rig_z):+.2f} mm"
    )
    if len(P) >= 3:
        plane = fit_plane(P)
        print(
            f"plane through the touches: tilt {plane['tilt_deg']:.3f} deg, "
            f"residual RMS {plane['residual_rms_mm']:.2f} mm"
        )


def save(path: Path, touches: list, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**meta, "touches": touches}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--touches", type=int, default=4)
    ap.add_argument(
        "--sim", default="", help="rehearse against a raiden_sim_server (host:port)"
    )
    ap.add_argument(
        "--fit", default="", help="re-report a saved run instead of touching anything"
    )
    ap.add_argument("--rig", type=Path, default=RIG)
    ap.add_argument("--out", type=Path, default=Path("data/real2sim/table"))
    ap.add_argument("--oculus-hand", choices=["l", "r"], default="l")
    ap.add_argument(
        "--oculus-ip", default="", help="Quest IP for Wi-Fi ADB; empty = USB"
    )
    ap.add_argument(
        "--pos-scale", type=float, default=0.7, help="robot metres per controller metre"
    )
    ap.add_argument("--rot-scale", type=float, default=0.5)
    args = ap.parse_args()

    chain = vendor_chain()
    rig_z = float(json.loads(args.rig.read_text())["table"]["z_in_base_at_origin"])

    if args.fit:
        saved = json.loads(Path(args.fit).read_text())
        # Re-evaluate from the joints, so a changed chain or probe is picked up.
        report(
            [evaluate(chain, np.array(t["q"]), t["gripper"]) for t in saved["touches"]],
            rig_z,
        )
        return

    from raiden.control import build_interface
    from raiden.robot.controller import RobotController

    interface = build_interface(
        "oculus",
        oculus_hand=args.oculus_hand,
        oculus_pos_scale=args.pos_scale,
        oculus_rot_scale=args.rot_scale,
        oculus_ip=args.oculus_ip,
    )
    controller = RobotController(
        use_right_leader=False,
        use_left_leader=False,
        use_right_follower=False,
        use_left_follower=True,
        sim=args.sim or None,
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = args.out / f"{'sim' if args.sim else 'teleop'}_{stamp}.json"
    meta = {
        "when": datetime.now().isoformat(timespec="seconds"),
        "source": "sim" if args.sim else "real rig",
        "chain": chain.name,
        "rig_table_z": rig_z,
    }
    touches: list = []

    interface.open()
    try:
        # Ctrl+C as in rd teleop: hold 5 s, go home, exit. Touches are saved as they are taken.
        signal.signal(signal.SIGINT, lambda *_: controller.emergency_stop())
        signal.signal(signal.SIGTERM, lambda *_: controller.emergency_stop())
        controller.setup_for_teleop_recording()
        interface.setup(controller)
        interface.start(controller)
        print(
            "\n"
            + "=" * 60
            + "\n  TABLE TOUCH-OFF (Quest)\n"
            + "=" * 60
            + "\n  HOLD TRIGGER to move the arm. Bring the fingertips down onto the table,\n"
            "  press a few mm past first contact, release the trigger (the arm keeps\n"
            "  pressing), then press Enter here. Gripper open or closed: either is fine.\n"
            "  Spread the touches over the table. 'u' + Enter drops the last touch.\n"
            "  Ctrl+C: hold 5 s, then home.\n" + "=" * 60
        )
        while len(touches) < args.touches:
            answer = input(
                f"\n[{len(touches) + 1}/{args.touches}] fingertips pressed on the table? Enter: "
            )
            if answer.strip().lower() == "u":
                if touches:
                    touches.pop()
                    save(path, touches, meta)
                    print(f"  dropped touch {len(touches) + 1}")
                continue
            touch, moved_mm = measure(controller.follower_l, chain)
            z_mm = 1000 * touch["probe"][2]
            if moved_mm > STILL_MM:
                print(
                    f"  the fingertip moved {moved_mm:.2f} mm while measuring; hold still and retry"
                )
                continue
            if touch["clearance_mm"] < 0:
                print(
                    f"  another part of the arm is {-touch['clearance_mm']:.1f} mm below the "
                    "fingertips; point the fingers down more and retry"
                )
                continue
            touches.append(touch)
            save(path, touches, meta)
            print(
                f"  touch {len(touches)}: fingertip z {z_mm:+.2f} mm (rig.json {1000 * rig_z:+.2f}), "
                f"tool tilt {touch['tilt_deg']:.1f} deg"
            )

        interface.stop(controller)
        controller.shutdown()
    except Exception as e:
        print(f"\nError: {e}")
        if controller.has_robots():
            controller.emergency_stop()
        raise
    finally:
        interface.close()

    print(f"\nwrote {path}")
    report(touches, rig_z)
    os._exit(0)  # the robot and Quest threads outlive close(), as in rd teleop


if __name__ == "__main__":
    main()
