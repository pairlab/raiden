#!/usr/bin/env python3
"""Measure the table plane in the robot base frame by touching it with the fingertips.

rig.json's table height comes from a RANSAC plane through the scene camera's depth
(``real2sim_calibrate.py``). The 2026-09-11 ChArUco fit then put the board 33.4 mm above that
plane although it rests on 25 mm of foam, so the table is about 8 mm higher than the twin
thinks. Depth cannot settle which number is right; contact can.

The arm descends at several table points with the gripper closed until the fingertips stop
following the command. The break in measured-versus-commanded fingertip height is the table
top, free of the millimetre of overshoot a lag threshold would bake in.

Each touch is stored as joint angles, never as a height, so the same run also says which
kinematic chain describes the real arm. Flatness cannot do that on its own: over poses that
touch a horizontal table the two models differ by an almost constant height, so both fit an
equally flat plane. The tool tilt can -- that offset sweeps about 5 mm between a vertical tool
and one tilted 45 degrees, while the table top is a single number. So every point is touched
at several tilts, and the chain that reports the same table height whichever way the wrist is
turned is the one that describes the real arm. The planner's chain cannot bias that: a
touch-off records where the fingertips *were*, and each chain then says where that is.

    # rehearse against the twin, where the table height is known
    #   (vla-benchmark) DISPLAY=:0 MUJOCO_GL=glfw uv run python scripts/raiden_sim_server.py
    uv run python scripts/real2sim_table.py --sim 127.0.0.1:5599

    # the real rig
    uv run python scripts/real2sim_table.py

    # re-fit a saved run, no hardware
    uv run python scripts/real2sim_table.py --fit data/real2sim/table/<run>.npz
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from raiden._xml_paths import get_yam_4310_linear_xml_path
from raiden.real2sim.kinematics import Contact, compare_chains, mesa_chain, vendor_chain

RIG = Path("data/real2sim/calibration/rig.json")
SIM_MODEL = Path("data/processed/cube_stack/0000/sim_model.xml")

# Table points to touch (base frame, metres). Spread across the half of the table the arm
# works over, so the plane is constrained in x and y rather than fitted through one spot.
POINTS = [(0.25, -0.15), (0.35, 0.00), (0.45, 0.15), (0.30, 0.20), (0.40, -0.25)]
# Every point is touched at each of these tool tilts off vertical. This is the axis that tells
# the two kinematic chains apart: they disagree by an almost constant height at a fixed tilt,
# but that offset sweeps about 5 mm across this range, while the table top is one number. 5
# degrees rather than 0 keeps joint 5 off the wrist singularity, where the IK has no unique roll.
TILTS = (5.0, 25.0, 45.0)
# Candidate approach azimuths; the tool roll follows the azimuth, so only some are within
# joint 6's travel at a given point. The reachable ones are found at plan time.
AZIMUTHS = tuple(float(a) for a in range(0, 360, 45))
MIN_CLEARANCE = 0.005  # the fingertips must lead the descent by at least this (m)
TRANSIT_FLOOR = (
    0.010  # every move between touches must stay this far above the table (m)
)
TRANSIT_HEIGHT = (
    0.080  # ... which it does by lifting to here first; joint-space interpolation
)
# between two poses that merely hover is not itself clear of the table
JOINT_MARGIN = 0.05  # keep solutions this far inside the joint limits (rad)


def tool_pose(
    x: float, y: float, z: float, tilt_deg: float, azimuth_deg: float
) -> np.ndarray:
    """Target for ``grasp_site``: fingers down, tilted ``tilt`` degrees towards ``azimuth``."""
    a, t = np.radians(azimuth_deg), np.radians(tilt_deg)
    approach = np.array([np.sin(t) * np.cos(a), np.sin(t) * np.sin(a), -np.cos(t)])
    ref = np.array([np.cos(a), np.sin(a), 0.0])
    side = np.cross(ref, approach)
    side /= np.linalg.norm(side)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([np.cross(side, approach), side, approach])
    T[:3, 3] = [x, y, z]
    return T


def solve_ik(kin, T: np.ndarray, seed: np.ndarray) -> Optional[np.ndarray]:
    """Joint solution for ``grasp_site`` at ``T``, or None if it is out of reach.

    ``Kinematics.ik`` enforces no joint limits unless it is given them, and its success flag
    only reports the pose error, so the limits go in and the result is checked on the way out.
    """
    import mink

    model = kin._configuration.model
    ok, q = kin.ik(
        T, "grasp_site", init_q=seed, limits=[mink.ConfigurationLimit(model)]
    )
    if not ok:
        return None
    lo, hi = model.jnt_range[:6, 0], model.jnt_range[:6, 1]
    q6 = np.asarray(q[: len(lo)], dtype=np.float64)
    if np.any(q6 < lo + JOINT_MARGIN) or np.any(q6 > hi - JOINT_MARGIN):
        return None
    reached = np.asarray(kin.fk(q, "grasp_site"), dtype=np.float64)
    if np.linalg.norm(reached[:3, 3] - T[:3, 3]) > 1e-3:
        return None
    return np.asarray(q, dtype=np.float64)


def solve_probe_pose(
    chain,
    kin,
    x: float,
    y: float,
    probe_z: float,
    tilt: float,
    az: float,
    seed: np.ndarray,
    iters: int = 4,
) -> Optional[np.ndarray]:
    """IK for a pose that puts the *fingertips* at ``probe_z``, not ``grasp_site``.

    grasp_site sits between the fingers, about 10 mm above the closed fingertips, and the gap
    turns with the wrist. Asking the IK for a grasp_site height and calling it a fingertip
    height would start the descent a centimetre below where it was meant to.
    """
    z, q = probe_z, None
    for _ in range(iters):
        q = solve_ik(kin, tool_pose(x, y, z, tilt, az), seed if q is None else q)
        if q is None:
            return None
        err = chain.probe_z(q[:6]) - probe_z
        if abs(err) < 1e-5:
            break
        z -= err
    return q


def plan(
    chain,
    kin,
    points,
    tilts,
    z_start: float,
    z_lift: float,
    seed: np.ndarray,
    per_pose: int = 1,
) -> List[dict]:
    """Solve every probe pose up front and refuse the run if one is unreachable or unsafe.

    Every point is touched at every tilt, so each tilt gets a full plane of its own and the
    tilt comparison is not confounded by where on the table it was measured. Not every azimuth
    is reachable -- the tool roll follows it and joint 6 runs out of travel -- so the reachable
    ones are found here and the run stops if a (point, tilt) has none.

    Heights here are fingertip heights (see :func:`solve_probe_pose`), which is what touches.
    """
    out, q_prev = [], seed
    for x, y in points:
        for tilt in tilts:
            found = []
            for az in AZIMUTHS:
                q = solve_probe_pose(chain, kin, x, y, z_start, tilt, az, q_prev)
                if q is None:
                    continue
                clear = chain.fingertip_clearance(q[:6])
                if clear < MIN_CLEARANCE:
                    continue
                found.append(
                    {
                        "xy": (x, y),
                        "tilt": tilt,
                        "azimuth": az,
                        "q": q[:6].copy(),
                        "clearance": clear,
                    }
                )
                q_prev = q
            if not found:
                raise SystemExit(
                    f"({x:+.3f}, {y:+.3f}) at {tilt:.0f} deg tilt: no approach is reachable with the "
                    "fingertips leading; move the point or drop that tilt"
                )
            for pose in found[:: max(1, len(found) // per_pose)][:per_pose]:
                up = solve_probe_pose(
                    chain,
                    kin,
                    x,
                    y,
                    z_lift,
                    tilt,
                    pose["azimuth"],
                    np.r_[pose["q"], np.zeros(2)],
                )
                if up is None:
                    raise SystemExit(
                        f"({x:+.3f}, {y:+.3f}) at {tilt:.0f} deg: no lift pose above the touch"
                    )
                out.append({**pose, "q_up": up[:6].copy()})
    return out


def move_sequence(poses: List[dict], home: np.ndarray) -> List[np.ndarray]:
    """Joint targets the run commands outside the descents, in order."""
    seq = [home]
    for pose in poses:
        seq += [pose["q_up"], pose["q"], pose["q_up"]]
    seq.append(home)
    return seq


def check_transits(chain, seq: List[np.ndarray], z_floor: float) -> float:
    """Lowest the fingertips get on the joint-space moves between touches.

    ``smooth_move_joints`` interpolates in joint space, so the path between two poses that are
    both clear of the table is not itself guaranteed to be. Checking it costs nothing and turns
    a possible collision into a refusal to start.
    """
    lowest = np.inf
    for a, b in zip(seq, seq[1:]):
        for alpha in np.linspace(0.0, 1.0, 41):
            lowest = min(lowest, chain.probe_z((1 - alpha) * a + alpha * b))
    if lowest < z_floor:
        raise SystemExit(
            f"a move between probe poses dips to {1000 * lowest:.1f} mm, below the {1000 * z_floor:.1f} mm "
            "floor; reorder or thin out POINTS"
        )
    return lowest


def descend(robot, kin, chain, pose: dict, args, table_z: float) -> Optional[Contact]:
    """Step the fingertips down until they stop following, then a little further, then retract."""
    x, y = pose["xy"]
    tilt = pose["tilt"]
    q_cmd = pose["q"].copy()
    gripper = np.array([args.gripper])
    cmd_joints, meas_joints, torque = [], [], []
    z = args.z_start
    baseline_lag, baseline_torque, contact_at = None, None, None

    for step in range(int(args.max_travel / args.step) + 1):
        q = solve_probe_pose(
            chain, kin, x, y, z, tilt, pose["azimuth"], np.r_[q_cmd, np.zeros(2)]
        )
        if q is None:
            print("    IK lost during the descent; retracting")
            break
        q_cmd = q[:6].copy()
        robot.command_joint_pos(np.r_[q_cmd, gripper])
        time.sleep(args.settle)
        obs = robot.get_observations()
        q_meas = np.asarray(obs["joint_pos"], dtype=np.float64)[:6]
        tau = np.asarray(obs["joint_torque"], dtype=np.float64)[:6]
        cmd_joints.append(q_cmd.copy())
        meas_joints.append(q_meas)
        torque.append(tau)

        # Once the table blocks them the fingertips stay above the command, and the gap grows.
        lag = chain.probe_z(q_meas) - chain.probe_z(q_cmd)
        if step == 2:  # settled at the hover pose: everything is measured against this
            baseline_lag, baseline_torque = lag, tau.copy()
        if baseline_torque is not None:
            over = np.abs(tau - baseline_torque).max()
            if over > args.torque_limit:
                print(f"    torque rose {over:.1f} N·m over hover; stopping this point")
                break
        if (
            contact_at is None
            and baseline_lag is not None
            and lag - baseline_lag > args.touch
        ):
            contact_at = z
            print(
                f"    touched at fingertip z {chain.probe_z(q_meas):+.4f} m; pressing {1000 * args.overshoot:.1f} mm on"
            )
        if contact_at is not None and contact_at - z >= args.overshoot:
            break
        z -= args.step
    else:
        print(
            f"    no contact within {1000 * args.max_travel:.0f} mm of travel; skipping this point"
        )

    from raiden.robot.controller import smooth_move_joints

    smooth_move_joints(
        robot,
        np.r_[pose["q"], gripper],
        start_joint_positions=np.r_[q_cmd, gripper],
        time_interval_s=1.5,
        steps=60,
    )
    if contact_at is None:
        return None
    return Contact(
        target=(x, y, table_z),
        tilt_deg=tilt,
        azimuth_deg=pose["azimuth"],
        cmd_joints=np.array(cmd_joints),
        meas_joints=np.array(meas_joints),
        torque=np.array(torque),
    )


def save(contacts: List[Contact], path: Path, meta: dict) -> None:
    n = max(len(c.cmd_joints) for c in contacts)

    def pad(arrays):
        out = np.full((len(contacts), n, 6), np.nan)
        for i, a in enumerate(arrays):
            out[i, : len(a)] = a
        return out

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        meta=json.dumps(meta),
        targets=np.array([c.target for c in contacts]),
        tilts=np.array([c.tilt_deg for c in contacts]),
        azimuths=np.array([c.azimuth_deg for c in contacts]),
        lengths=np.array([len(c.cmd_joints) for c in contacts]),
        cmd_joints=pad([c.cmd_joints for c in contacts]),
        meas_joints=pad([c.meas_joints for c in contacts]),
        torque=pad([c.torque for c in contacts]),
    )
    print(f"\nwrote {path}")


def load(path: Path) -> Tuple[List[Contact], dict]:
    d = np.load(path, allow_pickle=False)
    contacts = []
    for i, n in enumerate(d["lengths"]):
        contacts.append(
            Contact(
                target=tuple(d["targets"][i]),
                tilt_deg=float(d["tilts"][i]),
                azimuth_deg=float(d["azimuths"][i]),
                cmd_joints=d["cmd_joints"][i, :n],
                meas_joints=d["meas_joints"][i, :n],
                torque=d["torque"][i, :n],
            )
        )
    return contacts, json.loads(str(d["meta"]))


def report(
    contacts: List[Contact], chains, rig_table_z: float, truth: Optional[float] = None
) -> dict:
    result = compare_chains(contacts, chains)
    tilts = sorted({c.tilt_deg for c in contacts})
    print(
        f"\n{len(contacts)} contacts at tilts " + ", ".join(f"{t:.0f}°" for t in tilts)
    )

    print(
        "\ntable height by tool tilt (m in base frame) -- the chain that describes the real arm"
    )
    print("reports the same height at every tilt:\n")
    head = "".join(f"{t:>13.0f}°" for t in tilts)
    print(f"{'chain':16s}{head}{'spread':>12s}")
    for name, fit in result.items():
        row = "".join(
            f"{fit['by_tilt'][t]['z_in_base_at_origin']:>14.5f}" for t in tilts
        )
        print(f"{name:16s}{row}{fit['tilt_spread_mm']:>10.2f}mm")

    best = min(result, key=lambda k: result[k]["tilt_spread_mm"])
    worst = max(result, key=lambda k: result[k]["tilt_spread_mm"])
    margin = result[worst]["tilt_spread_mm"] - result[best]["tilt_spread_mm"]
    print(
        f"\n-> {best} is tilt-consistent to {result[best]['tilt_spread_mm']:.2f} mm; "
        f"{worst} drifts {result[worst]['tilt_spread_mm']:.2f} mm"
    )
    if margin < 1.0:
        print(
            "   the two are within 1 mm of each other: this run does not separate them"
        )

    print(
        f"\n{'chain':16s} {'table z (m)':>12s} {'vs rig.json':>13s} {'plane tilt':>11s} "
        f"{'flatness RMS':>14s} {'max':>9s}"
    )
    for name, fit in result.items():
        print(
            f"{name:16s} {fit['z_in_base_at_origin']:>12.5f} "
            f"{1000 * (fit['z_in_base_at_origin'] - rig_table_z):>+11.1f}mm "
            f"{fit['tilt_deg']:>10.3f}° {fit['residual_rms_mm']:>12.2f}mm {fit['residual_max_mm']:>7.2f}mm"
        )

    if truth is not None:
        print(f"\nsim ground truth table z {truth:+.5f} m")
        for name, fit in result.items():
            print(
                f"  {name:16s} error {1000 * (fit['z_in_base_at_origin'] - truth):+6.2f} mm"
            )

    slopes = [c["contact_slope"] for fit in result.values() for c in fit["contacts"]]
    if max(slopes) > 0.5:
        print(
            f"\nwarning: a contact kept following the command (slope {max(slopes):.2f}); "
            "it may have missed the table"
        )
    return result


def write_rig(fit: dict, chain_name: str, rig_path: Path) -> None:
    rig = json.loads(rig_path.read_text())
    old = rig["table"]["z_in_base_at_origin"]
    backup = rig_path.with_suffix(f".json.pre_touchoff_{datetime.now():%Y-%m-%d}.bak")
    shutil.copy2(rig_path, backup)
    rig["table"]["z_in_base_at_origin"] = fit["z_in_base_at_origin"]
    rig["table"]["normal_in_base"] = fit["normal_in_base"]
    rig["table"]["source"] = (
        f"fingertip touch-off, {chain_name} chain, {fit['n_points']} contacts"
    )
    if rig.get("rail"):
        rig["rail"]["top_z_in_base"] = (
            fit["z_in_base_at_origin"] + rig["rail"]["height_above_table"]
        )
    rig_path.write_text(json.dumps(rig, indent=2))
    print(
        f"\n{rig_path}: table z {old:+.5f} -> {fit['z_in_base_at_origin']:+.5f} m (backup {backup.name})"
    )
    print("the layout and the twin are built from this; rebuild both:")
    print(
        "  uv run python scripts/real2sim_layout.py ...        # see markdowns/mesa_env/2.md §8"
    )
    print("  cp data/real2sim/calibration/{rig.json,layout.json,table_atlas.png} \\")
    print("     ../vla-benchmark/mesa/task_suites/rigs/raiden_lab/")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--sim",
        default="",
        help="rehearse against a running raiden_sim_server (host:port)",
    )
    ap.add_argument(
        "--fit", default="", help="re-fit a saved run instead of touching anything"
    )
    ap.add_argument("--write", action="store_true", help="update rig.json from the fit")
    ap.add_argument(
        "--chain",
        choices=["vendor", "mesa"],
        help="which chain's height to adopt with --write",
    )
    ap.add_argument("--rig", type=Path, default=RIG)
    ap.add_argument(
        "--sim-model",
        type=Path,
        default=SIM_MODEL,
        help="any episode's sim_model.xml (MESA chain)",
    )
    ap.add_argument("--out", type=Path, default=Path("data/real2sim/table"))
    ap.add_argument(
        "--tilts",
        type=float,
        nargs="+",
        default=list(TILTS),
        help="tool tilts off vertical to touch at (deg); the chain discriminator",
    )
    ap.add_argument(
        "--z-start",
        type=float,
        default=0.015,
        help="start with the fingertips this far above the current table estimate (m)",
    )
    ap.add_argument("--step", type=float, default=0.00025, help="descent step (m)")
    ap.add_argument(
        "--settle", type=float, default=0.12, help="dwell after each step (s)"
    )
    ap.add_argument(
        "--touch",
        type=float,
        default=0.0004,
        help="fingertip lag that counts as contact (m)",
    )
    ap.add_argument(
        "--overshoot",
        type=float,
        default=0.002,
        help="press this far past first touch (m)",
    )
    ap.add_argument(
        "--max-travel",
        type=float,
        default=0.022,
        help="give up after this much descent (m); with the default --z-start this "
        "bounds how far below the current table estimate a missed touch can press",
    )
    ap.add_argument(
        "--torque-limit",
        type=float,
        default=4.0,
        help="abort a point at this rise over hover (N·m)",
    )
    ap.add_argument(
        "--gripper",
        type=float,
        default=0.0,
        help="gripper command during the touch (0 = closed)",
    )
    args = ap.parse_args()

    rig = json.loads(args.rig.read_text())
    rig_table_z = float(rig["table"]["z_in_base_at_origin"])
    chains = [vendor_chain(), mesa_chain(args.sim_model)]

    if args.fit:
        contacts, meta = load(Path(args.fit))
        result = report(contacts, chains, rig_table_z, meta.get("sim_table_z"))
    else:
        from i2rt.robots.kinematics import Kinematics

        from raiden.robot.controller import RobotController, smooth_move_joints

        kin = Kinematics(get_yam_4310_linear_xml_path(), "grasp_site")
        # Plan every pose before anything moves, and hand the descent the chain the real stack
        # already uses. Detection only needs to see the fingertips stall; the fit is redone
        # per chain afterwards, from the joint angles.
        args.z_start = rig_table_z + args.z_start
        poses = plan(
            chains[0],
            kin,
            POINTS,
            args.tilts,
            args.z_start,
            rig_table_z + TRANSIT_HEIGHT,
            np.zeros(8),
        )
        lowest = check_transits(
            chains[0], move_sequence(poses, np.zeros(6)), rig_table_z + TRANSIT_FLOOR
        )
        print(
            f"{len(poses)} probe poses planned over {len(POINTS)} points x {len(args.tilts)} tilts; "
            f"{1000 * min(p['clearance'] for p in poses):.0f} mm minimum fingertip clearance, "
            f"transits stay {1000 * (lowest - rig_table_z):.0f} mm above the table"
        )

        controller = RobotController(
            use_right_leader=False,
            use_left_leader=False,
            use_right_follower=False,
            use_left_follower=True,
            sim=args.sim or None,
        )
        if not args.sim and not controller.check_can_interfaces():
            raise SystemExit("CAN interfaces are not up; run scripts/reset_all_can.sh")
        controller.initialize_robots()
        robot = controller.follower_l
        sim_table_z = None
        if args.sim:
            sim_table_z = float(robot.get_rig()["table"]["z_in_base_at_origin"])
            print(
                f"rehearsing in sim; its table sits at z {sim_table_z:+.5f} m. The task objects "
                "spawn in the probe region, so a touch that lands on a cube reads tens of mm high."
            )

        contacts = []
        try:
            controller.move_to_home_positions()
            for i, pose in enumerate(poses):
                x, y = pose["xy"]
                print(
                    f"  [{i + 1}/{len(poses)}] ({x:+.2f}, {y:+.2f}) tilt {pose['tilt']:.0f}° "
                    f"azimuth {pose['azimuth']:.0f}°"
                )
                smooth_move_joints(
                    robot,
                    np.r_[pose["q_up"], [args.gripper]],
                    time_interval_s=2.5,
                    steps=120,
                )
                smooth_move_joints(
                    robot,
                    np.r_[pose["q"], [args.gripper]],
                    start_joint_positions=np.r_[pose["q_up"], [args.gripper]],
                    time_interval_s=2.0,
                    steps=100,
                )
                c = descend(robot, kin, chains[0], pose, args, rig_table_z)
                if c is not None:
                    contacts.append(c)
                smooth_move_joints(
                    robot,
                    np.r_[pose["q_up"], [args.gripper]],
                    start_joint_positions=np.r_[pose["q"], [args.gripper]],
                    time_interval_s=2.0,
                    steps=100,
                )
        finally:
            controller.move_to_home_positions()
            controller.close()
        if not contacts:
            raise SystemExit("no contacts recorded")

        meta = {
            "when": datetime.now().isoformat(timespec="seconds"),
            "source": "sim" if args.sim else "real rig",
            "rig_table_z": rig_table_z,
            "sim_table_z": sim_table_z,
            "tilts_deg": list(args.tilts),
            "step_m": args.step,
            "overshoot_m": args.overshoot,
            "gripper": args.gripper,
        }
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save(contacts, args.out / f"{'sim' if args.sim else 'real'}_{stamp}.npz", meta)
        result = report(contacts, chains, rig_table_z, sim_table_z)

    if args.write:
        if not args.chain:
            raise SystemExit(
                "--write needs --chain vendor|mesa: it decides which model the rig files describe"
            )
        name = chains[0].name if args.chain == "vendor" else chains[1].name
        write_rig(result[name], name, args.rig)


if __name__ == "__main__":
    main()
