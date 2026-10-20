#!/usr/bin/env python3
"""Prepare explicit sim episodes offline, then launch attended hardware trials.

No driver/camera is opened by prepare, check, or run without --execute. Execution
requires an attended terminal: --execute authorizes the usual public YAM
constructor, which enables motors and calibrates/moves the gripper. Each task requires RUN <id>
MATCHED_START. There is no automatic task, reset, home, shutdown, or episode loop.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import select
import shutil
import signal
import sys
import termios
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from raiden.robot_poses import canonical_home
from raiden.sim.hardware_replay import TIMING_PROFILES
from raiden.sim.hardware_session import (
    HealthyFollower,
    NativeControllerRecorder,
    jsonable,
    write_json,
)
from raiden.sim.real_replay import (
    ancestor_pids,
    controller_processes,
    prepare_selection,
    run_episode,
    validate_prepared,
)


def announce(*args, **kwargs):
    """A closed terminal/output pipe must not retire a native hold owner."""
    try:
        print(*args, **kwargs)
    except OSError:
        pass


@contextmanager
def attended_terminal():
    """Repair inherited raw/no-echo input for this process, restoring it on exit."""
    original, fd = None, None
    try:
        fd = sys.stdin.fileno()
        original = termios.tcgetattr(fd)
        configured = list(original)
        configured[0] = (configured[0] | termios.ICRNL) & ~(
            termios.IGNCR | termios.INLCR
        )
        configured[3] |= termios.ECHO | termios.ICANON | termios.ISIG
        if configured != original:
            termios.tcsetattr(fd, termios.TCSANOW, configured)
            announce(
                "Restored line input and local echo for the attended command loop.",
                flush=True,
            )
    except (OSError, ValueError, AttributeError):
        # Test/file-like streams have no terminal attributes. The public entry
        # point independently requires isatty before hardware initialization.
        original = None
    try:
        yield
    finally:
        if original is not None:
            try:
                termios.tcsetattr(fd, termios.TCSANOW, original)
            except OSError:
                pass


def require_no_controller(channel):
    blockers = controller_processes(channel, exclude=ancestor_pids())
    if blockers:
        raise RuntimeError(
            f"existing possible hardware owners must be retired in their owning terminal: {blockers}"
        )
    return blockers


def wait_for_power_off(output, stop, reason):
    """A failed diagnostic/input path must not retire an initialized controller."""
    stop.set()
    try:
        write_json(
            output / "owner_failure.json",
            {
                "error": reason,
                "controller_retained_until_physical_power_off": True,
                "host_wall_ns": time.time_ns(),
            },
        )
    except Exception as error:
        announce(f"Owner failure receipt could not be written: {error}", flush=True)
    announce(
        f"Owner retained after {reason}. Physically power off motors, then type SHUTDOWN_POWER_OFF_CONFIRMED or touch {output / 'POWER_OFF_CONFIRMED'}.",
        flush=True,
    )
    readable_input = True
    while not (output / "POWER_OFF_CONFIRMED").exists():
        if readable_input:
            try:
                readable, _, _ = select.select([sys.stdin], [], [], 0.2)
                if readable:
                    line = sys.stdin.readline()
                    if line.strip() == "SHUTDOWN_POWER_OFF_CONFIRMED":
                        return
                    if not line:
                        readable_input = False
            except (OSError, ValueError):
                readable_input = False
        else:
            time.sleep(0.1)


def owner_loop(args, manifest, recorder, follower, stop):
    action = None
    counter = 0
    stdin_open = True
    episodes = {row["id"]: row for row in manifest["episodes"]}

    def stopped():
        return stop.is_set() or (args.out / "STOP").exists()

    def hold():
        obs = follower.get_observations()
        follower.command_joint_pos(np.r_[obs["joint_pos"], obs["gripper_pos"]])

    def status():
        report = {
            "host_wall_ns": time.time_ns(),
            "host_monotonic_ns": time.monotonic_ns(),
            "health": recorder.health(follower.robot),
            "action_running": action is not None and action.is_alive(),
            "robot_info": follower.get_robot_info(),
            "observation": follower.get_observations(),
            "selected_episode_ids": list(episodes),
            "stop_file": str(args.out / "STOP"),
        }
        write_json(args.out / "status.json", report)
        return report

    def motion(command, episode_id, number):
        destination = (
            args.out / f"{number:03d}_{command.lower()}_{episode_id or 'home'}"
        )
        try:
            if command == "RESET":
                # Deliberately outside comparison streams; only an explicit RESET
                # requests this normal teleop home motion.
                destination.mkdir()
                target, source = canonical_home()
                obs = follower.get_observations()
                start_q = np.r_[obs["joint_pos"], obs["gripper_pos"]]
                duration = max(2.0, float(np.max(abs(start_q[:6] - target[:6]))) / 0.2)
                began = time.monotonic()
                while True:
                    if stopped():
                        raise InterruptedError("operator stopped reset")
                    alpha = min(1.0, (time.monotonic() - began) / duration)
                    follower.command_joint_pos((1.0 - alpha) * start_q + alpha * target)
                    if alpha >= 1.0:
                        break
                    time.sleep(0.01)
                write_json(
                    destination / "reset.json",
                    {
                        "source": source,
                        "target": target,
                        "initial_observation": obs,
                        "final_observation": follower.get_observations(),
                        "duration_s": duration,
                        "comparison_task": False,
                    },
                )
            else:
                config = json.loads(Path(manifest["camera_config"]).read_text())
                result = run_episode(
                    args.prepared / episodes[episode_id]["plan"],
                    follower,
                    recorder,
                    destination,
                    config,
                    stop_requested=stopped,
                    timing_profile=args.timing_profile,
                    arm_start_tolerance_rad=args.arm_start_tolerance,
                    aperture_start_tolerance=args.aperture_start_tolerance,
                    velocity_start_tolerance=args.velocity_start_tolerance,
                    enforce_initial_state=False,
                )
                announce(
                    json.dumps(
                        {
                            "trial": str(destination),
                            "status": result["status"],
                            "events_sent": result["execution"]["events_sent"],
                            "source_duration_s": result["execution"][
                                "source_duration_ns"
                            ]
                            / 1e9,
                            "measured_duration_s": result["execution"][
                                "stream_wall_duration_ns"
                            ]
                            / 1e9,
                        }
                    ),
                    flush=True,
                )
        except BaseException as error:
            try:
                hold()
            except Exception as hold_error:
                announce(f"Observed-pose hold unavailable: {hold_error}", flush=True)
            try:
                destination.mkdir(exist_ok=True)
                write_json(
                    destination / "failure.json",
                    {
                        "exception": f"{type(error).__name__}: {error}",
                        "next_action": "inspect trial.json, camera receipts, measured_start.json and rollout/execution.json; task never retries automatically",
                    },
                )
            except Exception as save_error:
                announce(
                    f"Failure receipt could not be written: {save_error}", flush=True
                )
            announce(f"{command} failed: {error}. Evidence: {destination}", flush=True)
        finally:
            announce(
                "Action ended. Native controller remains active; place objects manually before another RUN.",
                flush=True,
            )

    announce(
        "Commands: STATUS | RESET | RUN <id> MATCHED_START | HOLD | SHUTDOWN_POWER_OFF_CONFIRMED",
        flush=True,
    )
    announce(
        "RUN asserts: attended stop available, source objects aligned, oven closed, arm/gripper start reviewed; fresh controller history accepted.",
        flush=True,
    )
    announce(json.dumps(jsonable(status())), flush=True)
    while True:
        busy = action is not None and action.is_alive()
        if (args.out / "POWER_OFF_CONFIRMED").exists():
            if busy:
                stop.set()
            else:
                return
        if stdin_open:
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not readable:
                continue
            line = sys.stdin.readline()
            if not line:
                stdin_open = False
                stop.set()
                announce(
                    f"Input closed. Task stop requested; controller retained. After physically powering off motors, touch {args.out / 'POWER_OFF_CONFIRMED'} to retire this owner.",
                    flush=True,
                )
                continue
        else:
            time.sleep(0.1)
            continue
        parts = line.strip().split()
        if not parts:
            continue
        command = parts[0].upper()
        try:
            if command == "STATUS":
                announce(json.dumps(jsonable(status())), flush=True)
            elif command == "HOLD":
                stop.set()
                if not busy:
                    hold()
            elif command == "SHUTDOWN_POWER_OFF_CONFIRMED":
                if busy:
                    stop.set()
                    raise ValueError(
                        "task stopping; issue shutdown confirmation again after it ends"
                    )
                return
            elif command in ("RUN", "RESET"):
                if busy:
                    raise ValueError("another motion is running")
                if command == "RUN":
                    if (
                        len(parts) != 3
                        or parts[1] not in episodes
                        or parts[2] != "MATCHED_START"
                    ):
                        raise ValueError(
                            f"use RUN <selected-id> MATCHED_START; selected: {list(episodes)}"
                        )
                    episode_id = parts[1]
                    # Revalidate all source bindings immediately before each run.
                    validate_prepared(args.prepared)
                else:
                    episode_id = None
                    if len(parts) != 1:
                        raise ValueError("RESET takes no arguments")
                require_no_controller(args.channel)
                follower._check()
                if (args.out / "STOP").exists():
                    raise ValueError(
                        "remove the STOP file yourself before launching another motion"
                    )
                stop.clear()
                counter += 1
                action = threading.Thread(
                    target=motion, args=(command, episode_id, counter), daemon=True
                )
                args._active_action = action
                action.start()
            else:
                raise ValueError(
                    "unknown command; use STATUS, RESET, RUN <id> MATCHED_START, HOLD or SHUTDOWN_POWER_OFF_CONFIRMED"
                )
        except Exception as error:
            announce(f"Command refused: {error}", flush=True)


def execute(args, manifest):
    if args.out is None:
        raise ValueError("--execute requires a new --out directory")
    if not sys.stdin.isatty():
        raise ValueError("--execute requires an attended terminal")
    require_no_controller(args.channel)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    # Same lock used by fresh/calibration owners. Ordinary rd/i2rt users may not
    # cooperate; the process audit above is additional evidence, not a proof.
    with (
        open(f"/tmp/raiden-parity-{args.channel}.lock", "a") as lock,
        attended_terminal(),
    ):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        announce(
            "--execute: initializing the public YAM driver, including gripper calibration. Keep the workspace attended.",
            flush=True,
        )
        require_no_controller(args.channel)
        stop = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        recorder = NativeControllerRecorder()
        write_json(
            args.out / "owner.json",
            {
                "pid": os.getpid(),
                "channel": args.channel,
                "prepared": str(args.prepared.resolve()),
                "argv": sys.argv,
                "host_wall_ns": time.time_ns(),
                "host_monotonic_ns": time.monotonic_ns(),
                "stop_file": str(args.out / "STOP"),
                "power_off_confirmation_file": str(args.out / "POWER_OFF_CONFIRMED"),
                "one_writer_lock": lock.name,
                "initialization": "ordinary public YAM driver, gripper calibration enabled",
                "exclusive_control_evidence": "cooperative lock plus known live process scan; unrelated driver users must remain stopped",
                "controller_history": "fresh native history, not source-simulator history",
                "numerical_initial_state_check_enforced": False,
                "initial_state_policy": "disabled at user request; operator accepts physical reset; source and measured states retained separately",
                "postrun": "native pose hold remains; shutdown only after physical motor power-off confirmation",
            },
        )
        try:
            # Hardware modules are imported only beyond both explicit execution gates.
            from i2rt.motor_drivers.dm_driver import (
                DMChainCanInterface,
                DMSingleMotorCanInterface,
            )
            from i2rt.robots.get_robot import get_yam_robot
            from i2rt.robots.motor_chain_robot import MotorChainRobot
            from i2rt.robots.utils import GripperType

            recorder.install(
                DMSingleMotorCanInterface, MotorChainRobot, DMChainCanInterface
            )
            robot = get_yam_robot(
                channel=args.channel,
                gripper_type=GripperType.LINEAR_4310,
                zero_gravity_mode=False,
            )
            follower = HealthyFollower(robot, recorder)
            time.sleep(0.3)
            follower._check()
            obs = follower.get_observations()
            follower.command_joint_pos(np.r_[obs["joint_pos"], obs["gripper_pos"]])
        except BaseException as error:
            try:
                write_json(
                    args.out / "initialization_failed.json",
                    {
                        "exception": f"{type(error).__name__}: {error}",
                        "controller_may_have_started": True,
                        "task_started": False,
                    },
                )
                recorder.flush(args.out / "initialization_trace")
            except Exception as save_error:
                announce(f"Initialization diagnostic failed: {save_error}", flush=True)
            wait_for_power_off(
                args.out,
                stop,
                f"initialization failure: {type(error).__name__}: {error}",
            )
        else:
            try:
                owner_loop(args, manifest, recorder, follower, stop)
            except BaseException as error:
                stop.set()
                active = getattr(args, "_active_action", None)
                if active is not None and active.is_alive():
                    active.join(timeout=10.0)
                    if active.is_alive():
                        announce(
                            "Task worker did not finalize within 10 seconds; retain incomplete camera/native receipts.",
                            flush=True,
                        )
                wait_for_power_off(
                    args.out,
                    stop,
                    f"owner input/status failure: {type(error).__name__}: {error}",
                )
        try:
            recorder.flush(args.out / "final_owner_trace")
        finally:
            recorder.stop_threads_after_confirmed_power_off()
            write_json(
                args.out / "shutdown.json",
                {
                    "operator_confirmed_physical_motor_power_off": True,
                    "host_wall_ns": time.time_ns(),
                },
            )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser(
        "prepare", help="offline exact held-command export and validation"
    )
    prepare.add_argument("--source-root", type=Path, required=True)
    prepare.add_argument(
        "--episodes",
        nargs="+",
        required=True,
        help="explicit directory IDs; no latest/glob inference",
    )
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument(
        "--camera-config", type=Path, default=Path.home() / ".config/raiden/camera.json"
    )
    prepare.add_argument("--expected-state", default="parity_s2")
    prepare.add_argument("--accepted-calibration-id", default="20261018_175242")
    for name in ("check", "run"):
        child = sub.add_parser(
            name,
            help="offline preflight"
            if name == "check"
            else "attended owner; offline unless --execute",
        )
        child.add_argument("--prepared", type=Path, required=True)
        child.add_argument("--channel", default="can_follower_l")
        if name == "run":
            child.add_argument("--out", type=Path)
            child.add_argument("--execute", action="store_true")
            child.add_argument(
                "--timing-profile",
                choices=tuple(TIMING_PROFILES),
                default="best_effort_unbounded",
            )
            child.add_argument(
                "--arm-start-tolerance",
                type=float,
                default=0.01,
                help="diagnostic only; numerical start rejection disabled at user request",
            )
            child.add_argument(
                "--aperture-start-tolerance",
                type=float,
                default=0.02,
                help="diagnostic only; numerical start rejection disabled at user request",
            )
            child.add_argument(
                "--velocity-start-tolerance",
                type=float,
                default=0.05,
                help="diagnostic only; numerical start rejection disabled at user request",
            )
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare_selection(
            args.source_root,
            args.episodes,
            args.out,
            expected_state=args.expected_state,
            accepted_calibration_id=args.accepted_calibration_id,
            camera_config=args.camera_config,
        )
        announce(
            json.dumps(
                {
                    "prepared": str(args.out),
                    "hardware_initialized": False,
                    "episodes": [
                        {k: row[k] for k in ("id", "duration_ns", "request_events")}
                        for row in manifest["episodes"]
                    ],
                },
                indent=2,
            )
        )
        return
    manifest = validate_prepared(args.prepared)
    report = {
        "status": "offline preflight passed",
        "hardware_initialized": False,
        "episodes": [row["id"] for row in manifest["episodes"]],
        "accepted_calibration_id": manifest["accepted_calibration_id"],
        "free_disk_bytes": shutil.disk_usage(
            args.out.parent
            if getattr(args, "out", None) and args.out.parent.exists()
            else args.prepared
        ).free,
        "possible_controller_processes": controller_processes(
            args.channel, exclude=ancestor_pids()
        ),
        "device_and_camera_readiness": "not checked; no device was opened",
        "numerical_initial_state_check_enforced": False,
        "timing": "best_effort profiles retain each original minimum hold and measure accumulated duration dilation; strict uses absolute deadlines",
    }
    announce(json.dumps(report, indent=2), flush=True)
    if args.command == "run" and args.execute:
        for value in (
            args.arm_start_tolerance,
            args.aperture_start_tolerance,
            args.velocity_start_tolerance,
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError("start tolerances must be finite and nonnegative")
        execute(args, manifest)


if __name__ == "__main__":
    main()
