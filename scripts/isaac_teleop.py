#!/usr/bin/env python3
"""Run one Isaac scene with the existing Quest teleop or recording interface.

The simulator and Raiden retain separate pinned Python environments. This entry
point owns the simulator lifetime and passes the same explicit arm mode to both.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

from raiden.sim.launch import IsaacServer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--arms", required=True, choices=("single", "bimanual"))
    parser.add_argument("--task-config", type=Path)
    parser.add_argument(
        "--isaac-project", default=os.environ.get("MESA_ISAAC_PROJECT"), type=Path
    )
    parser.add_argument(
        "--isaac-python", default=os.environ.get("MESA_ISAAC_PYTHON"), type=Path
    )
    parser.add_argument("--mode", choices=("record", "teleop"), default="record")
    parser.add_argument("--control", choices=("oculus", "spacemouse"), default="oculus")
    parser.add_argument("--port", type=int, default=5599)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--visualizer", choices=("none", "kit"), default="none")
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate server startup without connecting any input device",
    )
    args, extra = parser.parse_known_args(argv)
    if not args.isaac_project or not args.isaac_python:
        parser.error(
            "provide --isaac-project/--isaac-python or MESA_ISAAC_PROJECT/MESA_ISAAC_PYTHON"
        )
    if any(
        value.split("=")[0] in ("--sim", "--arms", "--control", "--sim-fps")
        for value in extra
    ):
        parser.error(
            "server address, arm mode, control and camera rate are managed by this launcher"
        )

    def terminate(*_):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, terminate)
    with IsaacServer(
        scene=args.scene,
        arms=args.arms,
        project=args.isaac_project,
        python=args.isaac_python,
        task_config=args.task_config,
        port=args.port,
        fps=args.fps,
        seed=args.seed,
        startup_timeout=args.startup_timeout,
        log=args.server_log,
        visualizer=args.visualizer,
    ) as server:
        if args.check:
            print(
                f"Isaac scene ready: {args.scene}; arms={args.arms}; RPC={server.address}"
            )
            return 0
        command = [
            sys.executable,
            "-m",
            "raiden.cli",
            args.mode,
            "--control",
            args.control,
            "--sim",
            server.address,
            "--arms",
            args.arms,
            "--sim-fps",
            str(args.fps),
            *extra,
        ]
        client = subprocess.Popen(command, start_new_session=True)
        try:
            return client.wait()
        except KeyboardInterrupt:
            os.killpg(client.pid, signal.SIGINT)
            try:
                return client.wait(timeout=10)
            except subprocess.TimeoutExpired:
                return 130
        finally:
            if client.poll() is None:
                os.killpg(client.pid, signal.SIGTERM)
                try:
                    client.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(client.pid, signal.SIGKILL)
                    client.wait()


if __name__ == "__main__":
    raise SystemExit(main())
