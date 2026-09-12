#!/usr/bin/env python3
"""End-to-end check of the digital twin without a Quest: record one scripted episode.

Drives the sim follower through the *real* Cartesian teleop path (mink IK, gripper slew,
100 Hz command loop), records it with the real recorder (sim cameras at 30 fps), converts
it and, if lerobot is installed, exports it. Start the MESA server first:

    (vla-benchmark) DISPLAY=:0 MUJOCO_GL=glfw uv run python scripts/raiden_sim_server.py
    (raiden)        uv run python scripts/sim_scripted_episode.py --out data_sim_smoke
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np

from raiden._config import CAMERA_CONFIG
from raiden.recorder import DemonstrationRecorder
from raiden.robot.controller import RobotController
from raiden.sim import load_sim_cameras, write_sim_files


class _ScriptedInterface:
    name = "scripted"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sim", default="127.0.0.1:5599")
    ap.add_argument(
        "--out",
        default="data_sim_smoke",
        help="data root (raw/ and processed/ are created)",
    )
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--export", action="store_true", help="also export to LeRobot")
    ap.add_argument(
        "--sim-fps", type=int, default=30, help="camera rate (30 = real D435)"
    )
    ap.add_argument("--monitor", action="store_true", help="watch the recording live")
    ap.add_argument(
        "--monitor-view",
        default="",
        help="also stream an operator view, e.g. agentview",
    )
    ap.add_argument(
        "--monitor-web", action="store_true", help="serve the monitor in a browser"
    )
    args = ap.parse_args()

    task = "sim_smoke"
    rec_dir = Path(args.out) / "raw" / task / "0000"
    if rec_dir.exists():
        shutil.rmtree(rec_dir)

    rc = RobotController(
        use_right_leader=False,
        use_left_leader=False,
        use_right_follower=False,
        use_left_follower=True,
        sim=args.sim,
    )
    rc.setup_for_teleop_recording()
    cameras = load_sim_cameras(args.sim, CAMERA_CONFIG, fps=args.sim_fps)

    # Scripted EE path in the arm base frame: lift 10 cm and reach 10 cm forward over 3 s,
    # hold, close the gripper at 3 s and open it again at 6 s.
    state: dict = {}

    def target_fn(side, T_cur, gripper_actual):
        now = time.monotonic()
        if "T0" not in state:
            state["T0"], state["t0"] = T_cur.copy(), now
        tau = now - state["t0"]
        s = min(tau / 3.0, 1.0)
        s = 0.5 - 0.5 * np.cos(np.pi * s)  # smooth 0..1
        T = state["T0"].copy()
        T[:3, 3] += np.array([0.10 * s, 0.0, 0.10 * s])
        gripper = 0.0 if 3.0 <= tau < 6.0 else 1.0
        return T, gripper

    monitor = None
    if args.monitor or args.monitor_view:
        from raiden.guides import make_guides
        from raiden.monitor import make_monitor

        monitor = make_monitor(
            [c.name for c in cameras],
            sim=args.sim,
            view=args.monitor_view,
            web=args.monitor_web,
            guides=make_guides(cameras, sim=args.sim) or None,
        )

    recorder = DemonstrationRecorder(
        cameras=cameras,
        monitor=monitor,
        robot_controller=rc,
        recording_dir=rec_dir,
        task_name=task,
        task_instruction="scripted reach and grasp in the digital twin",
        interface=_ScriptedInterface(),
        extra_metadata={"simulator": {"address": args.sim}},
    )
    rc.start_cartesian_teleop(target_fn, dt=0.01)
    recorder.start_recording()
    rc.enable_estop()
    time.sleep(args.seconds)
    saved = recorder.stop_recording(
        complete=True
    )  # also shuts the controller down (home + close)
    write_sim_files(saved, args.sim)
    for cam in cameras:
        cam.close()
    if monitor is not None:
        monitor.close()

    d = np.load(saved / "robot_data.npz")
    ts = d["timestamps"]
    print(
        f"\nrobot frames {len(ts)}  rate {1e9 * (len(ts) - 1) / (ts[-1] - ts[0]):.1f} Hz"
    )
    for cam in cameras:
        n = len(np.load(saved / "cameras" / f"{cam.name}.simrec" / "timestamps.npy"))
        print(f"{cam.name}: {n} frames (states; the converter renders the pixels)")

    from raiden.converter import convert_recording

    counts = convert_recording(
        str(saved), episode_dir=str(Path(args.out) / "processed" / task / "0000")
    )
    print("converted frames per camera:", counts)
    ep = Path(args.out) / "processed" / task / "0000"
    import pickle

    frames = sorted((ep / "lowdim").glob("*.pkl"))
    with open(frames[len(frames) // 2], "rb") as fh:
        low = pickle.load(fh)
    print(f"lowdim frame {frames[len(frames) // 2].name} keys: {sorted(low)}")
    for k in ("joints", "action_joints", "actual_poses", "action"):
        if k in low:
            print(f"  {k}: {np.round(np.asarray(low[k]), 3)}")
    for k, v in low.get("extrinsics", {}).items():
        print(f"  extrinsics[{k}] t = {np.round(np.asarray(v)[:3, 3], 3)}")
    for k, v in low.get("intrinsics", {}).items():
        print(f"  intrinsics[{k}] = {np.round(np.asarray(v).ravel()[[0, 2, 4, 5]], 1)}")
    for k, v in low.get("object_poses", {}).items():
        print(f"  object_poses[{k}] t = {np.round(np.asarray(v)[:3, 3], 3)}")
    print(
        f"  table_pose t = {np.round(low['table_pose'][:3, 3], 4)}  subtasks {low.get('subtasks')}"
    )

    if args.export:
        from raiden.lerobot_export import export_task_to_lerobot

        root = export_task_to_lerobot(
            ep.parent, [ep], Path(args.out) / "lerobot", overwrite=True
        )
        print("lerobot dataset:", root)


if __name__ == "__main__":
    main()
