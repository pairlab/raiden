#!/usr/bin/env python3
"""Replay a real episode's 100 Hz joint commands into the digital twin and compare dynamics.

Sends ``follower_l_joint_cmd`` from ``robot_data.npz`` to the sim follower on the recorded
timestamps, records the sim's measured joints and prints, per joint, the real vs sim
tracking error and command->measurement lag. The physics thread in the server runs at 100 Hz
in real time, so this takes as long as the episode.

    (vla-benchmark) DISPLAY=:0 MUJOCO_GL=glfw uv run python scripts/raiden_sim_server.py
    (raiden)        uv run python scripts/sim_replay_compare.py data/raw/cube_stack/0010
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from raiden.robot.controller import smooth_move_joints
from raiden.sim import SimFollower


def lag_table(cmd: np.ndarray, meas: np.ndarray, dt: float, max_k: int = 40):
    rows = []
    for j in range(cmd.shape[1]):
        best = None
        for k in range(max_k + 1):
            e = cmd[: len(cmd) - k, j] - meas[k:, j]
            e = e - e.mean()
            r = float(np.sqrt(np.mean(e**2)))
            if best is None or r < best[1]:
                best = (k, r)
        e0 = cmd[:, j] - meas[:, j]
        rows.append(
            (
                j,
                float(np.sqrt(np.mean(e0**2))),
                float(np.abs(e0).max()),
                best[0] * dt * 1e3,
                best[1],
            )
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("episode", help="raw episode dir containing robot_data.npz")
    ap.add_argument("--sim", default="127.0.0.1:5599")
    ap.add_argument(
        "--start",
        type=int,
        default=100,
        help="skip the first N frames (home-move transient)",
    )
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole episode")
    ap.add_argument("--save", default="", help="npz path for the sim trace")
    ap.add_argument(
        "--contacts", action="store_true", help="log robot contacts every 10 frames"
    )
    args = ap.parse_args()

    d = np.load(Path(args.episode) / "robot_data.npz")
    ts = d["timestamps"].astype(np.float64) / 1e9
    cmd = d["follower_l_joint_cmd"].astype(np.float64)
    real = d["follower_l_joint_pos_7d"].astype(np.float64)
    sl = slice(args.start, args.start + args.max_frames if args.max_frames else None)
    ts, cmd, real = ts[sl], cmd[sl], real[sl]
    dt = float(np.median(np.diff(ts)))
    n = len(ts)
    print(f"{n} frames, {ts[-1] - ts[0]:.1f} s, median dt {dt * 1e3:.2f} ms")

    f = SimFollower(args.sim)
    smooth_move_joints(f, cmd[0], time_interval_s=3.0, steps=300)
    time.sleep(0.5)

    sim = np.zeros_like(real)
    t0 = time.perf_counter() - (ts[0] - ts[0])
    for i in range(n):
        target = t0 + (ts[i] - ts[0])
        while True:
            now = time.perf_counter()
            if now >= target:
                break
            time.sleep(min(0.002, target - now))
        f.command_joint_pos(cmd[i])
        sim[i] = f.get_joint_pos()
        if args.contacts and i % 10 == 0:
            cs = f._c.call("get_contacts")
            if cs:
                print(
                    f"  t={ts[i] - ts[0]:6.2f}s grip cmd {cmd[i, 6]:.2f} meas {sim[i, 6]:.2f} j5 {sim[i, 4]:+.2f}: "
                    + "; ".join(f"{a}|{b} d={d_:.3f} F={F:.1f}" for a, b, d_, F in cs)
                )
        if i % 500 == 0:
            print(f"  {i}/{n}", end="\r", flush=True)
    print()
    smooth_move_joints(
        f, np.array([0, 0, 0, 0, 0, 0, 1.0]), time_interval_s=3.0, steps=300
    )
    f.close()

    if args.save:
        np.savez(args.save, timestamps=ts, cmd=cmd, real=real, sim=sim)

    names = ["j1", "j2", "j3", "j4", "j5", "j6", "grip"]
    print(
        "\nper joint: RMS(cmd - meas) at k=0 | max | lag(ms) | RMS at lag        [real  vs  sim]"
    )
    for (j, r0, m0, lag0, rl0), (_, r1, m1, lag1, rl1) in zip(
        lag_table(cmd, real, dt), lag_table(cmd, sim, dt)
    ):
        print(
            f"{names[j]:>4}: {r0:.4f} {m0:.3f} {lag0:5.0f} {rl0:.4f}   |   {r1:.4f} {m1:.3f} {lag1:5.0f} {rl1:.4f}"
        )
    dev = sim - real
    print(
        "\nsim vs real measured: RMS",
        np.round(np.sqrt(np.mean(dev**2, axis=0)), 4),
        "max",
        np.round(np.abs(dev).max(axis=0), 3),
    )
    v_real = np.abs(np.diff(real, axis=0) / dt).max(axis=0)
    v_sim = np.abs(np.diff(sim, axis=0) / dt).max(axis=0)
    print("max |vel| real", np.round(v_real, 2), "sim", np.round(v_sim, 2))


if __name__ == "__main__":
    main()
