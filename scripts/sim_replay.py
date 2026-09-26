#!/usr/bin/env python3
"""Replay a sim episode in the twin, tick for tick, and compare it to the recording.

``cameras/<camera>.simrec/sim_state.npz`` holds the MuJoCo state and solver warmstart behind
every camera frame, and ``sim_action_log.npz`` the command the server latched on every physics
tick. The replay restores frame 0 and steps the paused server through the logged commands, so
it reproduces the episode and two replays are bit-identical.

    (vla-benchmark) MUJOCO_GL=egl uv run python scripts/raiden_sim_server.py --task-json <same task>
    (raiden)        uv run python scripts/sim_replay.py data/raw/<task>/0000 --repeat 2
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from raiden.sim import SimConnection
from raiden.sim.action_log import FILENAME as ACTION_LOG

ARM_TOL_RAD = 0.05
OBJ_TOL_M = 0.005


def load_episode(ep_dir: Path, camera: str) -> dict:
    """Recorded camera-frame states and success flags, and the per-tick commands."""
    simrec = ep_dir / "cameras" / f"{camera}.simrec" / "sim_state.npz"
    if not simrec.exists():
        sys.exit(f"{simrec} not found: is {ep_dir} a sim episode?")
    if not (ep_dir / ACTION_LOG).exists():
        sys.exit(f"{ep_dir} has no {ACTION_LOG}")
    sim = np.load(simrec)
    log = np.load(ep_dir / ACTION_LOG)
    return {
        "frame_t": sim["t_ns"] / 1e9,
        "state": sim["state"].astype(np.float64),
        "warmstart": sim["warmstart"].astype(np.float64),
        "subtasks": [str(s) for s in sim["subtasks"]],
        "success": sim["success"],
        "tick_time": log["sim_time"],
        "tick_cmd": log["cmd"],
    }


def qpos_split(model_xml: Path) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Robot qpos addresses, and ``(name, address)`` per free object, for this episode's model."""
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    arm, objects = [], []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"
        adr = int(model.jnt_qposadr[j])
        if name.startswith(("robot0", "gripper0")):
            arm.append(adr)
        elif model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            objects.append((name.removesuffix("_joint0"), adr))
    return np.array(arm, dtype=int), objects


def schedule(ep: dict) -> tuple[np.ndarray, np.ndarray]:
    """Commands from frame 0 on, and the tick count at which each camera frame was captured.

    Frames and ticks share the MuJoCo clock: every state carries the sim time it was captured
    at, every logged tick the sim time it ended at.
    """
    sim_t = ep["tick_time"]
    k0 = int(np.searchsorted(sim_t, ep["state"][0, 0], side="right"))
    frames = np.searchsorted(sim_t, ep["state"][:, 0], side="right") - k0
    return ep["tick_cmd"][k0:], frames


def replay(conn: SimConnection, ep: dict, cmds: np.ndarray, frames: np.ndarray) -> dict:
    """Restore frame 0, step every command, sample the state at every camera frame."""
    conn.call("set_paused", paused=True)
    conn.call("reset_to", state=ep["state"][0], warmstart=ep["warmstart"][0])

    sample = {int(t): j for j, t in enumerate(frames)}
    states = np.zeros((len(frames), ep["state"].shape[1]))
    states[0] = np.asarray(conn.call("get_state"))
    success = np.zeros(len(cmds) + 1, dtype=bool)
    done = np.zeros((len(cmds) + 1, len(ep["subtasks"])), dtype=bool)

    t0 = time.perf_counter()
    for t, cmd in enumerate(cmds, start=1):
        j = sample.get(t)
        out = conn.call("step", cmd=cmd, n=1, with_state=j is not None)
        success[t] = out["success"]
        done[t] = out["subtasks_done"]
        if j is not None:
            states[j] = out["state"]
    return {
        "states": states,
        "success": success,
        "done": done,
        "wall_s": time.perf_counter() - t0,
    }


def digest(a: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(a, dtype=np.float64).tobytes()
    ).hexdigest()[:16]


def first_true(mask: np.ndarray) -> int:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if len(idx) else -1


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("episode", help="sim episode dir, e.g. data/raw/<task>/0000")
    ap.add_argument("--sim", default="127.0.0.1:5599")
    ap.add_argument(
        "--camera", default="scene_camera", help="which .simrec holds the states"
    )
    ap.add_argument(
        "--repeat", type=int, default=2, help="replays to run (>=2 tests determinism)"
    )
    ap.add_argument("--save", default="", help="npz path for the replayed traces")
    args = ap.parse_args()

    ep_dir = Path(args.episode)
    ep = load_episode(ep_dir, args.camera)
    arm_adr, objects = qpos_split(ep_dir / "sim_model.xml")
    conn = SimConnection(args.sim)

    live = np.asarray(conn.call("get_state"))
    if live.shape != ep["state"][0].shape:
        sys.exit(
            f"state size mismatch: episode {ep['state'].shape[1]}, server {live.shape[0]}.\n"
            "Start raiden_sim_server.py with the task json this episode was recorded from."
        )

    cmds, frames = schedule(ep)
    rec = ep["state"]
    host_s = ep["frame_t"][-1] - ep["frame_t"][0]
    sim_s = rec[-1, 0] - rec[0, 0]
    print(f"episode   : {ep_dir}")
    print(
        f"recorded  : {len(rec)} frames, {len(cmds)} ticks, "
        f"{host_s:.2f} s host, {sim_s:.2f} s sim"
    )
    print(
        f"demo      : success={bool(ep['success'][-1])} "
        f"(first at frame {first_true(ep['success'])}/{len(ep['success'])})"
    )
    print(f"subtasks  : {', '.join(ep['subtasks'])}\n")

    runs = []
    for r in range(args.repeat):
        out = replay(conn, ep, cmds, frames)
        runs.append(out)
        hit = first_true(out["success"])
        print(
            f"run {r}: success={bool(out['success'][-1])} "
            f"(first at tick {hit if hit >= 0 else '-'}/{len(cmds)})  "
            f"subtasks {out['done'][-1].astype(int).tolist()}  "
            f"sha {digest(out['states'])}  {out['wall_s']:.1f} s"
        )

    print("\ndeterminism (run 0 vs run k):")
    for r, out in enumerate(runs[1:], start=1):
        d = np.abs(out["states"] - runs[0]["states"])
        same = bool((d == 0).all())
        print(
            f"  run {r}: {'BIT-IDENTICAL' if same else 'DIVERGES'}   "
            f"max|Δstate| {d.max():.3e}   "
            f"first differing frame {'-' if same else first_true((d > 0).any(axis=1))}"
        )

    rep = runs[0]["states"]
    arm_err = np.abs(rep[:, 1 + arm_adr] - rec[:, 1 + arm_adr]).max(axis=1)
    obj_err = np.zeros(len(frames))
    print(f"\nfidelity vs the recording (run 0, {len(frames)} frames):")
    print(
        f"  arm qpos      : max {arm_err.max():.4f} rad, final {arm_err[-1]:.4f} rad, "
        f"first frame over {ARM_TOL_RAD} rad: {first_true(arm_err > ARM_TOL_RAD)}"
    )
    for name, adr in objects:
        e = np.linalg.norm(
            rep[:, 1 + adr : 4 + adr] - rec[:, 1 + adr : 4 + adr], axis=1
        )
        obj_err = np.maximum(obj_err, e)
        print(
            f"  {name:14s}: max {e.max() * 1e3:6.1f} mm, final {e[-1] * 1e3:6.1f} mm, "
            f"first frame over {OBJ_TOL_M * 1e3:.0f} mm: {first_true(e > OBJ_TOL_M)}"
        )
    if max(arm_err.max(), obj_err.max()) < 1e-9:
        verdict = "EXACT"
    elif arm_err.max() < ARM_TOL_RAD and obj_err.max() < OBJ_TOL_M:
        verdict = "close"
    else:
        verdict = "DIVERGES"
    print(
        f"  verdict       : {verdict}  (replay success {bool(runs[0]['success'][-1])}, "
        f"demo success {bool(ep['success'][-1])})"
    )

    if args.save:
        np.savez(
            args.save,
            frames=frames,
            recorded=rec,
            arm_err=arm_err,
            obj_err=obj_err,
            **{f"run{r}": out["states"] for r, out in enumerate(runs)},
        )
        print(f"\nsaved traces → {args.save}")

    conn.call("set_paused", paused=False)
    conn.close()


if __name__ == "__main__":
    main()
