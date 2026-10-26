#!/usr/bin/env python3
"""Replay Isaac commands, or restore recorded states to rerender an Isaac episode.

Only talks to the Isaac simulator RPC server; this does not command hardware.
Command replay advances dynamics from the log's initial state with every native
physics-tick target. State replay restores snapshots for visual reconstruction.

    python scripts/isaac_replay.py data/isaac/raw/task/0000 --mode commands --repeat 2
    python scripts/isaac_replay.py data/isaac/raw/task/0000 --mode states --rerender
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from raiden.sim.client import DEFAULT_ADDRESS, SimConnection
from raiden.sim.serialization import jsonable, write_json, write_jsonl


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _scene_arrays(state: dict) -> dict[str, np.ndarray]:
    """Compare all portable scene numeric fields, keeping names in the result."""
    arrays = {}

    def walk(value, name):
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{name}/{key}" if name else key)
        elif isinstance(value, (list, tuple, float, int, np.ndarray, np.number)):
            array = np.asarray(value)
            if np.issubdtype(array.dtype, np.number):
                arrays[name] = array.astype(np.float64)

    walk(state["scene"], "scene")
    return arrays


def validate_server(conn, episode: Path, *, allow_scene_change=False) -> dict:
    metadata = json.loads((episode / "sim_metadata.json").read_text())
    live = jsonable(conn.call("get_metadata"))
    if metadata.get("backend") != "isaac" or live.get("backend") != "isaac":
        raise ValueError("Isaac replay requires an Isaac episode and Isaac server")
    identifiers = (
        "state_version", "physics_backend", "physics_dt", "versions", "arm_joints", "arms", "action_dim", "asset_input_sha256",
        "twin_input_sha256", "scene", "calibration_sha256", "implementation_sha256",
        "controller", "torque_limits_Nm", "finger_stops_m", "finger_target",
        "joint_limits_rad", "joint_limit_margin_rad", "gravity_implementation",
    )
    mismatches = []
    for key in identifiers:
        if key not in metadata:
            continue
        recorded, current = metadata[key], live.get(key)
        if key == "scene":
            # Each replay restores the episode's state instead of the live reset realization.
            reset_fields = {"initial_state", "physical_initial_state"}
            recorded = {k: v for k, v in recorded.items() if k not in reset_fields}
            if isinstance(current, dict):
                current = {k: v for k, v in current.items() if k not in reset_fields}
        if recorded != current:
            mismatches.append(key)
    if mismatches and not allow_scene_change:
        raise ValueError(f"server differs from episode in {mismatches}; use the recorded scene, or "
                         "--allow-scene-change for an intentional calibration/physics experiment")
    return live


def rerender(conn, episode: Path, *, output: Path | None = None, max_frames: int | None = None,
             allow_scene_change=False):
    """Populate raw simrec RGB from its own exact state snapshots and optical poses."""
    cameras = sorted((episode / "cameras").glob("*.simrec"))
    if not cameras:
        raise ValueError("episode has no .simrec cameras")
    counts = {}
    for source in cameras:
        rows = load_rows(source / "sim_state.jsonl")
        if max_frames is not None:
            rows = rows[:max_frames]
        target = source if output is None else output / source.name
        rgb_dir = target / "rgb"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        poses = []
        for index, row in enumerate(rows):
            conn.call("reset_to", state=row["state"])
            frame = conn.call("render", cameras=[source.stem], depth=False,
                              with_state=True, refresh=True)[source.stem]
            bgr = np.ascontiguousarray(np.asarray(frame["rgb"])[:, :, ::-1])
            pose = np.asarray(frame["T_base_cam"], dtype=np.float64)
            recorded_pose = np.asarray(row["T_base_cam"], dtype=np.float64)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError(f"{source.name}: rerender has invalid optical pose")
            if not allow_scene_change and not np.allclose(pose, recorded_pose, atol=2e-6, rtol=0):
                raise ValueError(f"{source.name}: restored camera pose differs from the recorded optical pose")
            if not cv2.imwrite(str(rgb_dir / f"{index:010d}.png"), bgr):
                raise OSError(f"could not save {target.name} frame {index}")
            poses.append(pose if allow_scene_change else recorded_pose)
            if allow_scene_change:
                row["T_base_cam"] = pose.tolist()
        np.save(target / "camera_poses.npy", np.asarray(poses).reshape(-1, 4, 4))
        np.save(target / "timestamps.npy", np.asarray([row["t_ns"] for row in rows], dtype=np.int64))
        if output is not None:
            info = json.loads((source / "camera_info.json").read_text())
            if allow_scene_change:
                live_info = jsonable(conn.call("get_camera_info", name=source.stem))
                K = np.asarray(live_info["K"])
                info.update(live_info, fx=float(K[0, 0]), fy=float(K[1, 1]),
                            cx=float(K[0, 2]), cy=float(K[1, 2]), recording_mode="frames")
                distortion = list(live_info.get("distortion_coeffs", [0.] * 5)) + [0.] * 5
                info.update(dict(zip(("k1", "k2", "p1", "p2", "k3"), distortion)))
            write_json(target / "camera_info.json", info)
            write_jsonl(target / "sim_state.jsonl", rows)
        counts[source.stem] = len(rows)
        print(f"{source.stem}: {len(rows)} frames -> {rgb_dir}")
    return counts


def replay_commands(conn, episode: Path, *, repeat: int = 1, output: Path | None = None):
    """Reapply every native tick; never interpolate the training-rate robot export."""
    initial = json.loads((episode / "sim_initial_state.json").read_text())
    with np.load(episode / "sim_action_log.npz", allow_pickle=False) as archive:
        commands = np.asarray(archive["cmd"], dtype=np.float64)
        times = np.asarray(archive["sim_time"], dtype=np.float64)
    dt = float(initial["physics_dt"])
    live_dt = float(conn.call("get_metadata")["physics_dt"])
    if not np.isclose(live_dt, dt, atol=1e-12, rtol=0):
        raise ValueError("command replay requires the recorded physics_dt to preserve native tick holds")
    if len(times):
        expected = float(initial["sim_time"]) + dt * np.arange(1, len(times) + 1)
        if not np.allclose(times, expected, atol=max(1e-8, dt * 1e-4), rtol=0):
            raise ValueError("tick times are not contiguous from the saved initial state; "
                             "refusing to change command hold timing")
    recorded = {}
    state_file = episode / "sim_action_states.jsonl"
    if state_file.exists():
        recorded = {round(float(row["state"]["sim_time"]) / dt):
                    _scene_arrays(row["state"]) for row in load_rows(state_file)}
    reports = []
    first = None
    for run in range(repeat):
        conn.call("reset_to", state=initial)
        trace = []
        for command in commands:
            result = conn.call("step", cmd=command, n=1, with_state=True)
            trace.append(result["state"])
        arrays = [_scene_arrays(state) for state in trace]
        if arrays and not all(arrays):
            raise ValueError("replay state contains no numeric scene fields; refusing an empty fidelity report")
        digest = hashlib.sha256()
        for state in arrays:
            for key, value in sorted(state.items()):
                digest.update(key.encode())
                digest.update(value.tobytes())
        report = {"run": run, "ticks": len(commands), "scene_sha256": digest.hexdigest()}
        errors = {}
        matches = 0
        for state, tick_time in zip(arrays, times):
            reference = recorded.get(round(float(tick_time) / dt))
            if reference is None:
                continue
            matches += 1
            for key, value in state.items():
                if value.size and key in reference:
                    errors[key] = max(errors.get(key, 0.0), float(np.max(np.abs(value - reference[key]))))
        if matches:
            report.update(recorded_state_samples_compared=matches, max_abs_scene_field_error=errors)
        if first is not None:
            differences = [float(np.max(np.abs(value - reference[key])))
                           for state, reference in zip(arrays, first)
                           for key, value in state.items() if value.size]
            report["max_scene_difference_from_run_0"] = max(differences, default=0.0)
        else:
            first = arrays
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            write_jsonl(output / f"run_{run:02d}.jsonl", trace)
        reports.append(report)
        print(json.dumps(report))
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--sim", default=DEFAULT_ADDRESS)
    parser.add_argument("--mode", choices=("commands", "states"), default="commands")
    parser.add_argument("--rerender", action="store_true", help="materialize Isaac RGB (requires --mode states)")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--out", type=Path, help="separate rendered .simrec directory or replay traces")
    parser.add_argument("--max-frames", type=int, help="limit state renders (requires a separate --out)")
    parser.add_argument("--allow-scene-change", action="store_true", help="permit an intentional scene or calibration change")
    args = parser.parse_args()
    if args.repeat < 1 or (args.rerender and args.mode != "states"):
        parser.error("repeat must be positive; --rerender requires --mode states")
    if args.mode == "states" and not args.rerender:
        parser.error("state restoration is for --rerender; use --mode commands to replay physics")
    if args.mode == "states" and args.allow_scene_change and args.out is None:
        parser.error("an intentional scene change requires separate --out to retain the original optical poses")
    if args.max_frames is not None and (args.out is None or args.max_frames < 1):
        parser.error("--max-frames requires --out and a positive count")
    conn = SimConnection(args.sim)
    original = None
    try:
        live = validate_server(conn, args.episode, allow_scene_change=args.allow_scene_change)
        conn.call("set_paused", paused=True)
        original = conn.call("get_state")
        if args.mode == "states":
            report = rerender(conn, args.episode, output=args.out, max_frames=args.max_frames,
                              allow_scene_change=args.allow_scene_change)
        else:
            report = replay_commands(conn, args.episode, repeat=args.repeat, output=args.out)
        if args.out is not None:
            write_json(args.out / "replay_metadata.json", {"source_episode": str(args.episode.resolve()),
                       "mode": args.mode, "server": live, "result": report})
        elif args.mode == "states":
            write_json(args.episode / "sim_rerender_metadata.json", {"server": live, "result": report})
    finally:
        try:
            if original is not None:
                conn.call("reset_to", state=original)
        finally:
            conn.close()



if __name__ == "__main__":
    main()
