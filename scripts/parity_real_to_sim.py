#!/usr/bin/env python3
"""Roll out sampled real held commands in paused Isaac and record native dynamics.

No hardware APIs. Only measured arm/aperture q/dq initialize the rollout. Object
and oven starts use the selected nominal scene. Rendering never advances physics.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from raiden.sim.client import SimConnection
from raiden.sim.command_replay import (
    limit_report,
    load_bundle,
    measured_initial_state,
    schedule,
)
from raiden.sim.serialization import jsonable


def encode(path, width, height, fps):
    return subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-n",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )


def run(args):
    arrays, source = load_bundle(args.bundle)
    if source["source_kind"] != "real_observation_clock_command_snapshots":
        raise ValueError(
            "requires sampled real commands; use native replay for sim logs"
        )
    if not args.allow_unmeasured_start:
        raise ValueError("nominal object/oven starts require --allow-unmeasured-start")
    if args.out.exists():
        raise FileExistsError(args.out)
    conn = SimConnection(args.sim)
    original = paused = None
    encoders = {}
    trace = {}
    frame_rows = []
    report = None
    try:
        live = jsonable(conn.call("get_metadata"))
        if (
            live["backend"] != "isaac"
            or live["physics_backend"] != "newton/mujoco-warp"
        ):
            raise ValueError("requires accepted Isaac Newton/MuJoCo-Warp backend")
        calibration = json.loads(args.calibration.read_text())
        if live["calibration"] != calibration:
            raise ValueError(
                "server calibration differs from explicitly requested calibration"
            )
        limits = limit_report(arrays, live["joint_limits_rad"])
        if any(limits["out_of_range_counts"]) or any(
            limits["initial_arm_out_of_range"]
        ):
            raise ValueError("source exceeds joint limits; refusing silent clamps")
        indices, timing = schedule(arrays, source, float(live["physics_dt"]))
        stride = round(1 / (args.fps * float(live["physics_dt"])))
        if stride < 1 or not np.isclose(stride * args.fps * live["physics_dt"], 1):
            raise ValueError("video fps must divide physics tick rate exactly")
        args.out.mkdir(parents=True)
        paused = bool(conn.call("is_paused"))
        conn.call("set_paused", paused=True)
        original = conn.call("get_state")
        conn.call("reset")
        initial = measured_initial_state(conn.call("get_state"), arrays, source, live)
        expected = json.loads(args.initial_states.read_text())["states"][args.state]
        for name, group in [("croissant", "rigid_object"), ("oven", "articulation")]:
            actual = np.asarray(initial["scene"][group][name]["root_pose"][0])
            nominal = np.asarray(expected[name]["twin_pos"])
            if not np.allclose(actual[:3], nominal, atol=3e-6, rtol=0):
                raise ValueError(
                    f"server reset {name} {actual[:3]} differs from requested {args.state} {nominal}"
                )
        placement_override = None
        if args.placement_overrides:
            placement_override = json.loads(args.placement_overrides.read_text())
            if (
                Path(placement_override["source_episode"]).resolve()
                != Path(source["source_episode"]).resolve()
            ):
                raise ValueError(
                    "placement sidecar belongs to a different source episode"
                )
            for name, pose in placement_override["root_pose"].items():
                if name not in ("oven", "croissant"):
                    raise ValueError("unsupported placement override")
                pose = np.asarray(pose, dtype=float)
                if (
                    pose.shape != (7,)
                    or not np.isfinite(pose).all()
                    or not np.isclose(np.linalg.norm(pose[3:]), 1, atol=1e-6)
                ):
                    raise ValueError("placement must be finite world xyz+unit xyzw")
                group = "articulation" if name == "oven" else "rigid_object"
                initial["scene"][group][name]["root_pose"] = [pose.tolist()]
        conn.call("reset_to", state=initial)
        report = {
            "schema": "raiden.real_to_sim_rollout.v1",
            "status": "running",
            "source": source,
            "bundle": str(args.bundle.resolve()),
            "server": live,
            "timing": timing,
            "nominal_start": args.state,
            "initial_state": jsonable(initial),
            "placement_override": placement_override,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "calibration_file": str(args.calibration.resolve()),
            "limitations": [
                source["timing_uncertainty"],
                source["start_uncertainty"],
                "One session placement measurement does not measure each episode initial object/oven pose.",
                "Solver warmstart/controller histories are not source hardware histories.",
                "Rigid simulated croissant approximates deformable real croissant.",
                "RPC success is a placeholder; use saved task geometry and video for outcome.",
            ],
            "video": {
                "fps": args.fps,
                "physics_tick_stride": stride,
                "timing": "frames at exact completed native ticks; no time stretch or lag fit",
            },
        }
        (args.out / "initial_state.json").write_text(
            json.dumps(jsonable(initial), indent=2) + "\n"
        )
        wall_start = time.monotonic()
        sim_start = float(initial["sim_time"])
        snapshots = gzip.open(args.out / "frame_states.jsonl.gz", "wt")
        frame_index = 0

        def capture(state, tick):
            nonlocal frame_index
            t = float(state["sim_time"]) - sim_start
            frames = conn.call(
                "render",
                cameras=["scene_camera", "left_wrist_camera"],
                depth=False,
                with_state=False,
                refresh=True,
            )
            item = {
                "frame": frame_index,
                "tick": tick,
                "elapsed_s": t,
                "state": jsonable(state),
                "cameras": {},
            }
            for name, f in frames.items():
                if not np.isclose(
                    float(f["sim_time"]), float(state["sim_time"]), atol=1e-8, rtol=0
                ):
                    raise ValueError(
                        "render advanced physics or state/pixel boundary differs"
                    )
                bgr = np.ascontiguousarray(np.asarray(f["rgb"])[:, :, ::-1])
                h, w = bgr.shape[:2]
                if name not in encoders:
                    encoders[name] = encode(args.out / f"{name}.mp4", w, h, args.fps)
                encoders[name].stdin.write(bgr.tobytes())
                item["cameras"][name] = {
                    "T_base_cam": jsonable(f["T_base_cam"]),
                    "shape": list(bgr.shape),
                }
                if frame_index % 20 == 0:
                    cv2.imwrite(str(args.out / f"{name}_t{t:06.2f}.jpg"), bgr)
            snapshots.write(json.dumps(item, separators=(",", ":")) + "\n")
            frame_rows.append((frame_index, tick, t))
            frame_index += 1

        try:
            capture(conn.call("get_state"), 0)
            for k, index in enumerate(indices):
                state = conn.call(
                    "step", cmd=arrays["cmd"][index], n=1, with_state=True
                )["state"]
                obs = state["observations"]
                oven = state["scene"]["articulation"]["oven"]
                obj = state["scene"]["rigid_object"]["croissant"]
                row = {
                    "q": np.r_[obs["joint_pos"], obs["gripper_pos"]],
                    "dq": np.r_[obs["joint_vel"], obs["gripper_vel"]],
                    "joint_eff": obs["joint_eff"],
                    "gripper_eff": obs["gripper_eff"],
                    "computed_effort_Nm": state["motor"]["computed_effort_Nm"],
                    "applied_effort_Nm": state["motor"]["applied_effort_Nm"],
                    "door_angle_rad": oven["joint_position"][0][0],
                    "oven_world_pose": oven["root_pose"][0],
                    "croissant_world_pose": obj["root_pose"][0],
                    "croissant_world_velocity": obj["root_velocity"][0],
                }
                for name, value in row.items():
                    value = np.asarray(value)
                    if not np.isfinite(value).all():
                        raise ValueError(f"nonfinite {name} at tick{k}")
                    trace.setdefault(name, []).append(value.copy())
                if (k + 1) % stride == 0:
                    capture(state, k + 1)
                if (k + 1) % 1000 == 0:
                    print(
                        f"{Path(source['source_episode']).name}: {k + 1}/{len(indices)} native ticks; {time.monotonic() - wall_start:.1f}s wall",
                        flush=True,
                    )
            if len(indices) % stride:
                capture(state, len(indices))
            report.update(
                status="complete",
                wall_s=time.monotonic() - wall_start,
                completed_ticks=len(indices),
                frame_count=frame_index,
                final_state=jsonable(state),
            )
        finally:
            snapshots.close()
            count = len(trace.get("q", []))
            np.savez_compressed(
                args.out / "trace.npz",
                tick_end_s=(np.arange(count) + 1) * live["physics_dt"],
                command_index=indices[:count],
                cmd=arrays["cmd"][indices[:count]],
                **{name: np.asarray(value) for name, value in trace.items()},
            )
            np.savez_compressed(
                args.out / "video_timing.npz", frame=np.asarray(frame_rows)
            )
    except BaseException as exc:
        if report is not None:
            report.update(status="failed", exception=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for encoder in encoders.values():
            encoder.stdin.close()
            if encoder.wait() != 0:
                if report is not None:
                    report["video_encoder_failed"] = True
        try:
            if original is not None:
                conn.call("reset_to", state=original)
            if paused is not None:
                conn.call("set_paused", paused=paused)
            if report is not None:
                report["original_server_state_restored"] = True
        finally:
            conn.close()
            if report is not None:
                (args.out / "rollout.json").write_text(
                    json.dumps(report, indent=2, allow_nan=False) + "\n"
                )
    print(
        json.dumps(
            {
                k: report[k]
                for k in ["status", "completed_ticks", "frame_count", "wall_s"]
            },
            indent=2,
        ),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--sim", default="127.0.0.1:5599")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--state", required=True)
    p.add_argument(
        "--initial-states", type=Path, default=Path("data/parity/initial_states.json")
    )
    p.add_argument(
        "--placement-overrides",
        type=Path,
        help="separate episode-bound placement sensitivity sidecar",
    )
    p.add_argument("--calibration", required=True, type=Path)
    p.add_argument("--allow-unmeasured-start", action="store_true")
    args = p.parse_args()
    if args.fps < 1:
        p.error("fps must be positive")
    run(args)


if __name__ == "__main__":
    main()
