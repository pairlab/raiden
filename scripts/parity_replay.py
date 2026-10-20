#!/usr/bin/env python3
"""Export held commands, replay in paused Isaac, or prepare hardware inputs offline.

This script cannot connect to hardware. Exported commands can be reviewed as
hardware replay inputs, but limits do not establish workspace/start readiness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from raiden.sim.command_replay import (
    export_episode,
    limit_report,
    load_bundle,
    measured_initial_state,
    schedule,
    sha256,
)
from raiden.sim.serialization import jsonable


def replay(args, arrays, metadata, connection=None):
    from raiden.sim.client import SimConnection

    if args.out is None:
        raise ValueError("Isaac replay requires separate --out directory")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite replay {args.out}")
    conn = SimConnection(args.sim) if connection is None else connection
    original = None
    paused = None
    experiment = None
    try:
        live = jsonable(conn.call("get_metadata"))
        if live.get("backend") != "isaac":
            raise ValueError("replay destination must identify as Isaac")
        limits = limit_report(arrays, live["joint_limits_rad"])
        if any(limits["out_of_range_counts"]) or any(
            limits["initial_arm_out_of_range"]
        ):
            raise ValueError(
                "targets or initial measured q exceed server limits; refusing silent clamps"
            )
        native = metadata["source_kind"] == "isaac_applied_tick_targets"
        portable = native or metadata["source_kind"] == "hardware_host_request_events"
        if portable:
            original_binding = metadata["source_sim_metadata"]
            keys = (
                "physics_backend",
                "physics_dt",
                "versions",
                "scene",
                "calibration_sha256",
                "implementation_sha256",
                "controller",
                "torque_limits_Nm",
                "finger_stops_m",
            )
            mismatches = [
                key for key in keys if live.get(key) != original_binding.get(key)
            ]
            if mismatches and not args.allow_scene_change:
                raise ValueError(
                    f"scene/controller differs in {mismatches}; intentional experiment needs --allow-scene-change"
                )
        elif not args.allow_unmeasured_start:
            raise ValueError(
                "real starts lack measured door/object poses; nominal-start experiment needs --allow-unmeasured-start"
            )
        indices, timing = schedule(arrays, metadata, float(live["physics_dt"]))
        args.out.mkdir(parents=True)
        experiment = {
            "bundle": str(args.bundle.resolve()),
            "timing": timing,
            "server": live,
            "status": "running",
            "runs": [],
            "limitations": metadata.get(
                "start_uncertainty", metadata["timing_uncertainty"]
            ),
        }
        paused = bool(conn.call("is_paused"))
        conn.call("set_paused", paused=True)
        original = conn.call("get_state")
        if portable:
            initial = metadata["initial_state"]
        else:
            conn.call("reset")
            initial = measured_initial_state(
                conn.call("get_state"), arrays, metadata, live
            )
        if getattr(args, "placement_overrides", None):
            override = json.loads(args.placement_overrides.read_text())
            if (
                Path(override["source_episode"]).resolve()
                != Path(metadata["source_episode"]).resolve()
            ):
                raise ValueError("placement override belongs to a different episode")
            for name, value in override["root_pose"].items():
                if name not in ("oven", "croissant"):
                    raise ValueError(f"unsupported placement override {name}")
                pose = np.asarray(value, dtype=float)
                if (
                    pose.shape != (7,)
                    or not np.isfinite(pose).all()
                    or not np.isclose(np.linalg.norm(pose[3:]), 1, atol=1e-6)
                ):
                    raise ValueError("placement must be finite world xyz+unit xyzw")
                group = "articulation" if name == "oven" else "rigid_object"
                initial["scene"][group][name]["root_pose"] = [pose.tolist()]
            experiment["placement_override"] = {
                "file": str(args.placement_overrides.resolve()),
                "sha256": sha256(args.placement_overrides),
                "content": override,
            }
        experiment["initial_state"] = jsonable(initial)
        traces, run_reports = [], experiment["runs"]
        for run in range(args.repeat):
            conn.call("reset_to", state=initial)
            trace = {
                key: []
                for key in (
                    "observation",
                    "joint_vel",
                    "joint_eff",
                    "gripper_eff",
                    "computed_effort_Nm",
                    "applied_effort_Nm",
                    "door_angle_rad",
                    "croissant_world_pose",
                )
            }
            try:
                # One RPC per physics tick preserves native holds and all response failures.
                for k, index in enumerate(indices):
                    result = conn.call(
                        "step", cmd=arrays["cmd"][index], n=1, with_state=True
                    )
                    state = result["state"]
                    obs = state["observations"]
                    trace["observation"].append(
                        np.r_[obs["joint_pos"], obs["gripper_pos"]]
                    )
                    trace["joint_vel"].append(obs["joint_vel"])
                    trace["joint_eff"].append(obs["joint_eff"])
                    trace["gripper_eff"].append(obs["gripper_eff"])
                    for name in ("computed_effort_Nm", "applied_effort_Nm"):
                        trace[name].append(state["motor"][name])
                    trace["door_angle_rad"].append(
                        state["scene"]["articulation"]["oven"]["joint_position"][0][0]
                    )
                    trace["croissant_world_pose"].append(
                        state["scene"]["rigid_object"]["croissant"]["root_pose"][0]
                    )
                    if not all(
                        np.isfinite(values[-1]).all() for values in trace.values()
                    ):
                        raise ValueError(
                            f"nonfinite physics response at tick {k}; preserving failed trace"
                        )
                    if k % 5000 == 0:
                        print(
                            f"run {run}: {k}/{len(indices)} physics ticks", flush=True
                        )
            finally:
                trace = {key: np.asarray(value) for key, value in trace.items()}
                count = len(trace["observation"])
                np.savez_compressed(
                    args.out / f"run_{run:02d}.npz",
                    tick_end_s=(np.arange(count) + 1) * live["physics_dt"],
                    command_index=indices[:count],
                    cmd=arrays["cmd"][indices[:count]],
                    **trace,
                )
            values = trace["observation"]
            traces.append(trace)
            report = {
                "run": run,
                "ticks": len(values),
                "gripper_min": float(values[:, 6].min()),
            }
            if run:
                report["max_joint_aperture_difference_from_run0"] = (
                    np.abs(values - traces[0]["observation"]).max(axis=0).tolist()
                )
            run_reports.append(report)
        experiment["status"] = "complete"
        print(
            json.dumps(
                {"output": str(args.out), "timing": timing, "runs": run_reports},
                indent=2,
            )
        )
    except BaseException as exc:
        if experiment is not None:
            experiment.update(status="failed", exception=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            try:
                if original is not None:
                    conn.call("reset_to", state=original)
            finally:
                if paused is not None:
                    conn.call("set_paused", paused=paused)
        finally:
            conn.close()
            if experiment is not None:
                (args.out / "replay.json").write_text(
                    json.dumps(experiment, indent=2, allow_nan=False) + "\n"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    export = sub.add_parser("export")
    export.add_argument("episode", type=Path)
    export.add_argument("bundle", type=Path)
    validate = sub.add_parser(
        "validate", help="offline dry-run; no robot or simulator connection"
    )
    validate.add_argument("bundle", type=Path)
    validate.add_argument("--physics-dt", type=float, default=0.002)
    validate.add_argument("--out", type=Path)
    prepare = sub.add_parser(
        "prepare-hardware",
        help="write native held sim-to-real request plan; never opens hardware",
    )
    prepare.add_argument("bundle", type=Path)
    prepare.add_argument("destination", type=Path)
    dry = sub.add_parser(
        "dry-run-hardware",
        help="prove prepared native holds reconstruct exactly; no driver imports",
    )
    dry.add_argument("plan", type=Path)
    dry.add_argument("--out", type=Path)
    measured = sub.add_parser(
        "with-hardware-start",
        help="copy native inputs with a separately bound live hardware q/dq start; never connects to hardware",
    )
    measured.add_argument("bundle", type=Path)
    measured.add_argument("snapshot", type=Path)
    measured.add_argument("destination", type=Path)
    sim = sub.add_parser("isaac", help="physics replay only; never hardware")
    sim.add_argument("bundle", type=Path)
    sim.add_argument("--sim", default="127.0.0.1:5599")
    sim.add_argument("--out", type=Path, required=True)
    sim.add_argument("--repeat", type=int, default=1)
    sim.add_argument("--allow-scene-change", action="store_true")
    sim.add_argument("--allow-unmeasured-start", action="store_true")
    sim.add_argument(
        "--placement-overrides",
        type=Path,
        help="separate episode-bound inferred world root poses",
    )
    args = parser.parse_args()
    if args.action == "with-hardware-start":
        from raiden.sim.command_replay import with_measured_hardware_start

        _, metadata = with_measured_hardware_start(
            args.bundle, args.snapshot, args.destination
        )
        print(
            json.dumps(
                {
                    "bundle": str(args.destination),
                    "initial_state_origin": metadata["initial_state_origin"],
                    "initial_state_override": metadata["initial_state_override"],
                    "native_inputs_unchanged": True,
                },
                indent=2,
            )
        )
        return
    if args.action in ("prepare-hardware", "dry-run-hardware"):
        from raiden.sim.hardware_replay import (
            dry_run_hardware_plan,
            prepare_hardware_plan,
        )

        if args.action == "prepare-hardware":
            _, metadata = prepare_hardware_plan(args.bundle, args.destination)
            report = {
                "plan": str(args.destination),
                "request_events": metadata["request_events"],
                "software_limit_check_passed": metadata["software_limit_check_passed"],
                "hardware_executed": False,
                "physical_readiness": "not checked",
            }
        else:
            report = dry_run_hardware_plan(args.plan)
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return
    if args.action == "export":
        arrays, metadata = export_episode(args.episode, args.bundle)
        print(
            json.dumps(
                {
                    "bundle": str(args.bundle),
                    "source_kind": metadata["source_kind"],
                    "limits": limit_report(arrays),
                },
                indent=2,
            )
        )
    else:
        arrays, metadata = load_bundle(args.bundle)
        if args.action == "validate":
            _, timing = schedule(arrays, metadata, args.physics_dt)
            report = {
                "limits": limit_report(arrays),
                "timing": timing,
                "source_kind": metadata["source_kind"],
                "timing_uncertainty": metadata["timing_uncertainty"],
            }
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
        else:
            if args.repeat < 1:
                parser.error("repeat must be positive")
            replay(args, arrays, metadata)


if __name__ == "__main__":
    main()
