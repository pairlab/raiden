"""Prepare native held inputs for hardware, without importing a hardware driver.

The CLI only prepares and dry-runs plans. Execution accepts an already initialized,
attended follower: constructing the current YAM driver calibrates/moves its jaws.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .command_replay import MIN_APERTURE, limit_report, load_bundle, sha256

PLAN_SCHEMA = "raiden.hardware_held_request_plan.v1"
TIMING_PROFILES = {
    "strict": 2_000_000,
    "best_effort": 50_000_000,
    "best_effort_unbounded": None,
}


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _source_bindings(metadata):
    episode = Path(metadata["source_episode"])
    result = {}
    for name, expected in metadata["source_sha256"].items():
        source = episode / name
        actual = sha256(source)
        if actual != expected:
            raise ValueError(f"source content changed: {source}")
        result[name] = actual
    return result


def _expected_controller(source):
    initial = source["initial_state"]
    return {
        "kp": initial["kp"],
        "kd": initial["kd"],
        "gravity_comp_factor": source["source_sim_metadata"]["controller"][
            "gravity_comp_factor"
        ],
        "limit_gripper_effort": 50.0,
    }


def _validate_plan_arrays(arrays, metadata):
    if metadata.get("schema") != PLAN_SCHEMA:
        raise ValueError("unsupported hardware plan schema")
    offsets = np.asarray(arrays["offset_ns"])
    commands = np.asarray(arrays["cmd"])
    if offsets.dtype.kind not in "iu" or offsets.ndim != 1 or not len(offsets):
        raise ValueError("plan needs integer native event offsets")
    if offsets[0] != 0 or np.any(offsets[1:] <= offsets[:-1]):
        raise ValueError("plan event offsets must increase from zero")
    if commands.shape != (len(offsets), 7) or not np.isfinite(commands).all():
        raise ValueError("plan commands must be finite N,7")
    if np.any((commands[:, 6] < MIN_APERTURE) | (commands[:, 6] > 1)):
        raise ValueError("hardware requests must already preserve the 20/95 floor")
    if int(metadata["duration_ns"]) <= int(offsets[-1]):
        raise ValueError("last native target needs a positive recorded hold")
    for key in ("initial_q", "initial_dq"):
        if np.asarray(arrays[key]).shape != (7,) or not np.isfinite(arrays[key]).all():
            raise ValueError(f"invalid {key}")


def prepare_hardware_plan(bundle: Path, destination: Path):
    """Compact only identical adjacent applied targets; never retime or interpolate.

    These values become *requests* at the hardware follower boundary. Its normal
    stateful gripper limiter and onboard PD remain active; applied torque is not
    prescribed by this plan.
    """
    bundle, destination = Path(bundle).resolve(), Path(destination)
    arrays, source = load_bundle(bundle)
    if source["source_kind"] != "isaac_applied_tick_targets":
        raise ValueError(
            "sim-to-real preparation requires native Isaac applied-tick inputs"
        )
    hashes = _source_bindings(source)
    cmd = arrays["cmd"]
    changed = np.r_[True, np.any(cmd[1:] != cmd[:-1], axis=1)]
    index = np.flatnonzero(changed)
    prepared = {
        "offset_ns": arrays["offset_ns"][index],
        "cmd": cmd[index].copy(),
        "source_requested_cmd": arrays["requested_cmd"][index].copy(),
        "source_tick_index": index,
        "initial_q": arrays["initial_q"].copy(),
        "initial_dq": arrays["initial_dq"].copy(),
    }
    episode = Path(source["source_episode"])
    with np.load(episode / "robot_data.npz", allow_pickle=False) as raw:
        # Preserve this independently sampled channel on its own observation clock.
        prepared["sampled_operator_intent_offset_ns"] = (
            raw["timestamps"] - raw["timestamps"][0]
        )
        prepared["sampled_operator_intent"] = raw["follower_l_joint_cmd"][:, 6].copy()
    dt = np.diff(prepared["offset_ns"])
    initial = source["initial_state"]
    plan = {
        "schema": PLAN_SCHEMA,
        "source_bundle": str(bundle),
        "source_bundle_sha256": {
            name: sha256(bundle / name) for name in ("commands.npz", "commands.json")
        },
        "source_episode": source["source_episode"],
        "source_sha256": hashes,
        "source_kind": source["source_kind"],
        "joint_names": source["joint_names"],
        "initial_state_origin": source.get(
            "initial_state_origin", "recorded Isaac portable initial state"
        ),
        "initial_state_override": source.get("initial_state_override"),
        "duration_ns": source["duration_ns"],
        "source_physics_dt_s": source["physics_dt"],
        "source_ticks": len(cmd),
        "request_events": len(index),
        "hold": "native zero-order holds; only identical adjacent cmd rows removed",
        "time_basis": "logical source physics tick start; native offsets/duration unchanged",
        "input_boundary": "native sim applied position targets become hardware requested position targets",
        "gripper": {
            "floor": MIN_APERTURE,
            "hardware_force_limiter_bypassed": False,
            "sampled_operator_intent": "separate robot_data channel; offsets from its own first observation; never substituted for applied targets",
            "native_request_intent": "sim requests may already be floored; no recovered pre-floor binary event log",
        },
        "expected_controller": _expected_controller(source),
        "source_physical_start": initial["scene"],
        "expected_initial_state_id": source["source_sim_metadata"]["scene"][
            "initial_state"
        ]["initial_state_id"],
        "event_intervals": {
            "minimum_ns": int(dt.min()) if len(dt) else None,
            "holds_shorter_than_hardware_4ms_loop": int(np.sum(dt < 4_000_000)),
            "interpretation": "current hardware update loop sleeps 4 ms; sending every native request cannot guarantee every target is applied by the device",
        },
        "software_stream_valid": True,
        "physical_readiness": "not checked",
        "hardware_executed": False,
        "remaining_execution_requirements": [
            "attended workspace and stop mechanism",
            "independently verified matched object/oven/arm/gripper start and controller histories",
            "already initialized follower with validated gains, gravity and active 50 N stateful limiter",
            "record send lateness, native device/application events and synchronized observations",
        ],
        "automatic_hardware_connection": "not implemented: current YAM constructor actively calibrates jaws; rd replay/serve interpolate or move/home",
    }
    _validate_plan_arrays(prepared, plan)
    report = limit_report(arrays)
    # Nominal limits are intentionally conservative. Never silently clamp even
    # numerical violations: the report distinguishes exact and 1e-9 tolerance.
    plan["limits"] = report
    plan["software_limit_check_passed"] = not any(
        report["out_of_range_counts_with_1e-9rad_tolerance"]
    ) and not any(report["initial_arm_out_of_range"])
    reconstructed = prepared["cmd"][
        np.searchsorted(prepared["offset_ns"], arrays["offset_ns"], side="right") - 1
    ]
    if not np.array_equal(reconstructed, cmd):
        raise ValueError("compaction changed held targets")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite prepared plan {destination}")
    destination.mkdir(parents=True)
    np.savez_compressed(destination / "hardware_plan.npz", **prepared)
    plan["plan_arrays_sha256"] = sha256(destination / "hardware_plan.npz")
    _write_json(destination / "hardware_plan.json", plan)
    return prepared, plan


def load_hardware_plan(path: Path):
    path = Path(path)
    plan = json.loads((path / "hardware_plan.json").read_text())
    if sha256(path / "hardware_plan.npz") != plan["plan_arrays_sha256"]:
        raise ValueError("prepared plan arrays changed")
    with np.load(path / "hardware_plan.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    _validate_plan_arrays(arrays, plan)
    return arrays, plan


def dry_run_hardware_plan(path: Path):
    """Prove exact native target reconstruction without a clock, driver or device."""
    arrays, plan = load_hardware_plan(path)
    bundle = Path(plan["source_bundle"])
    for name, digest in plan["source_bundle_sha256"].items():
        if sha256(bundle / name) != digest:
            raise ValueError(f"prepared plan's source bundle changed: {name}")
    source, metadata = load_bundle(bundle)
    hashes = _source_bindings(metadata)
    if (
        plan["source_sha256"] != hashes
        or plan["source_episode"] != metadata["source_episode"]
    ):
        raise ValueError("prepared plan source binding differs")
    if plan["expected_controller"] != _expected_controller(metadata):
        raise ValueError("prepared plan controller differs from source")
    if plan["source_physical_start"] != metadata["initial_state"]["scene"]:
        raise ValueError("prepared plan physical start differs from source")
    report = limit_report(source)
    passed = not any(report["out_of_range_counts_with_1e-9rad_tolerance"]) and not any(
        report["initial_arm_out_of_range"]
    )
    if plan["limits"] != report or plan["software_limit_check_passed"] != passed:
        raise ValueError("prepared plan limit receipt differs from source")
    changed = np.r_[True, np.any(source["cmd"][1:] != source["cmd"][:-1], axis=1)]
    expected_indices = np.flatnonzero(changed)
    for key, expected in {
        "offset_ns": source["offset_ns"][expected_indices],
        "source_tick_index": expected_indices,
        "cmd": source["cmd"][expected_indices],
        "source_requested_cmd": source["requested_cmd"][expected_indices],
        "initial_q": source["initial_q"],
        "initial_dq": source["initial_dq"],
    }.items():
        if not np.array_equal(arrays[key], expected):
            raise ValueError(f"prepared plan changed source {key}")
    indices = (
        np.searchsorted(arrays["offset_ns"], source["offset_ns"], side="right") - 1
    )
    exact = np.array_equal(arrays["cmd"][indices], source["cmd"])
    if not exact or int(plan["duration_ns"]) != int(metadata["duration_ns"]):
        raise ValueError("prepared stream does not preserve the source holds/duration")
    return {
        "status": "offline dry-run passed",
        "software_stream_valid": True,
        "native_source_ticks_reconstructed_exactly": len(indices),
        "native_request_events": len(arrays["offset_ns"]),
        "duration_ns": plan["duration_ns"],
        "interpolation": False,
        "time_scaling": False,
        "removed_only_identical_adjacent_targets": True,
        "physical_readiness": "not checked",
        "hardware_executed": False,
        "software_limit_check_passed": plan["software_limit_check_passed"],
        "timing_limit": plan["event_intervals"],
    }


def execute_prepared_hardware(
    path: Path,
    follower,
    output: Path,
    *,
    execute=False,
    attended=False,
    matched_physical_start=False,
    exclusive_control=False,
    controller_history_ready=False,
    arm_start_tolerance_rad=0.01,
    aperture_start_tolerance=0.02,
    velocity_start_tolerance=0.05,
    max_lateness_ns=None,
    timing_profile="strict",
    clock=time,
    stop_requested=lambda: False,
    enforce_initial_state=True,
):
    """Guarded API for an existing follower; never initialize, home, reset or close.

    Strict timing preserves absolute native deadlines and its historical 2 ms
    default budget. Explicit best_effort timing defaults to 50 ms: each deadline
    follows the previous API send completion by the original source interval.
    This preserves every minimum request hold without catching up or skipping;
    ordinary host jitter and API cost accumulate as recorded timing dilation.
    best_effort_unbounded uses the same holds with no host-lateness/stall cutoff.
    A final observed-pose hold is logged on stop/completion. These are software
    request/observation times, never device application timestamps or exact parity.
    """
    if not (
        execute
        and attended
        and matched_physical_start
        and exclusive_control
        and controller_history_ready
    ):
        raise ValueError(
            "execute, attended, matched_physical_start, exclusive_control and controller_history_ready must all be explicit"
        )
    if not isinstance(enforce_initial_state, bool):
        raise ValueError("enforce_initial_state must be a boolean")
    if not isinstance(timing_profile, str) or timing_profile not in TIMING_PROFILES:
        raise ValueError(f"timing_profile must be one of {tuple(TIMING_PROFILES)}")
    if max_lateness_ns is None:
        max_lateness_ns = TIMING_PROFILES[timing_profile]
    best_effort = timing_profile in ("best_effort", "best_effort_unbounded")
    if max_lateness_ns is not None and (
        isinstance(max_lateness_ns, bool)
        or not isinstance(max_lateness_ns, (int, np.integer))
        or max_lateness_ns < 0
    ):
        raise ValueError("max_lateness_ns must be a nonnegative integer")
    for value in (
        arm_start_tolerance_rad,
        aperture_start_tolerance,
        velocity_start_tolerance,
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError("start tolerances must be finite and nonnegative")
    dry_run = dry_run_hardware_plan(path)
    arrays, plan = load_hardware_plan(path)
    if not dry_run["software_limit_check_passed"]:
        raise ValueError("prepared stream exceeds nominal hardware limits")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite execution output {output}")
    info = follower.get_robot_info()
    expected = plan["expected_controller"]
    for key in ("kp", "kd", "gravity_comp_factor", "limit_gripper_effort"):
        expected_shape = (7,) if key in ("kp", "kd") else ()
        actual = np.asarray(info[key], dtype=float)
        if (
            actual.shape != expected_shape
            or not np.isfinite(actual).all()
            or not np.allclose(actual, expected[key], rtol=0, atol=1e-9)
        ):
            raise ValueError(
                f"follower setting differs from prepared controller: {key}"
            )
    initial = follower.get_observations()
    q = np.r_[initial["joint_pos"], initial["gripper_pos"]]
    dq = np.r_[initial["joint_vel"], initial["gripper_vel"]]
    if (
        q.shape != (7,)
        or dq.shape != (7,)
        or not np.isfinite(q).all()
        or not np.isfinite(dq).all()
    ):
        raise ValueError("follower start observation must be finite seven-channel q/dq")
    if enforce_initial_state and (
        np.max(np.abs(q[:6] - arrays["initial_q"][:6])) > arm_start_tolerance_rad
        or abs(q[6] - arrays["initial_q"][6]) > aperture_start_tolerance
    ):
        raise ValueError(
            "observed hardware start differs; no start movement is performed"
        )
    if (
        enforce_initial_state
        and np.max(np.abs(dq - arrays["initial_dq"])) > velocity_start_tolerance
    ):
        raise ValueError("observed hardware start velocity differs")
    # Also reject any target the actual follower would silently clamp.
    limits = np.asarray(info["joint_limits"], dtype=float)
    if (
        limits.shape != (6, 2)
        or not np.isfinite(limits).all()
        or np.any(limits[:, 0] >= limits[:, 1])
        or np.any(arrays["cmd"][:, :6] < limits[:, 0])
        or np.any(arrays["cmd"][:, :6] > limits[:, 1])
    ):
        raise ValueError("actual follower joint limits would clamp native targets")
    output.mkdir(parents=True)
    trace = []
    receipt = {
        "plan": str(Path(path).resolve()),
        "status": "running",
        "hardware_executed": False,
        "attended": attended,
        "matched_physical_start_asserted": matched_physical_start,
        "controller_history_ready_asserted": controller_history_ready,
        "numerical_initial_state_check_enforced": enforce_initial_state,
        "measured_initial_q_rad_aperture": q.tolist(),
        "measured_initial_dq_rad_s_aperture_s": dq.tolist(),
        "source_initial_q_error_rad_aperture": (q - arrays["initial_q"]).tolist(),
        "source_initial_dq_error": (dq - arrays["initial_dq"]).tolist(),
        "follower_info": {k: np.asarray(v).tolist() for k, v in info.items()},
        "clock": "monotonic software send times; device application timestamps unavailable",
        "no_start_move_or_home": True,
        "max_lateness_ns": None if max_lateness_ns is None else int(max_lateness_ns),
        "timing_profile": timing_profile,
        "timing_policy": (
            "absolute native deadlines; abort on deadline overrun"
            if timing_profile == "strict"
            else "previous send completion plus native interval; cumulative dilation, no catch-up or skipped events"
        ),
        "source_duration_ns": int(plan["duration_ns"]),
        "device_application_timestamps_available": False,
        "exact_native_timing_claimed": False,
    }
    start = clock.monotonic_ns()
    receipt["stream_start_monotonic_ns"] = start
    schedule_delay = 0
    scheduled_end = start + int(plan["duration_ns"])

    def check_host_stall(operation, before, after, expected_duration=0):
        stall = after - before - expected_duration
        if best_effort and max_lateness_ns is not None and stall > max_lateness_ns:
            receipt["host_stall"] = {
                "operation": operation,
                "before_monotonic_ns": before,
                "after_monotonic_ns": after,
                "unexpected_duration_ns": stall,
            }
            raise RuntimeError(
                f"best_effort host stall during {operation}: {stall} ns; motion aborted"
            )

    def wait_until(target):
        # Remain responsive to the attended software stop during long holds too.
        # A physical emergency stop remains owned by the caller/hardware session.
        while True:
            if stop_requested():
                raise InterruptedError("attended stop requested")
            remaining = target - clock.monotonic_ns()
            if remaining <= 0:
                return
            sleep_ns = min(remaining, 10_000_000)
            before_sleep = clock.monotonic_ns()
            clock.sleep(sleep_ns / 1e9)
            check_host_stall("wait", before_sleep, clock.monotonic_ns(), sleep_ns)

    try:
        for i, offset in enumerate(arrays["offset_ns"]):
            native_target = start + int(offset)
            target = native_target + schedule_delay
            wait_until(target)
            before = clock.monotonic_ns()
            late = before - target
            if max_lateness_ns is not None and late > max_lateness_ns:
                receipt["deadline_miss"] = {
                    "event": i,
                    "intended_offset_ns": int(offset),
                    "native_target_monotonic_ns": native_target,
                    "schedule_delay_ns": schedule_delay,
                    "target_monotonic_ns": target,
                    "observed_monotonic_ns": before,
                    "lateness_ns": late,
                }
                raise RuntimeError(
                    f"native command deadline missed at event {i}: {late} ns; no catch-up burst"
                )
            receipt["hardware_execution_attempted"] = True
            follower.command_joint_pos(arrays["cmd"][i].copy())
            receipt["hardware_executed"] = True
            after = clock.monotonic_ns()
            trace.append(
                {
                    "event": i,
                    "source_tick": int(arrays["source_tick_index"][i]),
                    "intended_offset_ns": int(offset),
                    "send_before_monotonic_ns": before,
                    "send_after_monotonic_ns": after,
                    "lateness_ns": late,
                    "native_target_monotonic_ns": native_target,
                    "scheduled_target_monotonic_ns": target,
                    "schedule_delay_before_ns": schedule_delay,
                    "source_lateness_ns": before - native_target,
                    "requested_cmd": arrays["cmd"][i].tolist(),
                    "observation": None,
                }
            )
            if best_effort:
                schedule_delay = after - native_target
            trace[-1]["schedule_delay_after_ns"] = schedule_delay
            check_host_stall("command_joint_pos", before, after)
            observation = follower.get_observations()
            observed_at = clock.monotonic_ns()
            trace[-1].update(
                observation_read_monotonic_ns=observed_at,
                observation={k: np.asarray(v).tolist() for k, v in observation.items()},
            )
            check_host_stall("get_observations", after, observed_at)
        scheduled_end = start + int(plan["duration_ns"]) + schedule_delay
        wait_until(scheduled_end)
        receipt["status"] = "complete"
    except BaseException as exc:
        receipt.update(status="failed", exception=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        stream_end = clock.monotonic_ns()
        scheduled_end = start + int(plan["duration_ns"]) + schedule_delay
        receipt.update(
            stream_end_monotonic_ns=stream_end,
            scheduled_end_monotonic_ns=scheduled_end,
            accumulated_schedule_delay_ns=schedule_delay,
            stream_wall_duration_ns=stream_end - start,
            completed_timing_dilation_ns=(stream_end - start - int(plan["duration_ns"]))
            if receipt["status"] == "complete"
            else None,
            final_hold_end_lateness_ns=(stream_end - scheduled_end)
            if receipt["status"] == "complete"
            else None,
        )
        for i, row in enumerate(trace):
            next_source = (
                int(arrays["offset_ns"][i + 1])
                if i + 1 < len(arrays["offset_ns"])
                else int(plan["duration_ns"])
            )
            row["source_hold_ns"] = next_source - row["intended_offset_ns"]
            next_send = (
                trace[i + 1]["send_before_monotonic_ns"]
                if i + 1 < len(trace)
                else stream_end
            )
            row["observed_hold_after_send_completion_ns"] = (
                next_send - row["send_after_monotonic_ns"]
            )
            row["full_hold_observed"] = (
                i + 1 < len(trace) or receipt["status"] == "complete"
            )
        # Do not continue the task or automatically move home after failure.
        # Holding observed q is the same-position PD hold used by teleop pause.
        try:
            observed = follower.get_observations()
            hold = np.r_[observed["joint_pos"], observed["gripper_pos"]]
            if hold.shape != (7,) or not np.isfinite(hold).all():
                raise ValueError("invalid final hold observation")
            hold[6] = np.clip(hold[6], MIN_APERTURE, 1.0)
            receipt["hardware_execution_attempted"] = True
            before_hold = clock.monotonic_ns()
            follower.command_joint_pos(hold.copy())
            receipt["hardware_executed"] = True
            receipt["final_observed_pose_hold"] = {
                "purpose": "completion hold"
                if receipt["status"] == "complete"
                else "abort hold",
                "requested_cmd": hold.tolist(),
                "send_before_monotonic_ns": before_hold,
                "send_after_monotonic_ns": clock.monotonic_ns(),
            }
        except BaseException as hold_error:  # noqa: BLE001 -- preserve the task failure and its final-hold receipt
            receipt["final_hold_error"] = f"{type(hold_error).__name__}: {hold_error}"
        receipt["including_final_pose_hold_wall_duration_ns"] = (
            clock.monotonic_ns() - start
        )
        completed_holds = [r for r in trace if r["full_hold_observed"]]
        receipt["request_hold_audit"] = {
            "fully_observed_holds": len(completed_holds),
            "shorter_than_source_holds": sum(
                r["observed_hold_after_send_completion_ns"] < r["source_hold_ns"]
                for r in completed_holds
            ),
            "boundary": "software API completion to next request start or final stream end; motor application unavailable",
        }
        receipt["events_sent"] = len(trace)
        receipt["successful_send_lateness_ns"] = {
            "max": max((r["lateness_ns"] for r in trace), default=None),
            "p50_p95": np.percentile(
                [r["lateness_ns"] for r in trace], [50, 95]
            ).tolist()
            if trace
            else [],
        }
        _write_json(output / "execution.json", receipt)
        _write_json(output / "send_observation_trace.json", trace)
    return receipt
