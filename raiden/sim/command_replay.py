"""Portable, held absolute joint targets for parity experiments.

Real ``robot_data`` contains observation-clock command snapshots, not a native
command-event log. Export preserves those timestamps and declares that limitation.
Isaac logs instead preserve every actual physics-tick target. This module never
imports a hardware driver and never interpolates measured states or commands.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

MIN_APERTURE = 20.0 / 95.0
JOINT_NAMES = [f"robot0_joint{i}" for i in range(1, 7)]
# get_robot.py YAM limits before its runtime +/-0.15 rad command margin.
YAM_NOMINAL_LIMITS = np.array(
    [
        [-2.618, 3.054],
        [0.0, 3.65],
        [0.0, 3.665],
        [-1.571, 1.571],
        [-1.571, 1.571],
        [-2.094, 2.094],
    ]
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_bundle(arrays: dict, metadata: dict) -> None:
    t = np.asarray(arrays["offset_ns"])
    cmd = np.asarray(arrays["cmd"])
    if metadata.get("schema") != "raiden.held_joint_commands.v1":
        raise ValueError("unsupported command bundle schema")
    if metadata.get("joint_names") != JOINT_NAMES:
        raise ValueError("joint order must be robot0_joint1..6")
    if t.dtype.kind not in "iu" or t.ndim != 1 or len(t) < 2:
        raise ValueError("need at least two integer native timestamps")
    if t[0] != 0 or np.any(np.diff(t) <= 0):
        raise ValueError("timestamps must increase strictly from zero")
    if cmd.shape != (len(t), 7) or not np.isfinite(cmd).all():
        raise ValueError("commands must be finite (N,7)")
    if np.any((cmd[:, 6] < 0) | (cmd[:, 6] > 1)):
        raise ValueError("aperture targets must lie in [0,1], 1=open")
    if int(metadata["duration_ns"]) < int(t[-1]):
        raise ValueError("duration precedes the last command")
    for key in ("initial_q", "initial_dq"):
        if np.asarray(arrays[key]).shape != (7,) or not np.isfinite(arrays[key]).all():
            raise ValueError(f"{key} must contain finite seven joint observations")
    if not 0 <= arrays["initial_q"][6] <= 1:
        raise ValueError("initial measured aperture must lie in [0,1]")


def export_episode(episode: Path, destination: Path) -> tuple[dict, dict]:
    episode = episode.resolve()
    with np.load(episode / "robot_data.npz", allow_pickle=False) as raw:
        initial_q = np.asarray(raw["follower_l_joint_pos_7d"][0], dtype=float)
        initial_dq = np.r_[
            raw["follower_l_joint_vel"][0], raw["follower_l_gripper_vel"][0]
        ]
        raw_t = np.asarray(raw["timestamps"], dtype=np.int64)
        raw_cmd = np.asarray(raw["follower_l_joint_cmd"], dtype=float)
    sources = ["robot_data.npz", "metadata.json"]
    metadata = {
        "schema": "raiden.held_joint_commands.v1",
        "joint_names": JOINT_NAMES,
        "source_episode": str(episode),
        "units": "arm=radians; aperture=normalized,1=open",
        "hold": "zero order; final sample has only its recorded hold interval",
        "hardware_ready": False,
        "hardware_readiness": "limits alone do not validate starts, workspace, contacts or force limiter",
        "initial_velocity_units": "arm rad/s; aperture/s (i2rt remapper.to_command_joint_vel_space)",
    }
    arrays = {"initial_q": initial_q, "initial_dq": initial_dq}
    if (episode / "sim_action_log.npz").exists():
        initial = json.loads((episode / "sim_initial_state.json").read_text())
        sources += ["sim_action_log.npz", "sim_initial_state.json", "sim_metadata.json"]
        dt = float(initial["physics_dt"])
        with np.load(episode / "sim_action_log.npz", allow_pickle=False) as log:
            times = np.asarray(log["sim_time"], dtype=float)
            expected = initial["sim_time"] + dt * np.arange(1, len(times) + 1)
            if not np.allclose(times, expected, atol=1e-7, rtol=0):
                raise ValueError(
                    "native Isaac ticks are not contiguous from initial state"
                )
            arrays.update(
                offset_ns=np.rint(np.arange(len(times)) * dt * 1e9).astype(np.int64),
                cmd=np.asarray(log["cmd"], dtype=float),
                requested_cmd=np.asarray(log["requested_cmd"], dtype=float),
            )
        arrays["initial_q"] = np.r_[
            initial["observations"]["joint_pos"], initial["observations"]["gripper_pos"]
        ]
        arrays["initial_dq"] = np.r_[
            initial["observations"]["joint_vel"], initial["observations"]["gripper_vel"]
        ]
        metadata.update(
            source_kind="isaac_applied_tick_targets",
            physics_dt=dt,
            duration_ns=round(len(times) * dt * 1e9),
            initial_state=initial,
            source_sim_metadata=json.loads((episode / "sim_metadata.json").read_text()),
            timing_uncertainty="application tick preserved; solver warmstarts are absent",
        )
    else:
        arrays.update(
            offset_ns=raw_t - raw_t[0], requested_cmd=raw_cmd.copy(), cmd=raw_cmd.copy()
        )
        # Recorded channel is binary teleop intent, whereas follower received this floor.
        arrays["cmd"][:, 6] = np.clip(raw_cmd[:, 6], MIN_APERTURE, 1.0)
        metadata.update(
            source_kind="real_observation_clock_command_snapshots",
            duration_ns=int(raw_t[-1] - raw_t[0]),
            aperture_mapping={
                "intent_close": 0.0,
                "inferred_follower_close": MIN_APERTURE,
            },
            timing_uncertainty="~10ms sampled last-request snapshots; native event/application times absent",
            start_uncertainty="arm observed; door/object actual starts unmeasured; scene placement is nominal",
        )
    metadata["source_sha256"] = {name: sha256(episode / name) for name in sources}
    validate_bundle(arrays, metadata)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "commands.npz").exists() or (
        destination / "commands.json"
    ).exists():
        raise FileExistsError(f"refusing to overwrite command bundle {destination}")
    np.savez_compressed(destination / "commands.npz", **arrays)
    (destination / "commands.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return arrays, metadata


def load_bundle(path: Path) -> tuple[dict, dict]:
    metadata = json.loads((path / "commands.json").read_text())
    with np.load(path / "commands.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    validate_bundle(arrays, metadata)
    return arrays, metadata


def with_recorded_host_timing(bundle: Path, rollout: Path, destination: Path):
    """Replay measured API sends while preserving their separately measured start.

    This is a distinct input experiment from the original native Isaac tick
    stream. No lag is fitted: the first recorded API send defines time zero.
    Device application timestamps remain unavailable.
    """
    bundle, rollout, destination = (
        Path(bundle).resolve(),
        Path(rollout).resolve(),
        Path(destination).resolve(),
    )
    arrays, original = load_bundle(bundle)
    if (
        original["source_kind"] != "isaac_applied_tick_targets"
        or "initial_state_override" not in original
    ):
        raise ValueError(
            "host timing requires native inputs with a separately measured shared start"
        )
    execution_path, trace_path = (
        rollout / "execution.json",
        rollout / "send_observation_trace.json",
    )
    execution, events = (
        json.loads(execution_path.read_text()),
        json.loads(trace_path.read_text()),
    )
    if (
        execution["status"] != "complete"
        or execution["events_sent"] != len(events)
        or len(events) < 2
    ):
        raise ValueError("need a complete measured API request stream")
    changes = np.r_[True, np.any(np.diff(arrays["cmd"], axis=0) != 0, axis=1)]
    expected = arrays["cmd"][changes]
    command = np.asarray([event["requested_cmd"] for event in events], dtype=float)
    if not np.array_equal(command, expected):
        raise ValueError("recorded API requests differ from the native held targets")
    send = np.asarray(
        [event["send_before_monotonic_ns"] for event in events], dtype=np.int64
    )
    completed = np.asarray(
        [event["send_after_monotonic_ns"] for event in events], dtype=np.int64
    )
    if np.any(completed < send) or np.any(np.diff(send) <= 0):
        raise ValueError("invalid measured API timing")
    end = int(execution["stream_end_monotonic_ns"])
    if end <= completed[-1]:
        raise ValueError("measured stream has no final request hold")
    result = {
        "offset_ns": send - send[0],
        "cmd": command,
        "requested_cmd": command.copy(),
        "initial_q": arrays["initial_q"].copy(),
        "initial_dq": arrays["initial_dq"].copy(),
    }
    metadata = copy.deepcopy(original)
    metadata.update(
        source_kind="hardware_host_request_events",
        duration_ns=int(end - send[0]),
        hold="actual API send-before timestamps; first send is time zero; measured final stream hold",
        timing_uncertainty="host API request times measured; next lower-controller/device application times unavailable",
        parent_native_bundle={
            "path": str(bundle),
            "sha256": {
                name: sha256(bundle / name)
                for name in ("commands.json", "commands.npz")
            },
        },
        recorded_host_timing={
            "clock": "monotonic",
            "origin_ns": int(send[0]),
            "origin_after_stream_start_ns": int(
                send[0] - execution["stream_start_monotonic_ns"]
            ),
            "rollout": str(rollout),
            "execution_sha256": sha256(execution_path),
            "trace_sha256": sha256(trace_path),
            "API_send_duration_ns_range": [
                int((completed - send).min()),
                int((completed - send).max()),
            ],
            "interpretation": "new actual-host-input simulation; original native-input comparison remains separate",
        },
    )
    validate_bundle(result, metadata)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite host-timed bundle {destination}")
    destination.mkdir(parents=True)
    np.savez_compressed(destination / "commands.npz", **result)
    (destination / "commands.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return result, metadata


def with_measured_hardware_start(bundle: Path, snapshot: Path, destination: Path):
    """Declare a new shared arm/aperture start while preserving every source input.

    Original episodes/bundles remain immutable. The separate sidecar records a
    live hardware measurement; it is not relabeled as the recorded Isaac start.
    Object/oven roots remain source-nominal/operator-matched, not newly measured.
    """
    bundle, snapshot, destination = (
        Path(bundle).resolve(),
        Path(snapshot).resolve(),
        Path(destination).resolve(),
    )
    arrays, original = load_bundle(bundle)
    if original["source_kind"] != "isaac_applied_tick_targets":
        raise ValueError("shared measured starts require native Isaac held inputs")
    measured = json.loads(snapshot.read_text())
    if not measured["health"]["healthy"] or not measured["controller_matches"]:
        raise ValueError(
            "live measured start must have healthy native control and matched settings"
        )
    q = np.asarray(measured["initial_q_rad_aperture"], dtype=float)
    dq = np.asarray(measured["initial_dq_rad_s_aperture_s"], dtype=float)
    if (
        q.shape != (7,)
        or dq.shape != (7,)
        or not np.isfinite(q).all()
        or not np.isfinite(dq).all()
    ):
        raise ValueError("measured start needs finite seven-channel q/dq")
    if not 0 <= q[6] <= 1 or np.max(abs(dq)) > 0.05:
        raise ValueError(
            "measured hardware start must be open-range and nearly stationary"
        )
    result = {key: value.copy() for key, value in arrays.items()}
    result["initial_q"], result["initial_dq"] = q.copy(), dq.copy()
    metadata = copy.deepcopy(original)
    initial = measured_initial_state(
        original["initial_state"], result, metadata, original["source_sim_metadata"]
    )
    initial["observations"].update(
        joint_pos=q[:6].tolist(),
        joint_vel=dq[:6].tolist(),
        gripper_pos=q[6:].tolist(),
        gripper_vel=dq[6:].tolist(),
    )
    initial["t_ns"], initial["monotonic_ns"] = (
        measured["host_time_ns"],
        measured["host_monotonic_ns"],
    )
    metadata["initial_state"] = initial
    metadata["initial_state_origin"] = (
        "live measured hardware arm/aperture override for both domains"
    )
    sidecar = {
        "schema": "raiden.measured_shared_initial_state.v1",
        "original_bundle": str(bundle),
        "original_bundle_sha256": {
            name: sha256(bundle / name) for name in ("commands.json", "commands.npz")
        },
        "live_snapshot_source": str(snapshot),
        "live_snapshot_sha256": sha256(snapshot),
        "live_snapshot": measured,
        "original_initial_q_rad_aperture": arrays["initial_q"].tolist(),
        "original_initial_dq_rad_s_aperture_s": arrays["initial_dq"].tolist(),
        "measured_initial_q_rad_aperture": q.tolist(),
        "measured_initial_dq_rad_s_aperture_s": dq.tolist(),
        "unchanged_inputs": {
            key: {
                "shape": list(arrays[key].shape),
                "dtype": str(arrays[key].dtype),
                "array_sha256": hashlib.sha256(arrays[key].tobytes()).hexdigest(),
            }
            for key in ("offset_ns", "cmd", "requested_cmd")
        },
        "interpretation": "new declared shared start, not exact reproduction of original simulator episode; object/oven poses remain source-nominal and operator matched; controller/solver histories are not measured hardware equivalence",
    }
    validate_bundle(result, metadata)
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite shared-start bundle {destination}"
        )
    destination.mkdir(parents=True)
    sidecar_path = destination / "initial_state_override.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    metadata["initial_state_override"] = {
        "path": str(sidecar_path),
        "sha256": sha256(sidecar_path),
        "kind": "live measured hardware arm/aperture q/dq; identical override for hardware and simulation",
    }
    np.savez_compressed(destination / "commands.npz", **result)
    (destination / "commands.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for key in ("offset_ns", "cmd", "requested_cmd"):
        if not np.array_equal(result[key], arrays[key]):
            raise AssertionError(f"measured start changed native source inputs: {key}")
    return result, metadata


def schedule(arrays: dict, metadata: dict, dt: float) -> tuple[np.ndarray, dict]:
    """Select the held command at each tick start; round events forward, never early."""
    dt_ns = round(dt * 1e9)
    if dt_ns <= 0 or not np.isclose(dt_ns * 1e-9, dt, rtol=0, atol=1e-12):
        raise ValueError("physics dt must be a positive integer number of nanoseconds")
    if metadata["source_kind"] == "isaac_applied_tick_targets" and not np.isclose(
        metadata["physics_dt"], dt, rtol=0, atol=1e-12
    ):
        raise ValueError("native tick replay requires the recorded physics dt")
    end = int(metadata["duration_ns"])
    starts = np.arange((end + dt_ns - 1) // dt_ns, dtype=np.int64) * dt_ns
    indices = np.searchsorted(arrays["offset_ns"], starts, side="right") - 1
    apply_ns = ((arrays["offset_ns"] + dt_ns - 1) // dt_ns) * dt_ns
    changed = np.r_[True, np.any(np.diff(arrays["cmd"], axis=0) != 0, axis=1)]
    active = changed & (apply_ns < starts.size * dt_ns)
    _values, counts = np.unique(apply_ns[active], return_counts=True)
    return indices, {
        "physics_dt_s": dt,
        "ticks": len(starts),
        "duration_requested_s": end * 1e-9,
        "duration_simulated_s": len(starts) * dt,
        "duration_overrun_ns": int(len(starts) * dt_ns - end),
        "changed_events": int(changed.sum()),
        "max_event_round_forward_ns": int(
            (apply_ns - arrays["offset_ns"])[active].max(initial=0)
        ),
        "changed_events_sharing_tick": int(np.maximum(counts - 1, 0).sum()),
        "last_boundary_events_without_hold": int(np.sum(changed & ~active)),
        "quantization": "first tick start at or after event; no interpolation",
    }


def limit_report(arrays: dict, limits=None) -> dict:
    limits = YAM_NOMINAL_LIMITS if limits is None else np.asarray(limits, dtype=float)
    cmd = arrays["cmd"][:, :6]
    delta = np.diff(cmd, axis=0)
    dt = np.diff(arrays["offset_ns"]) * 1e-9
    violations = (cmd < limits[:, 0]) | (cmd > limits[:, 1])
    return {
        "limits_rad": limits.tolist(),
        "out_of_range_counts": violations.sum(axis=0).tolist(),
        "out_of_range_counts_with_1e-9rad_tolerance": (
            (cmd < limits[:, 0] - 1e-9) | (cmd > limits[:, 1] + 1e-9)
        )
        .sum(axis=0)
        .tolist(),
        "initial_arm_out_of_range": (
            (arrays["initial_q"][:6] < limits[:, 0])
            | (arrays["initial_q"][:6] > limits[:, 1])
        ).tolist(),
        "cmd_min_rad": cmd.min(axis=0).tolist(),
        "cmd_max_rad": cmd.max(axis=0).tolist(),
        "max_target_step_rad": np.abs(delta).max(axis=0).tolist(),
        "max_target_slope_rad_s": (np.abs(delta) / dt[:, None]).max(axis=0).tolist(),
        "initial_q_rad_aperture": arrays["initial_q"].tolist(),
        "hardware_ready": False,
        "workspace_collision_validation": "not performed",
    }


def measured_initial_state(
    template: dict, arrays: dict, metadata: dict, live: dict
) -> dict:
    """Retain reset scene geometry; use measured q/dq once, never animate observations."""
    state = copy.deepcopy(template)
    names = state["joint_names"]["robot"]
    position = state["scene"]["articulation"]["robot"]["joint_position"][0]
    velocity = state["scene"]["articulation"]["robot"]["joint_velocity"][0]
    for j, name in enumerate(JOINT_NAMES):
        i = names.index(name)
        position[i], velocity[i] = (
            float(arrays["initial_q"][j]),
            float(arrays["initial_dq"][j]),
        )
    lo, hi = live["finger_stops_m"]
    # Pinned twin robot.xml: both positive slide coordinates, equality q_left=q_right.
    fingers = {"gripper0_right_left_finger", "gripper0_right_right_finger"}
    if set(names) != set(JOINT_NAMES) | fingers:
        raise ValueError(
            "unrecognized finger joints; audit their signs/transmission before replay"
        )
    for name in fingers:
        i = names.index(name)
        position[i] = float(lo + arrays["initial_q"][6] * (hi - lo))
        velocity[i] = float(arrays["initial_dq"][6] * (hi - lo))
    state["cmd"] = arrays["cmd"][0].tolist()
    state["requested_cmd"] = arrays["requested_cmd"][0].tolist()
    return state
