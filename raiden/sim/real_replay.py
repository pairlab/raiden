"""Offline preparation and evidence lifecycle for attended sim-to-real replay.

No hardware or camera device is imported or constructed at module import time.
The unchanged native held-command dispatcher owns timing and the final pose hold.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import threading
import time
from pathlib import Path

import numpy as np

from .command_replay import export_episode, sha256
from .hardware_replay import (
    dry_run_hardware_plan,
    execute_prepared_hardware,
    prepare_hardware_plan,
)
from .hardware_session import inspect_prepared_start, jsonable, write_json

SESSION_SCHEMA = "raiden.attended_sim_to_real/1"


def select_episodes(source_root, episode_ids):
    root = Path(source_root).resolve()
    if not episode_ids or len(set(episode_ids)) != len(episode_ids):
        raise ValueError("select at least one explicit episode ID, without duplicates")
    selected = []
    for episode_id in episode_ids:
        if not str(episode_id).isdigit():
            raise ValueError(
                f"episode IDs must be directory names, e.g. 0005: {episode_id}"
            )
        path = (root / episode_id).resolve()
        if path.parent != root or not path.is_dir():
            raise ValueError(
                f"selected episode does not exist inside source root: {path}"
            )
        for name in (
            "metadata.json",
            "robot_data.npz",
            "sim_action_log.npz",
            "sim_initial_state.json",
            "sim_metadata.json",
            "calibration_results.json",
        ):
            if not (path / name).is_file():
                raise ValueError(f"selected sim episode is missing {name}: {path}")
        meta = json.loads((path / "metadata.json").read_text())
        if not meta.get("complete"):
            raise ValueError(f"selected episode is incomplete: {path}")
        selected.append(path)
    return selected


def prepare_selection(
    source_root,
    episode_ids,
    destination,
    *,
    expected_state="parity_s2",
    accepted_calibration_id="20261018_175242",
    camera_config,
):
    from .replay_capture import validate_camera_config

    episodes = select_episodes(source_root, episode_ids)
    config_path = Path(camera_config).resolve()
    config = json.loads(config_path.read_text())
    validate_camera_config(config)
    # Validate the entire selection before creating any plans.
    for episode in episodes:
        sim = json.loads((episode / "sim_metadata.json").read_text())
        if sim["scene"]["initial_state"]["initial_state_id"] != expected_state:
            raise ValueError(f"{episode}: source start is not {expected_state}")
        calibration = sim["calibration"]
        if accepted_calibration_id not in json.dumps(calibration):
            raise ValueError(
                f"{episode}: accepted calibration {accepted_calibration_id} is absent"
            )
        # calibration_results.json is the simulator's exported camera/table
        # schema, rather than the original accepted calibration document.
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": SESSION_SCHEMA,
        "status": "preparing",
        "hardware_initialized": False,
        "source_root": str(Path(source_root).resolve()),
        "expected_state": expected_state,
        "accepted_calibration_id": accepted_calibration_id,
        "camera_config": str(config_path),
        "camera_config_sha256": sha256(config_path),
        "timing": "source applied physics-tick targets; identical adjacent rows compacted; no interpolation or 150 Hz resampling",
        "episodes": [],
        "limitations": [
            "simulation solver/contact/limiter history is not reproduced on hardware",
            "host API and CAN request/feedback times are measured; device application timestamps unavailable",
            "source object poses are nominal; operator placement plus camera evidence is required for every run",
        ],
    }
    try:
        for episode in episodes:
            folder = destination / episode.name
            folder.mkdir()
            _, source = export_episode(episode, folder / "bundle")
            _, plan = prepare_hardware_plan(folder / "bundle", folder / "plan")
            dry = dry_run_hardware_plan(folder / "plan")
            if not dry["software_limit_check_passed"]:
                raise ValueError(
                    f"{episode}: native commands exceed nominal arm limits"
                )
            write_json(folder / "dry_run.json", dry)
            sim = source["source_sim_metadata"]
            write_json(folder / "accepted_calibration.json", sim["calibration"])
            manifest["episodes"].append(
                {
                    "id": episode.name,
                    "source_episode": str(episode),
                    "source_recording": json.loads(
                        (episode / "metadata.json").read_text()
                    ),
                    "plan": f"{episode.name}/plan",
                    "plan_sha256": sha256(folder / "plan/hardware_plan.json"),
                    "calibration_sha256": sim["calibration_sha256"],
                    "source_calibration_results_sha256": sha256(
                        episode / "calibration_results.json"
                    ),
                    "accepted_calibration_file_sha256": sha256(
                        folder / "accepted_calibration.json"
                    ),
                    "source_start": sim["scene"]["initial_state"],
                    "door_runtime": sim.get("door_runtime"),
                    "gripper_runtime": sim.get("gripper_runtime"),
                    "duration_ns": plan["duration_ns"],
                    "request_events": plan["request_events"],
                    "dry_run": dry,
                }
            )
        manifest["status"] = "prepared"
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(destination / "prepared.json", manifest)
    return manifest


def validate_prepared(destination):
    destination = Path(destination).resolve()
    manifest = json.loads((destination / "prepared.json").read_text())
    if (
        manifest.get("schema") != SESSION_SCHEMA
        or manifest.get("status") != "prepared"
        or not manifest.get("episodes")
    ):
        raise ValueError("selected preparation is incomplete or unsupported")
    config = Path(manifest["camera_config"])
    if sha256(config) != manifest["camera_config_sha256"]:
        raise ValueError(
            "camera configuration changed since preparation; prepare a new session"
        )
    ids = [row["id"] for row in manifest["episodes"]]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate prepared episode IDs")
    for row in manifest["episodes"]:
        plan = (destination / row["plan"]).resolve()
        if (
            not plan.is_relative_to(destination)
            or sha256(plan / "hardware_plan.json") != row["plan_sha256"]
        ):
            raise ValueError("prepared plan binding changed")
        if (
            sha256(plan.parent / "accepted_calibration.json")
            != row["accepted_calibration_file_sha256"]
        ):
            raise ValueError("accepted calibration copy changed")
        if (
            sha256(Path(row["source_episode"]) / "calibration_results.json")
            != row["source_calibration_results_sha256"]
        ):
            raise ValueError("source exported calibration changed")
        dry_run_hardware_plan(plan)
    return manifest


def controller_processes(channel, *, proc_root=Path("/proc"), exclude=()):
    """Best-effort read-only blockers; a cooperative flock alone is insufficient."""
    # Older entry points can still own hardware even after their files are removed.
    scripts = {
        "parity_fresh_hardware.py",
        "parity_native_hardware.py",
        "parity_reset.py",
        "parity_hardware.py",
        "calibration_hardware.py",
        "replay_sim_in_real.py",
        "read_arm_poses.py",
        "gravity_comp_capture.py",
        "quest_teleop.py",
        "real2sim_table_teleop.py",
        "run_charuco_trajectory.py",
    }
    result = []
    for folder in Path(proc_root).iterdir():
        if not folder.name.isdigit() or int(folder.name) in exclude:
            continue
        try:
            args = [
                p.decode(errors="replace")
                for p in (folder / "cmdline").read_bytes().split(b"\0")
                if p
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not args or not (
            re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", Path(args[0]).name)
            or Path(args[0]).name in {"rd", "uv"}
        ):
            continue
        if "--sim" in args or any(arg.startswith("--sim=") for arg in args):
            continue
        if "--channel" in args:
            index = args.index("--channel")
            if index + 1 < len(args) and args[index + 1] != channel:
                continue
        names = {Path(arg).name for arg in args}
        known = bool(names & scripts)
        if "replay_sim_in_real.py" in names and "--execute" not in args:
            known = False
        rd = any(Path(arg).name == "rd" for arg in args) and any(
            arg in {"record", "serve", "replay"} for arg in args
        )
        module = (
            "-m" in args
            and any(arg.startswith("raiden") for arg in args)
            and any(arg in {"record", "serve", "replay"} for arg in args)
        )
        inline = "-c" in args and any("get_yam_robot(" in arg for arg in args)
        if known or rd or module or inline:
            result.append({"pid": int(folder.name), "argv": args})
    return result


def ancestor_pids():
    result, pid = set(), os.getpid()
    while pid > 0 and pid not in result:
        result.add(pid)
        try:
            status = Path(f"/proc/{pid}/status").read_text()
            pid = int(
                next(
                    line.split()[1]
                    for line in status.splitlines()
                    if line.startswith("PPid:")
                )
            )
        except (OSError, StopIteration, ValueError):
            break
    return result


class NativeTraceWriter:
    """Drain native rings during each trial so later trials cannot overwrite it."""

    def __init__(self, recorder, destination, interval=0.25):
        self.recorder, self.destination, self.interval = (
            recorder,
            Path(destination),
            interval,
        )
        self.stop_event = threading.Event()
        self.error = None
        self.counts = [0, 0]

    def start(self):
        self.destination.mkdir(parents=True, exist_ok=False)
        with self.recorder.lock:
            self.baseline = [
                self.recorder.total_can_rows,
                self.recorder.total_controller_rows,
            ]
            self.baseline_errors = self.recorder.logging_error_count
            self.recorder.can_rows.clear()
            self.recorder.controller_rows.clear()
        self.streams = []
        try:
            for name in ("native_can_transactions", "lower_controller"):
                self.streams.append(
                    gzip.open(self.destination / f"{name}.jsonl.gz", "wt")
                )
        except BaseException:
            for stream in self.streams:
                stream.close()
            raise
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="replay-native-evidence"
        )
        self.thread.start()

    def _drain(self):
        with self.recorder.lock:
            rows = [list(self.recorder.can_rows), list(self.recorder.controller_rows)]
            self.recorder.can_rows.clear()
            self.recorder.controller_rows.clear()
            totals = [self.recorder.total_can_rows, self.recorder.total_controller_rows]
        overflow = False
        for index, (stream, batch) in enumerate(zip(self.streams, rows)):
            if totals[index] - self.baseline[index] != self.counts[index] + len(batch):
                overflow = True
            for row in batch:
                stream.write(json.dumps(jsonable(row), allow_nan=False) + "\n")
            self.counts[index] += len(batch)
            stream.flush()
        if overflow:
            raise RuntimeError(
                "native telemetry ring overflowed; surviving rows retained but trial evidence is incomplete"
            )

    def _run(self):
        try:
            while not self.stop_event.wait(self.interval):
                self._drain()
        except BaseException as error:
            self.error = f"{type(error).__name__}: {error}"

    def check(self):
        if self.error:
            raise RuntimeError(f"native telemetry writer failed: {self.error}")
        if self.recorder.logging_error_count != self.baseline_errors:
            raise RuntimeError(
                "native recorder reported a new logging failure; trial evidence is incomplete"
            )

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=3.0)
        try:
            if self.thread.is_alive():
                raise RuntimeError("native telemetry writer did not stop")
            self._drain()
            self.check()
            if not all(self.counts):
                raise RuntimeError("native telemetry is empty")
        except BaseException as error:
            self.error = self.error or f"{type(error).__name__}: {error}"
            raise
        finally:
            for stream in self.streams:
                stream.close()
            write_json(
                self.destination / "instrumentation.json",
                {
                    "status": "failed" if self.error else "complete",
                    "error": self.error,
                    "native_can_transactions": self.counts[0],
                    "lower_controller_rows": self.counts[1],
                    "start_total_can_and_controller": self.baseline,
                    "end_total_can_and_controller": [
                        self.recorder.total_can_rows,
                        self.recorder.total_controller_rows,
                    ],
                    "start_logging_error_count": self.baseline_errors,
                    "bootstrap_events": self.recorder.bootstrap_events,
                    "logging_error_count": self.recorder.logging_error_count,
                    "logging_errors": list(self.recorder.logging_errors),
                    "interpretation": "Per-run host CAN request/feedback and post-limiter buffers; q/dq/effort retained in native row units. No device application timestamp or simulation controller-history equivalence.",
                },
            )


def run_episode(
    plan,
    follower,
    recorder,
    destination,
    camera_config,
    *,
    stop_requested=lambda: False,
    timing_profile="best_effort_unbounded",
    arm_start_tolerance_rad=0.01,
    aperture_start_tolerance=0.02,
    velocity_start_tolerance=0.05,
    camera_factory=None,
    trace_factory=NativeTraceWriter,
    clock=time,
    enforce_initial_state=True,
):
    """One explicitly launched trial, with evidence retained on every failure."""
    from .replay_capture import DualCameraCapture

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    capture = (camera_factory or DualCameraCapture)(
        camera_config, destination / "cameras"
    )
    trace = trace_factory(recorder, destination / "native_trace")
    trace_started, capture_started = False, False
    receipt = {
        "schema": SESSION_SCHEMA,
        "status": "starting",
        "plan": str(Path(plan).resolve()),
        "host_start_wall_ns": clock.time_ns(),
        "host_start_monotonic_ns": clock.monotonic_ns(),
        "operator_asserted_matched_objects_closed_door_and_attended_stop": True,
        "start_tolerances": {
            "arm_rad": arm_start_tolerance_rad,
            "aperture": aperture_start_tolerance,
            "velocity_driver_normalized_units": velocity_start_tolerance,
        },
        "controller_history": "fresh hardware limiter/controller history; not equivalent to simulation history",
        "numerical_initial_state_check_enforced": enforce_initial_state,
        "initial_state_policy": (
            "require source q/aperture/dq within tolerances"
            if enforce_initial_state
            else "user requested numerical start check disabled; physical reset accepted by operator; measured/source deviations retained"
        ),
        "device_application_timestamps_available": False,
    }

    def measured_start():
        snapshot = inspect_prepared_start(plan, follower, recorder, follower.robot)
        snapshot["default_tolerance_start_ready"] = snapshot["numerical_start_ready"]
        error = snapshot["initial_error_rad_aperture"]
        velocity_error = snapshot["initial_velocity_error"]
        snapshot["numerical_start_ready"] = bool(
            snapshot["controller_matches"]
            and np.isfinite(error).all()
            and np.isfinite(velocity_error).all()
            and max(abs(error[:6])) <= arm_start_tolerance_rad
            and abs(error[6]) <= aperture_start_tolerance
            and max(abs(velocity_error)) <= velocity_start_tolerance
        )
        snapshot["effective_start_tolerances"] = receipt["start_tolerances"]
        snapshot["numerical_initial_state_check_enforced"] = enforce_initial_state
        snapshot["next_action"] = (
            "source numerical start independently rechecked by dispatcher"
            if enforce_initial_state
            else "user requested numerical start rejection disabled; measured differences are diagnostic only"
        )
        return snapshot

    try:
        dry_run_hardware_plan(plan)
        write_json(destination / "measured_start_before_camera.json", measured_start())
        trace.start()
        trace_started = True
        capture_started = True
        capture.start(stop_requested=stop_requested)
        capture.check()
        trace.check()
        snapshot = measured_start()
        write_json(destination / "measured_start.json", snapshot)
        receipt["both_cameras_recording_before_dispatch"] = capture.snapshot()

        def stopped():
            if stop_requested():
                return True
            capture.check()
            trace.check()
            return False

        receipt["execution"] = execute_prepared_hardware(
            plan,
            follower,
            destination / "rollout",
            execute=True,
            attended=True,
            matched_physical_start=True,
            exclusive_control=True,
            # Existing dispatcher requires this assertion. It means the physical
            # history was reviewed, never that it equals source simulator history.
            controller_history_ready=True,
            timing_profile=timing_profile,
            arm_start_tolerance_rad=arm_start_tolerance_rad,
            aperture_start_tolerance=aperture_start_tolerance,
            velocity_start_tolerance=velocity_start_tolerance,
            stop_requested=stopped,
            clock=clock,
            enforce_initial_state=enforce_initial_state,
        )
        receipt["status"] = "complete"
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        cleanup_errors = []
        for label, active, close in (
            ("camera", capture_started, capture.stop),
            ("native_trace", trace_started, trace.close),
        ):
            if active:
                try:
                    close()
                except BaseException as error:
                    cleanup_errors.append(f"{label}: {type(error).__name__}: {error}")
        receipt.update(
            host_end_wall_ns=clock.time_ns(), host_end_monotonic_ns=clock.monotonic_ns()
        )
        if cleanup_errors:
            receipt.update(status="failed", cleanup_errors=cleanup_errors)
        write_json(destination / "trial.json", receipt)
        if cleanup_errors and "error" not in receipt:
            raise RuntimeError("; ".join(cleanup_errors))
    return receipt
