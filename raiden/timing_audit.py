"""Read-only timing diagnostics for saved episodes: python -m raiden.timing_audit.

Native bag metadata is read through a file-only playback device, independently
for each stream. No live-camera factory, image alignment, or pixel decoding is
used. SDK timestamps describe recorded clocks, not measured exposure simultaneity.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import threading
import time

import numpy as np

from raiden.realsense_files import list_realsense_recordings


NATIVE_TIMING_FIELDS = (
    "frame_counter", "frame_timestamp", "sensor_timestamp", "time_of_arrival",
    "backend_timestamp", "actual_exposure", "actual_fps",
)


def _summary(values, scale=1.0) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1) / scale
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "min": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "count": len(values), "min": float(np.min(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)), "max": float(np.max(values)),
    }


def _frame_record(frame, rs) -> dict:
    metadata = {}
    for name in NATIVE_TIMING_FIELDS:
        key = getattr(rs.frame_metadata_value, name, None)
        if key is not None and frame.supports_frame_metadata(key):
            metadata[name] = int(frame.get_frame_metadata(key))
    return {
        "timestamp_ns": int(frame.get_timestamp() * 1_000_000),
        "timestamp_domain": str(frame.get_frame_timestamp_domain()).split(".")[-1],
        "frame_number": int(frame.get_frame_number()), "native_metadata": metadata,
    }


def read_bag(path: Path, *, max_seconds=120.0, max_frames=1_000_000, rs=None) -> dict:
    """Read independent recorded streams, with bounded runtime and full cleanup.

    ``rs`` is injectable for hardware-free tests. Loading a file via
    context.load_device never selects or opens an enumerated physical camera.
    The limit counts color and depth frames together. Partial reads are labelled.
    """
    path = Path(path).resolve(strict=True)
    if max_seconds <= 0 or max_frames <= 0:
        raise ValueError("Playback limits must be positive")
    if rs is None:
        import pyrealsense2 as rs

    context = None
    playback = None
    opened, started = [], []
    samples = {"color": [], "depth": []}
    errors, cleanup_errors = [], []
    lock = threading.Lock()
    limit_reached = threading.Event()
    count = 0
    reason = "eof"
    begin = time.monotonic()

    def receive(frame):
        nonlocal count
        try:
            stream = frame.get_profile().stream_type()
            name = "color" if stream == rs.stream.color else "depth"
            record = _frame_record(frame, rs)
            with lock:
                if count >= max_frames:
                    limit_reached.set()
                    return
                samples[name].append(record)
                count += 1
                if count >= max_frames:
                    limit_reached.set()
        except Exception as error:
            with lock:
                errors.append(f"{type(error).__name__}: {error}")

    try:
        # Disable physical-device backends as well as selecting only a file.
        context = rs.context({"device-mask": int(rs.product_line.sw_only)})
        playback = context.load_device(str(path))
        playback.set_real_time(False)
        playback.pause()
        for sensor in playback.query_sensors():
            profiles = [p for p in sensor.get_stream_profiles()
                        if p.stream_type() in (rs.stream.color, rs.stream.depth)]
            if profiles:
                opened.append(sensor)
                sensor.open(profiles)
                started.append(sensor)
                sensor.start(receive)
        if not started:
            reason = "no_color_or_depth_streams"
        else:
            playback.resume()
            while True:
                if errors:
                    reason = "metadata_error"
                    break
                if limit_reached.is_set():
                    reason = "frame_limit"
                    break
                if playback.current_status() == rs.playback_status.stopped:
                    break
                if time.monotonic() - begin >= max_seconds:
                    reason = "time_limit"
                    break
                time.sleep(0.01)
    except Exception as error:
        reason = "playback_error"
        errors.append(f"{type(error).__name__}: {error}")
    finally:
        for sensor in reversed(started):
            try:
                sensor.stop()
            except Exception as error:
                cleanup_errors.append(f"stop: {error}")
        for sensor in reversed(opened):
            try:
                sensor.close()
            except Exception as error:
                cleanup_errors.append(f"close: {error}")
        if playback is not None:
            try:
                context.unload_device(str(path))
            except Exception as error:
                cleanup_errors.append(f"unload: {error}")
    if reason == "eof" and errors:
        reason = "metadata_error"
    elif reason == "eof" and count == 0:
        reason = "no_frames"
    return {"samples": samples, "read": {
        "complete": reason == "eof" and not cleanup_errors, "stop_reason": reason,
        "frames_read": count, "errors": errors, "cleanup_errors": cleanup_errors,
        "max_seconds": max_seconds, "max_frames": max_frames,
    }}


def _stream_summary(records) -> dict:
    if not records:
        return {"frame_count": 0, "missing": "No recorded frames available"}
    timestamps = np.array([r["timestamp_ns"] for r in records], dtype=np.int64)
    counters = np.array([r["native_metadata"].get("frame_counter", r["frame_number"])
                         for r in records], dtype=np.int64)
    delta = np.diff(counters)
    timestamp_delta = np.diff(timestamps)
    domains = [r["timestamp_domain"] for r in records]
    same_domain = np.array([left == right for left, right in zip(domains, domains[1:])], dtype=bool)
    coverage = {}
    for domain in sorted(set(domains)):
        times = timestamps[np.array([value == domain for value in domains])]
        coverage[domain] = {"first_timestamp_ns": int(times[0]), "last_timestamp_ns": int(times[-1]),
                            "time_span_ms": float(times[-1] - times[0]) / 1e6}
    metadata = {}
    for field in NATIVE_TIMING_FIELDS:
        values = [r["native_metadata"][field] for r in records
                  if field in r["native_metadata"]]
        metadata[field] = {"available_frames": len(values),
                           "first_raw": values[0] if values else None,
                           "last_raw": values[-1] if values else None}
    return {
        "frame_count": len(records),
        "timestamp_domains": dict(Counter(r["timestamp_domain"] for r in records)),
        "first_timestamp_ns": int(timestamps[0]), "last_timestamp_ns": int(timestamps[-1]),
        "time_span_ms": float(timestamps[-1] - timestamps[0]) / 1e6 if len(coverage) == 1 else None,
        "coverage_by_timestamp_domain": coverage,
        "timestamp_domain_transitions": int(np.count_nonzero(~same_domain)),
        "inter_frame_ms": _summary(timestamp_delta[same_domain], 1e6),
        "nonincreasing_timestamps": int(np.count_nonzero(timestamp_delta[same_domain] <= 0)),
        "duplicate_timestamps": len(timestamps) - len(np.unique(timestamps)),
        "duplicate_frame_counters": len(counters) - len(np.unique(counters)),
        "backward_frame_counters": int(np.count_nonzero(delta < 0)),
        "frame_counter_gap_events": int(np.count_nonzero(delta > 1)),
        "inferred_missing_frames": int(np.maximum(delta - 1, 0).sum()),
        "counter_source": "native frame_counter when available, otherwise SDK frame number",
        "native_metadata": metadata,
        "interpretation": "Counter gaps identify missing recorded stream frames; the loss location is unknown. Native metadata units are retained without reinterpretation.",
    }


def _nearest(reference, target) -> tuple[np.ndarray, np.ndarray]:
    """Indices in sorted target, signed target-minus-reference residuals."""
    reference = np.asarray(reference, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    right = np.searchsorted(target, reference).clip(0, len(target) - 1)
    left = np.maximum(right - 1, 0)
    indices = np.where(np.abs(target[left] - reference) <= np.abs(target[right] - reference),
                       left, right)
    return indices, target[indices] - reference


def _selection_summary(reference, target) -> dict:
    if not len(reference) or not len(target):
        return {"missing": "Both streams need comparable timestamps"}
    reference, target = np.sort(reference), np.sort(target)
    indices, residual = _nearest(reference, target)
    covered = (reference >= target[0]) & (reference <= target[-1])
    return {
        "reference_frame_count": len(reference), "target_frame_count": len(target),
        "reference_frames_within_target_coverage": int(covered.sum()),
        "reference_frames_outside_target_coverage": int((~covered).sum()),
        "selected_minus_reference_ms": _summary(residual, 1e6),
        "absolute_residual_ms": _summary(np.abs(residual), 1e6),
        "within_coverage_absolute_residual_ms": _summary(np.abs(residual[covered]), 1e6),
        "repeated_target_selections": len(indices) - len(np.unique(indices)),
    }


def _map_host_times(records, offset_ns, receipt) -> tuple[np.ndarray, dict]:
    mapped, methods = [], Counter()
    measurement = receipt.get("clock_mapping_receipt")
    offset_domain = ((measurement or {}).get("frame_timestamp_domain") or "").split(".")[-1]
    for record in records:
        domain = record["timestamp_domain"].split(".")[-1]
        timestamp = record["timestamp_ns"]
        if domain in ("global_time", "system_time"):
            mapped.append(timestamp)
            methods["sdk_" + domain] += 1
        elif domain == "hardware_clock" and offset_ns is not None and offset_domain in ("", "hardware_clock"):
            mapped.append(timestamp + int(offset_ns))
            methods["recorded_host_receipt_offset"] += 1
        else:
            methods["unmapped_" + domain] += 1
    wait_width_ms = None
    if measurement and "host_receive_ns" in measurement and "host_wait_start_ns" in measurement:
        wait_width_ms = (measurement["host_receive_ns"] - measurement["host_wait_start_ns"]) / 1e6
    return np.array(mapped, dtype=np.int64), {
        "methods": dict(methods), "host_mapped_frames": len(mapped),
        "unmapped_frames": len(records) - len(mapped), "recorded_offset_ns": offset_ns,
        "recorded_offset_timestamp_domain": offset_domain or None,
        "recording_start_measurement": measurement,
        "measurement_provenance_available": bool(measurement),
        "measurement_wait_width_ms": wait_width_ms,
        "physical_clock_error_bound_ms": None,
        "global_time_enabled_readback": receipt.get("color_receipt", {}).get(
            "color_options", {}).get("global_time_enabled"),
        "interpretation": (
            "SDK global/system timestamps share a declared host domain; hardware-relative "
            "clocks require a saved host-receipt offset. Offset accuracy, drift and "
            "transport delay are unmeasured. A legacy offset has no measurement provenance."
        ),
    }


def _rgb_depth_summary(color, depth) -> dict:
    # Hardware clocks may be compared only within this recorded device and domain.
    comparable = []
    for domain in sorted(set(r["timestamp_domain"] for r in color + depth)):
        if domain not in ("hardware_clock", "system_time", "global_time"):
            continue
        c = [r["timestamp_ns"] for r in color if r["timestamp_domain"] == domain]
        d = [r["timestamp_ns"] for r in depth if r["timestamp_domain"] == domain]
        if c and d:
            comparable.append({"timestamp_domain": domain,
                               **_selection_summary(np.array(c), np.array(d))})
    return {
        "by_timestamp_domain": comparable,
        "missing": None if comparable else "No RGB/depth frames in a known matching device clock domain",
        "interpretation": "Nearest native depth timestamp minus RGB timestamp, before depth alignment; does not prove matched exposure times or hardware synchronization.",
    }


def audit_robot(path: Path) -> dict:
    """Summarize host read/cache/API timing without inferring CAN acquisition time."""
    if not path.is_file():
        return {"missing": "robot_data.npz is absent"}
    with np.load(path, allow_pickle=False) as data:
        row_times = data["timestamps"] if "timestamps" in data else np.array([], dtype=np.int64)
        row_end = data["readout_end_ns"] if "readout_end_ns" in data else None
        arms = sorted({key.removesuffix("_joint_pos") for key in data.files
                       if key.endswith("_joint_pos")})
        result = {"row_count": len(row_times), "row_interval_ms": _summary(np.diff(row_times), 1e6),
                  "arms": {}, "interpretation": "Feedback age describes the cached host feedback read timestamp, not CAN acquisition. Command age describes host API issue, not physical motion latency."}
        for arm in arms:
            def get(suffix):
                return data[f"{arm}_{suffix}"] if f"{arm}_{suffix}" in data else None

            report, missing = {}, []
            read_end = get("read_end_ns")
            for name, start_name, end_name in (
                ("feedback_read_bracket_ms", "read_start", "read_end"),
                ("command_api_duration_ms", "command_start", "command_end"),
            ):
                start, end = get(start_name + "_monotonic_ns"), get(end_name + "_monotonic_ns")
                clock = "host_monotonic"
                if start is None or end is None:
                    start, end = get(start_name + "_ns"), get(end_name + "_ns")
                    if start is None and name == "command_api_duration_ms":
                        start = get("command_timestamp_ns")
                    clock = "host_wall_clock"
                if start is None or end is None:
                    missing.append(name)
                else:
                    valid = (start >= 0) & (end >= 0)
                    report[name] = {**_summary(end[valid] - start[valid], 1e6), "clock": clock}
            for name, prefix in (("feedback_read_age_ms", "state"), ("command_age_ms", "command")):
                timestamp, valid = get(prefix + "_timestamp_ns"), get(prefix + "_timestamp_valid")
                source = get(prefix + "_timestamp_source")
                reference = row_end if prefix == "command" else read_end
                if timestamp is None or valid is None or reference is None:
                    missing.append(name)
                    continue
                valid = valid.astype(bool) & (timestamp >= 0) & (reference >= 0)
                ages = reference[valid] - timestamp[valid]
                report[name] = {**_summary(ages, 1e6), "unavailable_rows": len(valid) - int(valid.sum()),
                                "negative_age_rows": int(np.count_nonzero(ages < 0)),
                                "reference": "row_capture_completion" if prefix == "command" else "arm_observation_read_end",
                                "timestamp_sources": dict(Counter(str(x) for x in source)) if source is not None else {"unavailable": len(valid)}}
                if prefix == "command":
                    report[name]["interpretation"] = "API issue age at completed observation/command capture; this is not motor execution latency."
                    report["rows_reusing_previous_command_issue"] = int(np.count_nonzero(
                        valid[1:] & valid[:-1] & (timestamp[1:] == timestamp[:-1])))
            command = get("joint_cmd")
            if command is not None and command.ndim == 2:
                report["unchanged_consecutive_command_vectors"] = int(np.count_nonzero(np.all(command[1:] == command[:-1], axis=1)))
            report["missing_metrics"] = missing
            if missing:
                report["missing_interpretation"] = "Legacy or unavailable timing fields; no freshness or simultaneity inferred from row labels or unchanged values."
            result["arms"][arm] = report
        return result


def audit_alignment(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {"missing": "No camera_alignment.json supplied/found; conversion selections cannot be reconstructed from this audit"}
    report = json.loads(path.read_text())
    streams = {}
    for name, stream in report.get("streams", {}).items():
        residual = np.asarray(stream.get("residual_ns", []), dtype=np.int64)
        indices = np.asarray(stream.get("selected_source_indices", []), dtype=np.int64)
        streams[name] = {
            **{key: value for key, value in stream.items()
               if key not in ("selected_source_indices", "selected_timestamps_ns", "residual_ns")},
            "selection_count": len(indices),
            "selected_minus_grid_ms": _summary(residual, 1e6),
            "absolute_residual_ms": _summary(np.abs(residual), 1e6),
            "repeated_source_selections": len(indices) - len(np.unique(indices)),
        }
    return {"path": str(path),
            **{key: value for key, value in report.items()
               if key not in ("streams", "grid_timestamps_ns")}, "streams": streams,
            "interpretation": "Saved conversion nearest-frame selection diagnostics. Relative-camera-start alignment does not establish a shared physical clock or exposure synchronization."}


def audit_episode(episode: Path, *, alignment_report=None, reference_camera=None,
                  max_seconds=120.0, max_frames=1_000_000, bag_reader=read_bag) -> dict:
    episode = Path(episode).resolve(strict=True)
    native_files = list_realsense_recordings(episode / "cameras")
    robot_path = episode / "robot_data.npz"
    if not native_files and not robot_path.is_file():
        raise ValueError(
            f"{episode}: expected a saved raw episode containing cameras/*.bag, "
            "cameras/*.db3, or robot_data.npz"
        )
    metadata_path = episode / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    cameras, mapped = {}, {}
    for bag in native_files:
        result = bag_reader(bag, max_seconds=max_seconds, max_frames=max_frames)
        color, depth = result["samples"]["color"], result["samples"]["depth"]
        receipt = metadata.get("camera_capture_receipts", {}).get(bag.stem, {})
        mapped[bag.stem], clock = _map_host_times(
            color, metadata.get("realsense_clock_offsets", {}).get(bag.stem), receipt)
        cameras[bag.stem] = {"path": str(bag), "read": result["read"],
                             "color": _stream_summary(color), "depth": _stream_summary(depth),
                             "rgb_depth": _rgb_depth_summary(color, depth), "clock_mapping": clock}
    reference_camera = reference_camera or next(iter(cameras), None)
    if reference_camera is not None and reference_camera not in cameras:
        raise ValueError(f"Reference camera {reference_camera!r} has no saved native RealSense file")
    cross_camera = {}
    for name, times in mapped.items():
        if name != reference_camera:
            cross_camera[name] = _selection_summary(mapped[reference_camera], times)
    alignment_path = Path(alignment_report) if alignment_report else episode / "camera_alignment.json"
    if alignment_report is None and not alignment_path.is_file():
        alignment_path = episode / "0000" / "camera_alignment.json"
    return {
        "schema": "raiden.saved_episode_timing_audit/1", "episode": str(episode),
        "cameras": cameras, "reference_camera": reference_camera,
        "cross_camera_host_timestamp_nearest_selection": cross_camera,
        "robot": audit_robot(robot_path),
        "conversion_alignment": audit_alignment(alignment_path),
        "limitations": [
            "Timestamp diagnostics do not measure physical exposure synchronization, flash-event precision, robot sensor acquisition time or command-to-motion delay.",
            "Hardware-relative cameras with no recorded host mapping are excluded from cross-camera host comparisons; their clocks must not be compared directly.",
            "Native RealSense .bag/.db3 files are audited; other camera formats require their SDK/exported timing diagnostics.",
        ],
    }


def _console(report) -> str:
    lines = [f"Saved timing audit: {report['episode']}"]
    for name, camera in report["cameras"].items():
        lines.append(f"  {name}: {camera['read']['stop_reason']}")
        for stream in ("color", "depth"):
            item = camera[stream]
            lines.append(f"    {stream}: {item['frame_count']} frames"
                         + (f", inferred missing {item['inferred_missing_frames']}, duplicate counters {item['duplicate_frame_counters']}, max interval {item['inter_frame_ms']['max']} ms" if item['frame_count'] else " (unavailable)"))
        for pair in camera["rgb_depth"]["by_timestamp_domain"]:
            stats = pair["absolute_residual_ms"]
            lines.append(f"    RGB/depth nearest skew ({pair['timestamp_domain']}): p95 {stats['p95']:.3f} ms, max {stats['max']:.3f} ms")
    for name, selection in report["cross_camera_host_timestamp_nearest_selection"].items():
        if "missing" in selection:
            lines.append(f"  {report['reference_camera']} -> {name}: {selection['missing']}")
        else:
            stats = selection["absolute_residual_ms"]
            lines.append(f"  {report['reference_camera']} -> {name} nearest host timestamp: p95 {stats['p95']:.3f} ms, max {stats['max']:.3f} ms; {selection['reference_frames_outside_target_coverage']} outside coverage")
    robot = report["robot"]
    if "missing" in robot:
        lines.append(f"  Robot: {robot['missing']}")
    for arm, data in robot.get("arms", {}).items():
        for metric in ("feedback_read_bracket_ms", "feedback_read_age_ms", "command_age_ms"):
            stats = data.get(metric, {})
            lines.append(f"  {arm} {metric}: p95 {stats.get('p95')} ms, max {stats.get('max')} ms"
                         if stats.get("count", 0) else f"  {arm} {metric}: unavailable")
    alignment = report["conversion_alignment"]
    lines.append(f"  Conversion alignment: {alignment.get('missing', alignment.get('path'))}")
    lines.append("  These timestamps do not establish exposure synchronization or hardware acquisition precision.")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path, help="Saved raw episode directory")
    parser.add_argument("--alignment-report", type=Path, help="Converted sequence camera_alignment.json")
    parser.add_argument("--reference-camera", help="Native RealSense file name without extension; defaults to first sorted name")
    parser.add_argument("--max-seconds", type=float, default=120.0, help="Per-bag playback wall-time limit")
    parser.add_argument("--max-frames", type=int, default=1_000_000, help="Per-bag combined color/depth frame limit")
    parser.add_argument("--json-out", type=Path, help="Write compact summary JSON to a NEW file, or '-' for stdout")
    args = parser.parse_args(argv)
    if args.max_seconds <= 0 or args.max_frames <= 0:
        parser.error("Playback limits must be positive")
    try:
        report = audit_episode(args.episode, alignment_report=args.alignment_report,
                               reference_camera=args.reference_camera,
                               max_seconds=args.max_seconds, max_frames=args.max_frames)
    except ValueError as error:
        parser.error(str(error))
    if args.json_out is not None:
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if str(args.json_out) == "-":
            print(encoded, end="")
        else:
            with args.json_out.open("x") as output:
                output.write(encoded)
            print(_console(report))
            print(f"  JSON summary: {args.json_out}")
    else:
        print(_console(report))
    # Explicit partial/error reads are failures; timing values have no arbitrary pass/fail threshold.
    return int(any(not camera["read"]["complete"] for camera in report["cameras"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
