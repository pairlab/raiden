#!/usr/bin/env python3
"""Capture lossless color-target evidence without setting camera controls or moving robots.

Uses only configured RealSense devices; requires existing locked controls to match
camera.json and records every supported ISP readback. No auto exposure/WB changes,
USB resets, firmware writes, CAN, or motor imports. Run with a stationary target and
arm. Each output must be a new directory. Sensor timing is retained, not calibrated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from raiden.cameras.realsense_receipt import color_frame_receipt, device_color_receipt


def require_locked(receipt, expected):
    """Refuse a capture whose actual controls differ; never repair them implicitly."""
    options = receipt["color_options"]
    for name, value in {
        **expected,
        "enable_auto_exposure": 0,
        "enable_auto_white_balance": 0,
    }.items():
        actual = options.get(name, {}).get("value")
        if actual is None or not np.isclose(actual, value, rtol=0, atol=0.5):
            raise ValueError(
                f"Existing {name}={actual} differs from required {value}; no control was changed"
            )


def capture_camera(name, entry, out, frames, warmup):
    result = {
        "schema": "raiden.radiometric_camera_capture/1",
        "camera": name,
        "expected_sdk_serial": str(entry["serial"]),
        "policy_crop": entry.get("crop"),
        "frames": [],
        "settings_written": False,
    }
    destination = out / name
    destination.mkdir()
    pipeline, started = None, False
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        width, height = entry.get("resolution", [640, 480])
        config.enable_device(str(entry["serial"]))
        config.enable_stream(
            rs.stream.color, width, height, rs.format.bgr8, entry.get("fps", 30)
        )
        profile = pipeline.start(config)
        started = True
        before = device_color_receipt(profile.get_device())
        result["before"] = before
        if before["device"].get("serial_number") != str(entry["serial"]):
            raise ValueError(
                "SDK serial mismatch; USB ASIC serial is a separate identifier"
            )
        require_locked(before, entry["color_controls"])
        for _ in range(warmup):
            pipeline.wait_for_frames(timeout_ms=5000)
        for i in range(frames):
            wait_before = time.time_ns()
            wait_before_monotonic = time.monotonic_ns()
            frame = pipeline.wait_for_frames(timeout_ms=5000).get_color_frame()
            wait_after_monotonic = time.monotonic_ns()
            wait_after = time.time_ns()
            if not frame:
                raise RuntimeError("No color frame")
            pixels = np.asanyarray(frame.get_data())
            image = destination / f"{i:04d}.png"
            if not cv2.imwrite(str(image), pixels):
                raise OSError(f"Could not write {image}")
            result["frames"].append(
                {
                    "image": str(image.relative_to(out)),
                    "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                    "host_wait_before_ns": wait_before,
                    "host_wait_after_ns": wait_after,
                    "host_wait_before_monotonic_ns": wait_before_monotonic,
                    "host_wait_after_monotonic_ns": wait_after_monotonic,
                    **color_frame_receipt(frame),
                }
            )
        after = device_color_receipt(profile.get_device())
        result["after"] = after
        require_locked(after, entry["color_controls"])
        intr = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        result["native_intrinsics"] = {
            "width": intr.width,
            "height": intr.height,
            "fx": intr.fx,
            "fy": intr.fy,
            "cx": intr.ppx,
            "cy": intr.ppy,
            "distortion_model": str(intr.model),
            "distortion_coefficients": intr.coeffs,
        }
        result["status"] = "complete"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            if started:
                pipeline.stop()
        except Exception as error:
            result.update(
                status="failed", stop_error=f"{type(error).__name__}: {error}"
            )
            raise
        finally:
            (destination / "receipt.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path.home() / ".config/raiden/camera.json"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--label",
        required=True,
        help="Target identity, location, normal, wrist pose and repeat",
    )
    parser.add_argument(
        "--cameras", nargs="+", default=["scene_camera", "left_wrist_camera"]
    )
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=15)
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 0:
        parser.error("frames must be positive and warmup nonnegative")
    config = json.loads(args.config.read_text())
    entries = {name: config[name] for name in args.cameras}
    for name, entry in entries.items():
        if entry.get("type") != "realsense" or not entry.get("color_controls"):
            parser.error(f"{name} needs a configured locked RealSense")
    # Enumerate only SDK serials explicitly named in the capture configuration.
    devices = {
        d.get_info(rs.camera_info.serial_number): d
        for d in rs.context().query_devices()
    }
    preflight = {}
    for name, entry in entries.items():
        device = devices.get(str(entry["serial"]))
        if device is None:
            raise RuntimeError(
                f"Configured camera {name}/{entry['serial']} not present"
            )
        preflight[name] = device_color_receipt(device)
        require_locked(preflight[name], entry["color_controls"])
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "raiden.radiometric_target_capture/1",
        "label": args.label,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "preflight": preflight,
        "camera_settings_written": False,
        "motion_commands_sent": False,
        "raw_sensor_output": False,
        "color_encoding": "native ISP-processed BGR8 saved losslessly as PNG; response unknown",
        "timing": "Independent rolling-shutter cameras; host handoff and SDK time retained separately",
        "target_reflectance_and_illumination": "Record measured target identity/reflectances and lighting separately; not inferred from RGB",
    }
    try:
        with ThreadPoolExecutor(max_workers=len(entries)) as executor:
            futures = {
                name: executor.submit(
                    capture_camera, name, entry, args.out, args.frames, args.warmup
                )
                for name, entry in entries.items()
            }
            manifest["cameras"] = {
                name: task.result() for name, task in futures.items()
            }
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (args.out / "capture.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.out / "capture.json")


if __name__ == "__main__":
    main()
