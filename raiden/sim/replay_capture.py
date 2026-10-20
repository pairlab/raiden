"""Two-camera replay evidence. Importing this module opens no devices.

Camera workers read existing ISP controls and refuse mismatches. Native color
bags retain image data; MP4s are previews, with independent frame-clock JSONL.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np

from raiden.sim.hardware_session import write_json

CAMERAS = ("scene_camera", "left_wrist_camera")


def validate_camera_config(config):
    entries = {name: config[name] for name in CAMERAS}
    if len({str(entry["serial"]) for entry in entries.values()}) != len(CAMERAS):
        raise ValueError("scene and wrist must name different SDK serials")
    for name, entry in entries.items():
        if entry.get("type") != "realsense" or not entry.get("color_controls"):
            raise ValueError(
                f"{name} needs an existing locked RealSense color configuration"
            )
        width, height = entry.get("resolution", [640, 480])
        if width <= 0 or height <= 0 or entry.get("fps", 30) <= 0:
            raise ValueError(f"invalid camera dimensions/fps: {name}")
    return entries


def require_locked(receipt, expected):
    for name, value in {
        **expected,
        "enable_auto_exposure": 0,
        "enable_auto_white_balance": 0,
    }.items():
        actual = receipt["color_options"].get(name, {}).get("value")
        if actual is None or not np.isclose(actual, value, rtol=0, atol=0.5):
            raise ValueError(
                f"Existing {name}={actual} differs from required {value}; no control was changed"
            )


class DualCameraCapture:
    """Capture both cameras and expose nonempty, recent-frame readiness."""

    def __init__(
        self,
        config,
        destination,
        *,
        warmup=15,
        maximum_age_ns=500_000_000,
        clock=time,
        worker=None,
    ):
        self.entries = validate_camera_config(config)
        self.destination = Path(destination)
        self.warmup, self.maximum_age_ns, self.clock = warmup, maximum_age_ns, clock
        self.worker = worker or self._capture
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.status = {name: {"status": "starting", "frames": 0} for name in CAMERAS}
        self.threads = []
        self.unfinished_workers = set()

    def start(self, timeout=20.0, stop_requested=lambda: False):
        self.destination.mkdir(parents=True, exist_ok=False)
        write_json(self.destination / "config.json", self.entries)
        for name, entry in self.entries.items():
            thread = threading.Thread(
                target=self.worker,
                args=(name, entry),
                daemon=True,
                name=f"replay-{name}",
            )
            self.threads.append(thread)
            thread.start()
        deadline = self.clock.monotonic() + timeout
        while True:
            if stop_requested():
                raise InterruptedError("operator stopped camera startup")
            snapshot = self.snapshot()
            if any(row["status"] == "failed" for row in snapshot.values()):
                raise RuntimeError(f"camera startup failed: {snapshot}")
            if all(
                row["status"] == "recording" and row["frames"] > 0
                for row in snapshot.values()
            ):
                self.check()
                write_json(
                    self.destination / "ready.json",
                    {
                        "host_wall_ns": self.clock.time_ns(),
                        "host_monotonic_ns": self.clock.monotonic_ns(),
                        "cameras": snapshot,
                    },
                )
                return
            if self.clock.monotonic() >= deadline:
                raise TimeoutError(
                    f"both cameras must deliver recorded frames before motion: {snapshot}"
                )
            self.clock.sleep(0.02)

    def snapshot(self):
        with self.lock:
            return {name: dict(row) for name, row in self.status.items()}

    def check(self):
        now = self.clock.monotonic_ns()
        for name, row in self.snapshot().items():
            if (
                row["status"] != "recording"
                or row["frames"] < 1
                or now - row.get("last_frame_monotonic_ns", 0) > self.maximum_age_ns
            ):
                raise RuntimeError(f"{name} capture failed, empty or stale: {row}")

    def stop(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=3.0)
        for name, thread in zip(CAMERAS, self.threads):
            if thread.is_alive():
                with self.lock:
                    self.unfinished_workers.add(name)
                    self.status[name].update(
                        status="failed",
                        stop_error="camera worker did not stop within 3 seconds",
                    )
        result = self.snapshot()
        write_json(
            self.destination / "capture.json",
            {
                "cameras": result,
                "camera_settings_written": False,
                "timing": "SDK and host wall/monotonic clocks retained independently; rolling shutter/exposure timing uncalibrated",
            },
        )
        if any(
            row["status"] != "complete" or not row["frames"] for row in result.values()
        ):
            raise RuntimeError(
                f"camera capture incomplete; retain both failed receipts: {result}"
            )
        return result

    def _capture(self, name, entry):
        # These optional/device modules are deliberately absent from offline preflight.
        pipeline, writer, started = None, None, False
        folder = self.destination / name
        record = {
            "camera": name,
            "sdk_serial": str(entry["serial"]),
            "frames": 0,
            "camera_settings_written": False,
            "native_bag": "color.bag",
            "preview": "preview.mp4",
            "policy_crop": entry.get("crop"),
            "depth_recorded": False,
            "status": "starting",
        }
        frames = None
        try:
            folder.mkdir()
            import cv2
            import pyrealsense2 as rs

            from raiden.cameras.realsense_receipt import (
                color_frame_receipt,
                device_color_receipt,
            )

            pipeline = rs.pipeline()
            cfg = rs.config()
            width, height = entry.get("resolution", [640, 480])
            fps = entry.get("fps", 30)
            cfg.enable_device(str(entry["serial"]))
            cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            cfg.enable_record_to_file(str(folder / "color.bag"))
            profile = pipeline.start(cfg)
            started = True
            record["before"] = device_color_receipt(profile.get_device())
            if record["before"]["device"].get("serial_number") != str(entry["serial"]):
                raise ValueError("SDK serial mismatch")
            require_locked(record["before"], entry["color_controls"])
            intr = (
                profile.get_stream(rs.stream.color)
                .as_video_stream_profile()
                .get_intrinsics()
            )
            record["native_intrinsics"] = {
                "width": intr.width,
                "height": intr.height,
                "fx": intr.fx,
                "fy": intr.fy,
                "cx": intr.ppx,
                "cy": intr.ppy,
                "distortion_model": str(intr.model),
                "distortion_coefficients": intr.coeffs,
            }
            writer = cv2.VideoWriter(
                str(folder / "preview.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError("MP4 preview writer failed to open")
            frames = open(folder / "frames.jsonl", "w", buffering=1)
            previous = None
            for index in range(2**63):
                if self.stop_event.is_set():
                    break
                before_wall, before_mono = (
                    self.clock.time_ns(),
                    self.clock.monotonic_ns(),
                )
                frame = pipeline.wait_for_frames(timeout_ms=1000).get_color_frame()
                after_mono, after_wall = self.clock.monotonic_ns(), self.clock.time_ns()
                if not frame:
                    raise RuntimeError("No color frame")
                if index < self.warmup:
                    continue
                data = np.asanyarray(frame.get_data())
                if data.shape != (height, width, 3):
                    raise ValueError(f"unexpected native color shape {data.shape}")
                receipt = color_frame_receipt(frame)
                number = int(frame.get_frame_number())
                if previous is not None and number <= previous:
                    raise ValueError(
                        f"camera frame counter stopped or reversed: {previous} -> {number}"
                    )
                writer.write(data)
                if record["frames"] == 0 and not cv2.imwrite(
                    str(folder / "first.png"), data
                ):
                    raise OSError("Could not save first camera frame")
                frames.write(
                    json.dumps(
                        {
                            "index": record["frames"],
                            "host_wait_before_wall_ns": before_wall,
                            "host_wait_after_wall_ns": after_wall,
                            "host_wait_before_monotonic_ns": before_mono,
                            "host_wait_after_monotonic_ns": after_mono,
                            **receipt,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                record["frames"] += 1
                record["frame_counter_gaps"] = record.get("frame_counter_gaps", 0) + (
                    max(0, number - previous - 1) if previous else 0
                )
                record.update(status="recording", last_frame_monotonic_ns=after_mono)
                previous = number
                with self.lock:
                    self.status[name] = dict(record)
            record["after"] = device_color_receipt(profile.get_device())
            require_locked(record["after"], entry["color_controls"])
            record["status"] = "complete" if record["frames"] else "failed"
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
        finally:
            for label, close in (
                ("frames", frames.close if frames else None),
                ("preview", writer.release if writer else None),
                ("pipeline", pipeline.stop if started else None),
            ):
                if close:
                    try:
                        close()
                    except BaseException as error:
                        record.update(
                            status="failed",
                            **{
                                f"{label}_stop_error": f"{type(error).__name__}: {error}"
                            },
                        )
            with self.lock:
                if name in self.unfinished_workers:
                    record.update(
                        status="failed",
                        stop_error="camera worker exceeded stop deadline; native bag finalization was unverified",
                    )
                self.status[name] = dict(record)
            # The aggregate receipt also survives a failure creating this folder.
            if folder.is_dir():
                write_json(folder / "receipt.json", record)
