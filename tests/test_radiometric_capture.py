"""Camera-only target captures retain provenance and failed lifecycle receipts."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

spec = importlib.util.spec_from_file_location(
    "radiometric_capture", Path(__file__).parents[1] / "scripts/radiometric_capture.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CaptureTest(unittest.TestCase):
    def capture(self, root, *, start_error=False, stop_error=False, exposure=250):
        pixels = np.full((12, 16, 3), [20, 80, 160], np.uint8)
        frame = SimpleNamespace(get_data=lambda: pixels)
        intr = SimpleNamespace(
            width=16,
            height=12,
            fx=20.0,
            fy=20.0,
            ppx=8.0,
            ppy=6.0,
            model="none",
            coeffs=[0.0] * 5,
        )
        stream = SimpleNamespace(get_intrinsics=lambda: intr)
        stream.as_video_stream_profile = lambda: stream
        profile = SimpleNamespace(
            get_device=lambda: object(), get_stream=lambda _: stream
        )
        self.stopped = False

        def start(_):
            if start_error:
                raise RuntimeError("camera busy")
            return profile

        def stop():
            self.stopped = True
            if stop_error:
                raise RuntimeError("stop failed")

        pipeline = SimpleNamespace(
            start=start,
            stop=stop,
            wait_for_frames=lambda **_: SimpleNamespace(get_color_frame=lambda: frame),
        )
        config = SimpleNamespace(
            enable_device=lambda *_: None, enable_stream=lambda *_: None
        )
        receipt = {
            "device": {"serial_number": "test"},
            "color_options": {
                "enable_auto_exposure": {"value": 0},
                "enable_auto_white_balance": {"value": 0},
                "exposure": {"value": exposure},
            },
        }
        entry = {
            "serial": "test",
            "resolution": [16, 12],
            "color_controls": {"exposure": 250},
        }
        with (
            patch.object(module.rs, "pipeline", return_value=pipeline),
            patch.object(module.rs, "config", return_value=config),
            patch.object(module, "device_color_receipt", return_value=receipt),
            patch.object(
                module,
                "color_frame_receipt",
                return_value={
                    "sdk_timestamp_ms": 1.0,
                    "timestamp_domain": "global_time",
                },
            ),
        ):
            return module.capture_camera("scene_camera", entry, root, 1, 0)

    def test_lossless_frame_hash_and_both_host_clock_anchors_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.capture(root)
            self.assertTrue(self.stopped)
            self.assertEqual(result["status"], "complete")
            self.assertFalse(result["settings_written"])
            frame = result["frames"][0]
            path = root / frame["image"]
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), frame["sha256"]
            )
            np.testing.assert_array_equal(cv2.imread(str(path))[0, 0], [20, 80, 160])
            for suffix in ("ns", "monotonic_ns"):
                self.assertLessEqual(
                    frame[f"host_wait_before_{suffix}"],
                    frame[f"host_wait_after_{suffix}"],
                )

    def test_busy_camera_leaves_failed_receipt_without_stopping_unstarted_pipeline(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "busy"):
                self.capture(root, start_error=True)
            result = json.loads((root / "scene_camera/receipt.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertFalse(self.stopped)
            self.assertEqual(result["frames"], [])

    def test_settings_mismatch_is_a_failed_capture_and_releases_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "no control was changed"):
                self.capture(root, exposure=200)
            self.assertTrue(self.stopped)
            self.assertEqual(
                json.loads((root / "scene_camera/receipt.json").read_text())["status"],
                "failed",
            )

    def test_stop_failure_still_writes_failed_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "stop failed"):
                self.capture(root, stop_error=True)
            result = json.loads((root / "scene_camera/receipt.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertIn("stop failed", result["stop_error"])


if __name__ == "__main__":
    unittest.main()
