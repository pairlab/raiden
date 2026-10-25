"""Receipt unit checks use getters-only fakes and never open hardware."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pyrealsense2 as rs

from raiden.cameras.realsense_receipt import (
    COLOR_OPTIONS,
    color_frame_receipt,
    device_color_receipt,
)
from raiden.cameras.realsense import RealSenseCamera


class Sensor:
    def get_stream_profiles(self):
        return [SimpleNamespace(stream_type=lambda: rs.stream.color)]

    def supports(self, key):
        return key in (rs.option.exposure, rs.option.gain, rs.option.gamma)

    def get_option(self, key):
        return {
            rs.option.exposure: 250.0,
            rs.option.gain: 22.0,
            rs.option.gamma: 300.0,
        }[key]

    def get_option_description(self, key):
        return "native UVC control"

    def get_option_range(self, key):
        return SimpleNamespace(min=1.0, max=10000.0, step=1.0, default=166.0)

    def set_option(self, *args):
        raise AssertionError("receipt must never write device settings")


class Device:
    def is_playback(self):
        return False

    def supports(self, key):
        return key in (rs.camera_info.name, rs.camera_info.serial_number)

    def get_info(self, key):
        return {
            rs.camera_info.name: "Intel RealSense D435I",
            rs.camera_info.serial_number: "test",
        }[key]

    def query_sensors(self):
        return [Sensor()]


class ReceiptTest(unittest.TestCase):
    def test_native_values_are_retained_without_inventing_gamma_or_iso(self):
        receipt = device_color_receipt(Device())
        json.dumps(receipt, allow_nan=False)
        self.assertEqual(receipt["color_options"]["gamma"]["value"], 300.0)
        self.assertEqual(receipt["color_options"]["gain"]["value"], 22.0)
        self.assertEqual(receipt["rgb_exposure_seconds_from_uvc_control"], 0.025)
        self.assertEqual(set(receipt["color_options"]), set(COLOR_OPTIONS))
        self.assertFalse(receipt["color_options"]["white_balance"]["supported"])

    def test_playback_ranges_are_labeled_as_frozen(self):
        device = Device()
        device.is_playback = lambda: True
        self.assertIn("frozen", device_color_receipt(device)["option_range_scope"])

    def test_other_models_do_not_inherit_d435_exposure_units(self):
        device = Device()
        device.get_info = lambda key: "other camera"
        self.assertNotIn(
            "rgb_exposure_seconds_from_uvc_control", device_color_receipt(device)
        )

    def test_readback_failure_remains_visible(self):
        device = Device()
        device.query_sensors = lambda: (_ for _ in ()).throw(
            RuntimeError("sensor unavailable")
        )
        self.assertIn(
            "sensor unavailable",
            device_color_receipt(device)["readback_errors"]["color_sensor"],
        )

    def test_frame_raw_exposure_is_not_silently_reinterpreted(self):
        frame = SimpleNamespace(
            get_frame_number=lambda: 4,
            get_timestamp=lambda: 12.5,
            get_frame_timestamp_domain=lambda: "hardware_clock",
            supports_frame_metadata=lambda key: (
                key == rs.frame_metadata_value.actual_exposure
            ),
            get_frame_metadata=lambda key: 250,
        )
        receipt = color_frame_receipt(frame)
        self.assertEqual(receipt["native_metadata"], {"actual_exposure": 250})
        self.assertIn("gamma", receipt["unavailable_fields"])

    def test_clock_mapping_records_receipt_provenance_without_exposure_claim(self):
        frame = SimpleNamespace(get_timestamp=lambda: 12.5,
                                get_frame_timestamp_domain=lambda: "hardware_clock")
        camera = RealSenseCamera.__new__(RealSenseCamera)
        camera._pipeline = SimpleNamespace(wait_for_frames=lambda timeout_ms:
                                          SimpleNamespace(get_color_frame=lambda: frame))
        with patch("raiden.cameras.realsense.time.time_ns", side_effect=[19_000_000, 20_000_000]):
            camera._measure_clock_offset()
        receipt = camera._clock_mapping_receipt
        self.assertEqual(camera._clock_offset_ns, 7_500_000)
        self.assertEqual(receipt["offset_ns"], 7_500_000)
        self.assertEqual(receipt["frame_timestamp_domain"], "hardware_clock")
        self.assertEqual(receipt["host_wait_start_ns"], 19_000_000)
        self.assertEqual(receipt["host_receive_ns"], 20_000_000)
        self.assertIn("unknown transport/queue delay", receipt["interpretation"])

    def test_failed_clock_measurement_cannot_reuse_previous_episode_offset(self):
        camera = RealSenseCamera.__new__(RealSenseCamera)
        camera._clock_offset_ns = 123
        camera._pipeline = SimpleNamespace(wait_for_frames=lambda timeout_ms:
                                          (_ for _ in ()).throw(RuntimeError("saved failure")))
        camera._measure_clock_offset()
        self.assertIsNone(camera._clock_offset_ns)
        self.assertIn("saved failure", camera._clock_mapping_receipt["error"])

    def test_native_recording_format_is_explicit_and_validated_without_hardware(self):
        with patch("raiden.cameras.realsense.rs.pipeline") as pipeline:
            self.assertEqual(RealSenseCamera("view", "synthetic").recording_extension, "bag")
            self.assertEqual(RealSenseCamera("view", "synthetic", recording_format="db3").recording_extension, "db3")
            for value in ("svo2", "", None, ".db3"):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "recording_format"):
                    RealSenseCamera("view", "synthetic", recording_format=value)
            pipeline.assert_not_called()

    def test_contradicting_output_path_fails_before_stopping_or_writing(self):
        from pathlib import Path

        camera = RealSenseCamera("view", "synthetic", recording_format="db3")
        camera._pipeline = SimpleNamespace(stop=lambda: self.fail("must not stop a stream"))
        with self.assertRaisesRegex(ValueError, "requires a .db3 output path"):
            camera.start_recording(Path("view.bag"))

    def test_playback_rejects_unrelated_suffix_before_creating_a_pipeline(self):
        from pathlib import Path

        with patch("raiden.cameras.realsense.rs.pipeline") as pipeline:
            with self.assertRaisesRegex(ValueError, ".bag or .db3"):
                RealSenseCamera.from_bag("view", Path("view.mp4"))
            pipeline.assert_not_called()

    def test_playback_keeps_both_native_formats_without_changing_sdk_settings(self):
        from pathlib import Path

        for suffix in ("bag", "db3"):
            with self.subTest(suffix=suffix), patch("raiden.cameras.realsense.rs.pipeline") as pipeline, \
                    patch("raiden.cameras.realsense.rs.config") as config:
                profile = pipeline.return_value.start.return_value
                profile.get_streams.return_value = []
                camera = RealSenseCamera.from_bag("view", Path(f"view.{suffix}"))
                self.assertEqual(camera.recording_extension, suffix)
                config.enable_device_from_file.assert_called_once_with(
                    config.return_value, f"view.{suffix}", repeat_playback=False)
                profile.get_device.return_value.as_playback.return_value.set_real_time.assert_called_once_with(False)
                camera.close()
                pipeline.return_value.stop.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
