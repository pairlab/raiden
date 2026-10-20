"""Receipt unit checks use getters-only fakes and never open hardware."""

import json
import unittest
from types import SimpleNamespace

import pyrealsense2 as rs

from raiden.cameras.realsense_receipt import (
    COLOR_OPTIONS,
    color_frame_receipt,
    device_color_receipt,
)


class Sensor:
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

    def first_color_sensor(self):
        return Sensor()


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
        device.first_color_sensor = lambda: (_ for _ in ()).throw(
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


if __name__ == "__main__":
    unittest.main()
