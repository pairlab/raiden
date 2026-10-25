"""Color controls use the actual producing sensor without opening hardware."""

from types import SimpleNamespace
from unittest.mock import Mock

import pyrealsense2 as rs
import pytest

from raiden.cameras.realsense import RealSenseCamera
from raiden.cameras.realsense_receipt import color_stream_sensor, device_color_receipt


class Sensor:
    def __init__(self, streams, *, depth=False, unsupported=()):
        self.streams = streams
        self.depth = depth
        self.unsupported = unsupported
        self.writes = []
        self.values = {
            rs.option.exposure: 33000 if depth else 166,
            rs.option.gain: 16,
            rs.option.white_balance: 4600,
        }

    def get_stream_profiles(self):
        return [SimpleNamespace(stream_type=lambda stream=stream: stream)
                for stream in self.streams]

    def is_depth_sensor(self):
        return self.depth

    def supports(self, option):
        return option not in self.unsupported and option in (
            *self.values, rs.option.enable_auto_exposure,
            rs.option.enable_auto_white_balance,
        )

    def set_option(self, option, value):
        assert self.supports(option)
        self.writes.append((option, value))
        self.values[option] = value

    def get_option(self, option):
        return self.values.get(option, 1)

    def get_option_description(self, option):
        return "native control"

    def get_option_range(self, option):
        return SimpleNamespace(min=0, max=165000, step=1, default=1)


def device_for(sensors, model="Intel RealSense D405"):
    return SimpleNamespace(
        query_sensors=lambda: sensors,
        first_color_sensor=Mock(side_effect=RuntimeError("Could not find requested sensor type!")),
        is_playback=lambda: False,
        supports=lambda option: option in (rs.camera_info.name, rs.camera_info.serial_number),
        get_info=lambda option: model if option == rs.camera_info.name else "serial",
    )


@pytest.mark.parametrize("depth_color", [True, False])
def test_d405_stereo_and_d435_rgb_select_color_producer_for_controls_and_receipts(depth_color):
    producer = Sensor([rs.stream.depth, rs.stream.color] if depth_color else [rs.stream.color],
                      depth=depth_color)
    depth_only = Sensor([rs.stream.depth, rs.stream.infrared], depth=True)
    device = device_for([producer] if depth_color else [depth_only, producer],
                        "Intel RealSense D405" if depth_color else "Intel RealSense D435")
    assert color_stream_sensor(device) is producer
    camera = RealSenseCamera("camera", "serial")
    camera._apply_color_controls(device)
    assert producer.writes == [(rs.option.enable_auto_exposure, 1),
                               (rs.option.enable_auto_white_balance, 1)]
    assert depth_only.writes == []
    receipt = device_color_receipt(device)
    assert receipt["color_options"]["exposure"]["value"] == (33000 if depth_color else 166)
    if depth_color:
        assert "rgb_exposure_seconds_from_uvc_control" not in receipt
    else:
        assert receipt["rgb_exposure_seconds_from_uvc_control"] == pytest.approx(0.0166)
    device.first_color_sensor.assert_not_called()


def test_auto_skips_unsupported_optional_white_balance_switch():
    sensor = Sensor([rs.stream.color], depth=True,
                    unsupported=(rs.option.enable_auto_white_balance,))
    RealSenseCamera("camera", "serial")._apply_color_controls(device_for([sensor]))
    assert sensor.writes == [(rs.option.enable_auto_exposure, 1)]


def test_unsupported_manual_control_fails_before_any_device_write():
    sensor = Sensor([rs.stream.color], depth=True, unsupported=(rs.option.white_balance,))
    camera = RealSenseCamera("camera", "serial", color_controls={
        "white_balance": 4600, "exposure": 1000, "gain": 16,
    })
    with pytest.raises(ValueError, match="requested controls: white_balance"):
        camera._apply_color_controls(device_for([sensor]))
    assert sensor.writes == []


@pytest.mark.parametrize("depth_color,exposure,valid", [
    (True, 33000, True), (True, 33400, False),
    (False, 333, True), (False, 334, False),
])
def test_frame_period_guard_uses_producing_sensor_exposure_units(depth_color, exposure, valid):
    sensor = Sensor([rs.stream.color], depth=depth_color)
    controls = {"white_balance": 4600, "exposure": exposure, "gain": 16}
    camera = RealSenseCamera("camera", "serial", fps=30, color_controls=controls)
    if valid:
        camera._apply_color_controls(device_for([sensor]))
        assert camera.color_controls == controls
    else:
        with pytest.raises(ValueError, match="longer than a frame at 30 fps"):
            camera._apply_color_controls(device_for([sensor]))
        assert sensor.writes == []


def test_missing_color_producer_raises_and_receipt_preserves_error():
    sensor = Sensor([rs.stream.depth], depth=True)
    device = device_for([sensor])
    with pytest.raises(RuntimeError, match="No RealSense sensor provides a color stream"):
        RealSenseCamera("camera", "serial")._apply_color_controls(device)
    assert sensor.writes == []
    assert "No RealSense sensor" in device_color_receipt(device)["readback_errors"]["color_sensor"]
