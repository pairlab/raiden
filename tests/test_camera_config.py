"""Native recording format config is validated before any SDK camera is opened."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import test_recording_integrity as integrity

from raiden.camera_config import CameraConfig


def write_config(tmp_path, entries):
    path = tmp_path / "camera.json"
    path.write_text(json.dumps(entries))
    return path


@pytest.mark.parametrize("recording_format", ["bag", "db3"])
def test_explicit_realsense_format_reaches_factory_and_portable_snapshot(tmp_path, monkeypatch, recording_format):
    path = write_config(tmp_path, {
        "view": {"serial": "serial", "type": "realsense", "recording_format": recording_format}
    })
    camera = SimpleNamespace(open=Mock())
    factory = Mock(return_value=camera)
    monkeypatch.setitem(sys.modules, "raiden.cameras.realsense", SimpleNamespace(RealSenseCamera=factory))
    config = CameraConfig(str(path))
    assert config.get_recording_format("view") == recording_format
    assert config.create_camera("view") is camera
    assert factory.call_args.kwargs["recording_format"] == recording_format
    camera.open.assert_not_called()
    snapshot = integrity.recorder._camera_recording_config(config, ["view"])
    assert snapshot["view"]["recording_format"] == recording_format


@pytest.mark.parametrize("recording_format", ["mp4", "DB3", None, [], 1])
def test_invalid_format_is_rejected_before_native_factory(tmp_path, monkeypatch, recording_format):
    path = write_config(tmp_path, {
        "view": {"serial": "serial", "type": "realsense", "recording_format": recording_format}
    })
    factory = Mock()
    monkeypatch.setitem(sys.modules, "raiden.cameras.realsense", SimpleNamespace(RealSenseCamera=factory))
    with pytest.raises(ValueError, match="recording_format must be 'bag' or 'db3'"):
        CameraConfig(str(path))
    factory.assert_not_called()


def test_legacy_absence_and_zed_factory_do_not_receive_new_kwargs(tmp_path, monkeypatch):
    path = write_config(tmp_path, {
        "native": {"serial": "serial", "type": "realsense"}, "legacy": 1234,
    })
    native, zed = Mock(), Mock()
    monkeypatch.setitem(sys.modules, "raiden.cameras.realsense", SimpleNamespace(RealSenseCamera=native))
    monkeypatch.setitem(sys.modules, "raiden.cameras.zed", SimpleNamespace(ZedCamera=zed))
    config = CameraConfig(str(path))
    config.create_camera("native")
    config.create_camera("legacy")
    assert "recording_format" not in native.call_args.kwargs
    assert zed.call_args.kwargs == {"fps": 30}
    snapshot = integrity.recorder._camera_recording_config(config, ["native", "legacy"])
    assert all("recording_format" not in entry for entry in snapshot.values())


def test_non_realsense_recording_format_is_rejected_before_factory(tmp_path):
    path = write_config(tmp_path, {
        "zed": {"serial": 1234, "type": "zed", "recording_format": "db3"}
    })
    with pytest.raises(ValueError, match="supported only for RealSense"):
        CameraConfig(str(path))
