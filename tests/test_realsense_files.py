"""Native filename discovery and replay targets need no camera hardware."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from raiden.realsense_files import find_realsense_recording, list_realsense_recordings
from raiden.sim.replay_capture import CAMERAS, DualCameraCapture


def test_native_discovery_lists_both_formats_and_ignores_other_files(tmp_path):
    for name in ("scene.db3", "wrist.bag", "notes.txt", "zed.svo2"):
        (tmp_path / name).write_bytes(name.encode())
    (tmp_path / "directory.bag").mkdir()
    assert [path.name for path in list_realsense_recordings(tmp_path)] == ["scene.db3", "wrist.bag"]
    assert find_realsense_recording(tmp_path, "scene") == tmp_path / "scene.db3"
    assert find_realsense_recording(tmp_path, "wrist") == tmp_path / "wrist.bag"
    assert find_realsense_recording(tmp_path, "absent") is None


def test_missing_directory_preserves_empty_list_and_missing_camera(tmp_path):
    missing = tmp_path / "absent"
    assert list_realsense_recordings(missing) == []
    assert find_realsense_recording(missing, "scene") is None
    assert not missing.exists()


def test_duplicate_camera_formats_are_rejected_without_choosing_or_modifying_files(tmp_path):
    for suffix in ("bag", "db3"):
        (tmp_path / f"scene.{suffix}").write_bytes(suffix.encode())
    with pytest.raises(ValueError, match="Duplicate.*scene.*scene.bag, scene.db3"):
        list_realsense_recordings(tmp_path)
    with pytest.raises(ValueError, match="Duplicate"):
        find_realsense_recording(tmp_path, "scene")
    assert (tmp_path / "scene.bag").read_bytes() == b"bag"
    assert (tmp_path / "scene.db3").read_bytes() == b"db3"


@pytest.mark.parametrize("recording_format", [None, "bag", "db3"])
def test_replay_capture_writer_and_receipt_use_the_declared_filename(tmp_path, monkeypatch, recording_format):
    entries = {
        name: {"serial": str(index), "type": "realsense", "color_controls": {"gain": 22}}
        for index, name in enumerate(CAMERAS)
    }
    if recording_format is not None:
        for entry in entries.values():
            entry["recording_format"] = recording_format
    capture = DualCameraCapture(entries, tmp_path)
    config = SimpleNamespace(enable_device=Mock(), enable_stream=Mock(), enable_record_to_file=Mock())
    pipeline = SimpleNamespace(start=Mock(side_effect=RuntimeError("offline stop before SDK start")))
    sdk = SimpleNamespace(
        pipeline=lambda: pipeline, config=lambda: config,
        stream=SimpleNamespace(color="color"), format=SimpleNamespace(bgr8="bgr8"),
    )
    monkeypatch.setitem(sys.modules, "pyrealsense2", sdk)
    monkeypatch.setitem(sys.modules, "raiden.cameras.realsense_receipt", SimpleNamespace(
        color_frame_receipt=Mock(), device_color_receipt=Mock(),
    ))
    capture._capture(CAMERAS[0], entries[CAMERAS[0]])
    expected = f"color.{recording_format or 'bag'}"
    config.enable_record_to_file.assert_called_once_with(str(tmp_path / CAMERAS[0] / expected))
    receipt = json.loads((tmp_path / CAMERAS[0] / "receipt.json").read_text())
    assert receipt["native_bag"] == expected
    assert receipt["status"] == "failed"
    assert "offline stop" in receipt["error"]
