"""Verify camera prerequisites before pose recording can initialize robot hardware."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest


@pytest.fixture
def recorder_module():
    source = Path(__file__).parents[1] / "raiden/calibration/recorder.py"
    spec = importlib.util.spec_from_file_location("_calibration_recorder_test", source)
    module = importlib.util.module_from_spec(spec)
    stubs = {
        spec.name: module,
        "raiden.control": SimpleNamespace(TeleopInterface=object),
        "raiden.robot.controller": SimpleNamespace(RobotController=Mock(
            side_effect=AssertionError("robot setup before camera prerequisites")
        )),
    }
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
        yield module


def make_recorder(module, tmp_path, cameras):
    config = tmp_path / "camera.json"
    config.write_text(json.dumps(cameras))
    recorder = module.CalibrationPoseRecorder(
        interface=Mock(),
        output_file=str(tmp_path / "poses.json"),
        camera_config_file=str(config),
    )
    recorder._start_rerun_stream = Mock()
    return recorder


def test_empty_camera_config_prevents_robot_and_interface_start(recorder_module, tmp_path):
    module = recorder_module
    recorder = make_recorder(module, tmp_path, {})
    with patch.object(module.CameraConfig, "create_camera") as create_camera:
        with pytest.raises(ValueError, match="No cameras configured"):
            recorder.run_interactive_recording()
    create_camera.assert_not_called()
    recorder._start_rerun_stream.assert_not_called()
    module.RobotController.assert_not_called()
    recorder.interface.open.assert_not_called()
    recorder.interface.setup.assert_not_called()
    recorder.interface.start.assert_not_called()


@pytest.mark.parametrize("failure", ["open", "missing_color_sensor", "warmup_false", "warmup_exception"])
def test_camera_failure_closes_all_cameras_before_robot_start(
    recorder_module, tmp_path, failure
):
    module = recorder_module
    recorder = make_recorder(module, tmp_path, {"left": 123, "right": 456})
    cameras = [Mock(), Mock()]
    if failure == "open":
        cameras[1].open.side_effect = OSError("camera disconnected")
    elif failure == "missing_color_sensor":
        cameras[1].open.side_effect = RuntimeError("No RealSense sensor provides a color stream")
    elif failure == "warmup_false":
        cameras[1].grab.return_value = False
    else:
        cameras[1].grab.side_effect = OSError("camera disconnected")
    # Cleanup must still close the remaining camera if one close itself fails.
    cameras[0].close.side_effect = OSError("cleanup failed")
    with patch.object(module.CameraConfig, "create_camera", side_effect=cameras):
        with pytest.raises((OSError, RuntimeError), match="disconnected|color stream|warmup"):
            recorder.run_interactive_recording()
    for camera in cameras:
        camera.close.assert_called_once()
    assert recorder.cameras == {}
    recorder._start_rerun_stream.assert_not_called()
    module.RobotController.assert_not_called()
    recorder.interface.open.assert_not_called()
    recorder.interface.setup.assert_not_called()
    recorder.interface.start.assert_not_called()


@pytest.mark.parametrize("entry", [123, {"serial": 123, "role": "scene"},
                                  {"serial": 123, "role": "left_wrist"}])
def test_legacy_scene_and_single_wrist_configs_can_reach_robot_setup(
    recorder_module, tmp_path, entry
):
    module = recorder_module
    recorder = make_recorder(module, tmp_path, {"camera": entry})
    camera = Mock()
    module.RobotController.side_effect = RuntimeError("robot setup reached")
    with patch.object(module.CameraConfig, "create_camera", return_value=camera):
        with pytest.raises(RuntimeError, match="robot setup reached"):
            recorder.run_interactive_recording()
    camera.open.assert_called_once()
    assert camera.grab.call_count == 10
    module.RobotController.assert_called_once()
    camera.close.assert_called_once()
    recorder.interface.close.assert_called_once()
    assert recorder.cameras == {}


@pytest.mark.parametrize("phase", ["robot_setup", "interface_setup", "interface_start", "interface_open"])
def test_startup_failure_stops_and_closes_partial_controller_without_homing(
    recorder_module, tmp_path, phase
):
    module = recorder_module
    recorder = make_recorder(module, tmp_path, {"left": {"serial": 123, "role": "left_wrist"}})
    camera = Mock()
    controller = Mock()
    module.RobotController.side_effect = None
    module.RobotController.return_value = controller
    fail = {
        "robot_setup": controller.setup_for_teleop_recording,
        "interface_setup": recorder.interface.setup,
        "interface_start": recorder.interface.start,
        "interface_open": recorder.interface.open,
    }[phase]
    fail.side_effect = RuntimeError("startup failed")
    with patch.object(module.CameraConfig, "create_camera", return_value=camera):
        with pytest.raises(RuntimeError, match="startup failed"):
            recorder.run_interactive_recording()
    controller.stop_teleoperation.assert_called_once()
    controller.close.assert_called_once()
    controller.shutdown.assert_not_called()
    controller.cleanup.assert_not_called()
    recorder.interface.close.assert_called_once()
    camera.close.assert_called_once()


@pytest.mark.parametrize("phase", ["grpc", "web_viewer", "thread"])
def test_preview_startup_failure_disconnects_and_closes_all_cameras_before_robots(
    recorder_module, tmp_path, phase
):
    module = recorder_module
    recorder = make_recorder(module, tmp_path, {"left": 123, "right": 456})
    del recorder._start_rerun_stream
    cameras = [Mock(), Mock()]
    cameras[0].close.side_effect = OSError("cleanup failed")
    with patch.object(module.CameraConfig, "create_camera", side_effect=cameras), patch.multiple(
        module.rr, init=Mock(), serve_grpc=Mock(return_value="rerun+http://localhost:9878/proxy"),
        serve_web_viewer=Mock(), disconnect=Mock(),
    ), patch.object(module.threading, "Thread") as thread_type:
        fail = {
            "grpc": module.rr.serve_grpc,
            "web_viewer": module.rr.serve_web_viewer,
            "thread": thread_type.return_value.start,
        }[phase]
        fail.side_effect = RuntimeError("preview startup failed")
        with pytest.raises(RuntimeError, match="preview startup failed"):
            recorder.run_interactive_recording()
        module.rr.disconnect.assert_called_once()
    for camera in cameras:
        camera.close.assert_called_once()
    assert recorder.cameras == {}
    module.RobotController.assert_not_called()
    recorder.interface.open.assert_not_called()


def test_preview_opens_connected_viewer_and_shows_each_camera(
    recorder_module, tmp_path, capsys
):
    module = recorder_module
    recorder = make_recorder(module, tmp_path, {})
    del recorder._start_rerun_stream
    color = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    recorder.cameras = {
        name: Mock(get_frame=Mock(return_value=SimpleNamespace(color=color)))
        for name in ("left", "right", "scene")
    }
    cameras = list(recorder.cameras.values())
    recorder._rerun_stop_event = Mock()
    recorder._rerun_stop_event.is_set.return_value = False
    recorder._rerun_stop_event.wait.side_effect = (
        lambda _: setattr(recorder._rerun_stop_event.is_set, "return_value", True)
    )
    server_uri = "rerun+http://localhost:9878/proxy"
    image_type = module.rr.Image
    with patch.multiple(
        module.rr,
        init=Mock(),
        serve_grpc=Mock(return_value=server_uri),
        serve_web_viewer=Mock(),
        disconnect=Mock(),
        set_time=Mock(),
        log=Mock(),
        Image=Mock(wraps=image_type),
    ), patch.object(module.threading, "Thread"):
        recorder._start_rerun_stream()
        recorder._rerun_stream()
        module.rr.serve_web_viewer.assert_called_once_with(
            web_port=9877, open_browser=True, connect_to=server_uri
        )
        blueprint = module.rr.init.call_args.kwargs["default_blueprint"]
        assert [view.origin for view in blueprint.root_container.contents] == [
            "/cameras/left", "/cameras/right", "/cameras/scene"
        ]
        assert [call.args[0] for call in module.rr.log.call_args_list] == [
            "cameras/left", "cameras/right", "cameras/scene"
        ]
        images = module.rr.Image.call_args_list
        np.testing.assert_array_equal(images[0].args[0], color)
        np.testing.assert_array_equal(images[1].args[0], color)
        np.testing.assert_array_equal(images[2].args[0], color)
        recorder._stop_rerun_stream()
        module.rr.disconnect.assert_called_once()
        assert recorder.cameras == {}
        for camera in cameras:
            camera.close.assert_called_once()
    recorder._rerun_stop_event.wait.assert_called_once_with(0.1)
    assert (
        "http://localhost:9877?url=rerun%2Bhttp%3A%2F%2Flocalhost%3A9878%2Fproxy"
        in capsys.readouterr().out
    )
    module.RobotController.assert_not_called()
    recorder.interface.open.assert_not_called()


def test_preview_stop_waits_for_inflight_grab_before_closing_camera(
    recorder_module, tmp_path
):
    module = recorder_module
    recorder = make_recorder(module, tmp_path, {})
    in_grab = module.threading.Event()
    finish_grab = module.threading.Event()
    camera_closed = module.threading.Event()
    closed_during_grab = []

    def grab():
        in_grab.set()
        finish_grab.wait(timeout=5)
        in_grab.clear()
        return False

    def close():
        closed_during_grab.append(in_grab.is_set())
        camera_closed.set()

    camera = Mock(grab=Mock(side_effect=grab), close=Mock(side_effect=close))
    recorder.cameras = {"camera": camera}
    recorder._rerun_thread = module.threading.Thread(target=recorder._rerun_stream)
    stop_thread = module.threading.Thread(target=recorder._stop_rerun_stream)
    recorder._rerun_thread.start()
    try:
        assert in_grab.wait(timeout=1)
        stop_thread.start()
        assert recorder._rerun_stop_event.wait(timeout=1)
        # A timed join must not allow teardown to race a slow SDK call.
        assert not camera_closed.wait(timeout=2.1)
    finally:
        finish_grab.set()
        recorder._rerun_thread.join(timeout=2)
        if stop_thread.ident is not None:
            stop_thread.join(timeout=2)
    assert not stop_thread.is_alive()
    assert not recorder._rerun_thread.is_alive()
    assert closed_during_grab == [False]
    camera.grab.assert_called_once()
    assert recorder.cameras == {}
