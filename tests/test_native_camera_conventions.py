"""Metric depth and native optical frames agree in conversion and serving."""

import importlib.util
import pickle
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from raiden.cameras.realsense import RealSenseCamera
from raiden.converter import _build_lowdim, _extract_bag


def camera_frame(units=0.0001, *, color_only=False):
    color = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    raw = np.array([[0, 1000, 2000, 3000]] * 3, dtype=np.uint16)
    depth = None if color_only else SimpleNamespace(
        get_data=lambda: raw, get_units=lambda: units,
    )
    frames = SimpleNamespace(
        get_color_frame=lambda: SimpleNamespace(get_data=lambda: color, get_timestamp=lambda: 1234),
        get_depth_frame=lambda: depth,
    )
    camera = RealSenseCamera("right_wrist_camera", "offline")
    camera._latest_frames = frames
    camera._align = SimpleNamespace(process=lambda value: value)
    return camera, color, raw


@pytest.mark.parametrize("units", [0.0001, 0.001])
def test_metric_depth_uses_actual_aligned_frame_units(units):
    camera, color, raw = camera_frame(units)
    # A cached/sensor-level D435 default must not override D405 frame units.
    camera._depth_scale = 0.001
    frame = camera.get_frame()
    np.testing.assert_array_equal(frame.color, color)
    np.testing.assert_allclose(frame.depth, raw * units, rtol=1e-7)
    assert frame.depth.dtype == np.float32
    assert frame.timestamp_ns == 1_234_000_000


@pytest.mark.parametrize("units", [0, -0.001, np.nan, np.inf])
def test_invalid_depth_units_fail_instead_of_exporting_wrong_geometry(units):
    camera, _, _ = camera_frame(units)
    with pytest.raises(RuntimeError, match="invalid depth frame units"):
        camera.get_frame()


def test_color_only_and_crop_keep_native_orientation():
    camera, color, _ = camera_frame(color_only=True)
    camera._crop = (1, 0, 2, 2)
    frame = camera.get_frame()
    np.testing.assert_array_equal(frame.color, color[:2, 1:3])
    assert frame.depth is None


def test_native_right_wrist_pixels_and_millimetre_depth_are_exported(tmp_path, monkeypatch):
    camera, color, raw = camera_frame()
    camera.grab = Mock(side_effect=[True, False])
    camera.close = Mock()
    camera.get_camera_info = Mock(return_value={"width": 4, "height": 3})
    monkeypatch.setattr(RealSenseCamera, "from_bag", lambda *args, **kwargs: camera)
    rgb, depth = tmp_path / "rgb", tmp_path / "depth"
    _extract_bag(tmp_path / "right_wrist_camera.db3", rgb, depth)
    np.testing.assert_array_equal(cv2.imread(str(rgb / "0000000000.png")), color)
    np.testing.assert_array_equal(np.load(depth / "0000000000.npz")["depth"], raw / 10)
    camera.close.assert_called_once()


def server_class():
    spec = importlib.util.spec_from_file_location(
        "_native_camera_server", Path(__file__).parents[1] / "raiden/server.py",
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {
        "chiral": SimpleNamespace(PolicyServer=object, CameraConfig=object, ProprioConfig=object),
        "chiral.types": SimpleNamespace(CameraInfo=object, Observation=object),
        "raiden.robot.controller": SimpleNamespace(RobotController=object, _GRIPPER_MIN_OPENING=20 / 95),
    }):
        spec.loader.exec_module(module)
    return module.RaidenPolicyServer


def test_converter_and_server_use_same_native_right_wrist_intrinsics_and_hand_eye(tmp_path):
    name = "right_wrist_camera"
    k = np.array([[320, 0, 1.25], [0, 321, 0.75], [0, 0, 1.]])
    he = np.eye(4)
    he[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    he[:3, 3] = [0.01, 0.02, 0.03]
    base = np.eye(4)
    base[0, 3] = 0.6
    calibration = {
        "bimanual_transform": {"T_left_from_right": base.tolist()},
        "cameras": {name: {"hand_eye_calibration": {
            "success": True, "rotation_matrix": he[:3, :3].tolist(),
            "translation_vector": he[:3, 3].tolist(),
        }}},
    }
    cls = server_class()
    server = cls.__new__(cls)
    server._calibration = calibration
    server._raiden_cam_cfg = SimpleNamespace(list_camera_names=lambda: [name])
    server._cam_handles = {name: {"intrinsics": k, "w": 4, "h": 3}}
    server._resize = None
    server._cam_intrinsics, server._cam_extrinsics, server._T_cam2ee = {}, {}, {}
    server._prepare_camera_transforms()
    np.testing.assert_array_equal(server._cam_intrinsics[name], k)
    np.testing.assert_array_equal(server._T_cam2ee[name], he)

    data = {"timestamps": np.array([1_800_000_000_000_000_000], dtype=np.int64)}
    data["follower_r_joint_pos"] = np.zeros((1, 6))
    data["follower_r_joint_pos_7d"] = np.zeros((1, 7))
    data["follower_r_joint_cmd"] = np.zeros((1, 7))
    camera_info = {name: {"fx": 320, "fy": 321, "cx": 1.25, "cy": .75, "width": 4, "height": 3}}
    kin = SimpleNamespace(fk=lambda q: np.eye(4), _configuration=SimpleNamespace(model=SimpleNamespace(nq=7)))
    with patch("raiden.converter._get_kinematics", return_value=kin):
        _build_lowdim(
            tmp_path, [name], 1, camera_info, calibration, data, {}, base,
            {name: data["timestamps"]}, wrist_camera_joint_keys={name: "follower_r_joint_pos"},
        )
    lowdim = pickle.loads((tmp_path / "lowdim/0000000000.pkl").read_bytes())
    np.testing.assert_array_equal(lowdim["intrinsics"][name], k)
    np.testing.assert_allclose(lowdim["extrinsics"][name], base @ he, atol=1e-7)


def test_right_wrist_serving_keeps_native_pixels_depth_and_timestamp():
    cls = server_class()
    server = cls.__new__(cls)
    camera, color, raw = camera_frame()
    name = "right_wrist_camera"
    server._running = True
    server._resize = None
    server.depths = {name: None}
    server._cam_ts_locks = {name: threading.Lock()}
    server._cam_capture_ts_ns = {}
    images, depths = [], []

    def image_received(camera_name, image):
        assert camera_name == name
        images.append(image.copy())
        server._running = False

    server.update_image = image_received
    server.update_depth = lambda camera_name, depth: depths.append(depth.copy())
    camera.grab = lambda: True
    server._realsense_capture_loop(name, {"camera": camera})
    np.testing.assert_array_equal(images[0], color[..., ::-1])
    np.testing.assert_allclose(depths[0], raw * .0001, rtol=1e-7)
    assert server._cam_capture_ts_ns[name] == 1_234_000_000
