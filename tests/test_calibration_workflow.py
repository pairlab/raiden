"""Synthetic camera geometry and CLI checks; no driver or hardware initialization."""

import ast
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Literal
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest
import tyro
from scipy.spatial.transform import Rotation

from raiden._config import CALIBRATION_POSES_FILE, CAMERA_CONFIG
from raiden.calibration_utils import left_from_right


@pytest.fixture
def calibration_modules():
    root = Path(__file__).parents[1] / "raiden"
    package = ModuleType("raiden.calibration")
    package.__path__ = [str(root / "calibration")]
    stubs = {
        "raiden.calibration": package,
        "raiden.robot.controller": SimpleNamespace(RobotController=object, smooth_move_joints=Mock()),
        "i2rt.robots.kinematics": SimpleNamespace(Kinematics=object),
        "raiden.cameras.base": SimpleNamespace(Camera=object),
    }
    with patch.dict(sys.modules, stubs):
        loaded = []
        for name in ("core", "runner"):
            full_name = f"raiden.calibration.{name}"
            spec = importlib.util.spec_from_file_location(full_name, root / f"calibration/{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_name] = module
            spec.loader.exec_module(module)
            loaded.append(module)
        yield loaded


def transform(xyz, angles):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
    result[:3, 3] = xyz
    return result


def board_observations(robot_poses, hand_eye, base_board):
    result = []
    for robot in robot_poses:
        camera_board = np.linalg.inv(robot @ hand_eye) @ base_board
        result.append((cv2.Rodrigues(camera_board[:3, :3])[0], camera_board[:3, 3:4]))
    return result


def test_synthetic_hand_eye_and_bimanual_direction_with_separate_pose_sets(calibration_modules):
    core, runner = calibration_modules
    expected = transform([0.612, -0.028, 0.015], [0.1, -0.05, 0.24])
    left_board = transform([0.5, 0.2, -0.018], [0.05, 0.1, 0.3])
    hand_eyes = [transform([0.02, -0.03, 0.04], [0.1, 0.2, -0.3]),
                 transform([-0.01, 0.025, 0.035], [-0.2, 0.1, 0.2])]
    rng = np.random.default_rng(24)
    robots = [[transform(rng.normal(0, 0.2, 3), rng.normal(0, 0.5, 3))
               for _ in range(count)] for count in (9, 6)]
    boards = [board_observations(robots[0], hand_eyes[0], left_board),
              board_observations(robots[1], hand_eyes[1], np.linalg.inv(expected) @ left_board)]
    calibrator = core.CameraCalibrator(core.ChArUcoBoardConfig(4, 5, 0.05, 0.037))
    solves = [calibrator.calibrate_hand_eye(robot, board) for robot, board in zip(robots, boards)]
    for solved, known in zip(solves, hand_eyes):
        np.testing.assert_allclose(solved["rotation_matrix"], known[:3, :3], atol=1e-7)
        np.testing.assert_allclose(solved["translation_vector"], known[:3, 3], atol=1e-7)
    legacy = runner.compute_bimanual_base_transform_from_calibration(
        robots[0], robots[1], boards[0], boards[1], solves[0], solves[1]
    )
    np.testing.assert_allclose(legacy, np.linalg.inv(expected), atol=1e-7)
    for entry in ({"right_base_to_left_base": legacy.tolist()},
                  {"T_left_from_right": expected.tolist()},
                  {"T_left_from_right": expected.tolist(), "right_base_to_left_base": np.eye(4).tolist()}):
        np.testing.assert_allclose(left_from_right({"bimanual_transform": entry}), expected, atol=1e-7)
    point = np.array([0.1, 0.3, 0.4, 1])
    np.testing.assert_allclose(expected @ point, left_board @ np.linalg.inv(np.linalg.inv(expected) @ left_board) @ point)


def test_failed_pnp_and_fk_filter_together_without_shifting(calibration_modules):
    _, module = calibration_modules
    runner = module.CalibrationRunner()
    boards = [(np.zeros((3, 1)), np.full((3, 1), i)) if i != 1 else (None, None)
              for i in range(4)]
    runner.calibrator = SimpleNamespace(detector=SimpleNamespace(estimate_pose=Mock(side_effect=boards)))
    robots = [np.eye(4) * (i + 1) if i != 3 else None for i in range(4)]
    data = {"corners": list(range(4)), "ids": list(range(4)),
            "robot_poses": robots, "observation_ids": [0, 2, 5, 7]}
    valid_robots, valid_boards = runner._valid_wrist_observations(data, np.eye(3), np.zeros(5))
    assert len(valid_robots) == 2
    np.testing.assert_array_equal(valid_robots[1], robots[2])
    np.testing.assert_array_equal(valid_boards[1][1], np.full((3, 1), 2))


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf")])
def test_missing_or_nonfinite_extrinsics_cannot_be_saved(calibration_modules, bad):
    _, module = calibration_modules
    extrinsics = {"success": True, "rotation_matrix": np.eye(3).tolist(),
                  "translation_vector": [0, 0, bad]}
    if bad is None:
        extrinsics = {}
    results = {"coordinate_frame": "left_arm_base", "cameras": {
        "left_wrist_camera": {"type": "hand_eye", "hand_eye_calibration": extrinsics}}}
    with pytest.raises(ValueError):
        module.CalibrationRunner._validate_results(results, ["left_wrist_camera"], False)
    with pytest.raises(ValueError, match="Bimanual"):
        module.CalibrationRunner._validate_results(results, ["left_wrist_camera"], True)


def test_hand_eye_rejects_nonfinite_solver_result(calibration_modules):
    core, _ = calibration_modules
    calibrator = core.CameraCalibrator(core.ChArUcoBoardConfig())
    with patch.object(cv2, "calibrateHandEye", return_value=(np.full((3, 3), np.nan), np.zeros((3, 1)))):
        result = calibrator.calibrate_hand_eye([np.eye(4)] * 3, [(np.zeros((3, 1)), np.zeros((3, 1)))] * 3)
    assert not result["success"]


def test_capture_artifact_retains_both_joints_and_observation_id(calibration_modules, tmp_path):
    _, module = calibration_modules
    runner = module.CalibrationRunner(output_file=str(tmp_path / "candidate.json"))
    runner.board_config = SimpleNamespace(squares_x=4, squares_y=5)
    runner._captures = [{"id": 7, "pose": "pose_7", "joints_l": [1] * 7,
                         "joints_r": [2] * 7, "images": {}}]
    capture_dir = runner._save_captures()
    capture = json.loads((capture_dir / "captures.json").read_text())["poses"][0]
    assert capture["id"] == 7
    assert capture["joints_l"] == [1] * 7
    assert capture["joints_r"] == [2] * 7


def test_replay_thread_failure_prevents_camera_capture(calibration_modules, monkeypatch):
    _, module = calibration_modules
    runner = module.CalibrationRunner()
    runner.poses_data = {"poses": [{"name": "pose_0", "follower_l": [0] * 7}]}
    runner.robot_controller = SimpleNamespace(follower_l=object(), follower_r=None)
    runner.cameras = {"left": Mock()}
    runner.camera_config = SimpleNamespace(get_role=lambda _: "left_wrist")
    monkeypatch.setattr(module, "smooth_move_joints", Mock(side_effect=OSError("CAN read failed")))
    with pytest.raises(RuntimeError, match="movement failed"):
        runner.collect_calibration_data()
    runner.cameras["left"].grab.assert_not_called()


def test_explicit_camera_ids_support_legacy_configs_without_roles(calibration_modules, monkeypatch):
    _, module = calibration_modules
    runner = module.CalibrationRunner()
    runner._wrist_arms = {"custom_l": "left", "custom_r": "right"}
    runner.poses_data = {"poses": [{"name": "pose_0"}]}
    runner.camera_config = SimpleNamespace(get_role=lambda _: None)
    runner.robot_controller = SimpleNamespace(
        follower_l=SimpleNamespace(get_joint_pos=lambda: np.arange(7)),
        follower_r=SimpleNamespace(get_joint_pos=lambda: np.arange(7) + 10))
    runner.cameras = {name: SimpleNamespace(
        grab=lambda: True, get_frame=lambda: SimpleNamespace(color=np.zeros((2, 2, 3), np.uint8)))
        for name in ("custom_l", "custom_r")}
    runner.calibrator = SimpleNamespace(detector=SimpleNamespace(detect=lambda _: (np.ones((4, 1, 2)), np.arange(4))))
    monkeypatch.setattr(module, "_kinematics_cache", SimpleNamespace(fk=lambda q: np.eye(4)))
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    data = runner.collect_calibration_data()
    assert len(data["custom_l"]["robot_poses"]) == len(data["custom_r"]["robot_poses"]) == 1
    assert data["custom_l"]["observation_ids"] == data["custom_r"]["observation_ids"] == [0]
    assert runner._captures[0]["joints_r"] == list(range(10, 17))


def test_new_bimanual_run_discards_previous_in_process_base_transform(calibration_modules):
    _, module = calibration_modules
    runner = module.CalibrationRunner()
    runner.load_configuration = Mock()
    runner.initialize_cameras = Mock(side_effect=RuntimeError("stop before hardware"))
    module.T_RIGHT_BASE_TO_LEFT_BASE = np.eye(4)
    with pytest.raises(RuntimeError, match="stop before hardware"):
        runner.run_calibration("custom_l", "custom_r")
    assert module.T_RIGHT_BASE_TO_LEFT_BASE is None
    assert runner._wrist_arms == {"custom_l": "left", "custom_r": "right"}


@pytest.mark.parametrize("stage", ["open", "warmup"])
def test_camera_preflight_failure_closes_partial_handles_before_arm_initialization(
    calibration_modules, stage
):
    _, module = calibration_modules
    runner = module.CalibrationRunner()
    cameras = {name: Mock() for name in ("left", "right")}
    runner.load_configuration = Mock()
    runner.initialize_robots = Mock()
    runner.camera_config = SimpleNamespace(
        get_camera_type=lambda _: "realsense", get_serial_by_name=lambda name: name,
        create_camera=cameras.__getitem__,
    )
    if stage == "open":
        cameras["left"].open.side_effect = OSError("USB open failed")
        expected_error, message = OSError, "USB open failed"
    else:
        cameras["left"].grab.return_value = False
        expected_error, message = RuntimeError, "warmup failed"
    with pytest.raises(expected_error, match=message):
        runner.run_calibration("left", "right")
    # Targets open right first; a partially opened left handle must also close.
    for camera in cameras.values():
        camera.close.assert_called_once()
    runner.initialize_robots.assert_not_called()


def test_capture_retains_exact_measured_joints_when_detection_or_joint_read_fails(
    calibration_modules, monkeypatch
):
    _, module = calibration_modules
    runner = module.CalibrationRunner()
    runner.poses_data = {"poses": [{"name": f"pose_{i}"} for i in range(3)]}
    runner.camera_config = SimpleNamespace(get_role=lambda _: "left_wrist")
    joint_reader = Mock(side_effect=[OSError("joint read failed"), np.arange(7), np.arange(7) + 10])
    runner.robot_controller = SimpleNamespace(
        follower_l=SimpleNamespace(get_joint_pos=joint_reader), follower_r=None
    )
    runner.cameras = {"left": SimpleNamespace(
        grab=lambda: True, get_frame=lambda: SimpleNamespace(color=np.zeros((2, 2, 3), np.uint8))
    )}
    detected = (np.ones((4, 1, 2)), np.arange(4))
    runner.calibrator = SimpleNamespace(detector=SimpleNamespace(
        detect=Mock(side_effect=[detected, (None, None), detected])
    ))
    kinematics = SimpleNamespace(fk=Mock(return_value=np.eye(4)))
    monkeypatch.setattr(module, "_kinematics_cache", kinematics)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    data = runner.collect_calibration_data()["left"]
    assert joint_reader.call_count == 3
    assert data["observation_ids"] == [0, 2]
    assert data["robot_poses"][0] is None
    np.testing.assert_array_equal(data["robot_poses"][1], np.eye(4))
    assert [pose["joints_l"] for pose in runner._captures] == [None, list(range(7)), list(range(10, 17))]
    np.testing.assert_array_equal(kinematics.fk.call_args.args[0][:6], np.arange(6) + 10)
    assert all("left" in pose["images"] for pose in runner._captures)


@pytest.mark.parametrize("camera_close_fails", [False, True])
def test_failed_run_retains_partial_captures_and_closes_all_resources(
    calibration_modules, monkeypatch, tmp_path, camera_close_fails
):
    core, module = calibration_modules
    runner = module.CalibrationRunner(output_file=str(tmp_path / "candidate.json"))
    runner.board_config = core.ChArUcoBoardConfig(4, 5, 0.05, 0.037)
    runner.load_configuration = Mock()
    runner.initialize_cameras = Mock()
    runner.initialize_robots = Mock()
    calls = []
    cameras = {name: Mock() for name in ("left", "right")}
    for name, camera in cameras.items():
        camera.get_intrinsics.return_value = np.eye(3), np.zeros(5), (2, 2)
        camera.close.side_effect = lambda name=name: calls.append(name)
    if camera_close_fails:
        def fail_close():
            calls.append("left")
            raise OSError("camera close failed")
        cameras["left"].close.side_effect = fail_close
    runner.cameras = cameras
    controller = Mock()
    controller.has_robots.return_value = True
    controller.emergency_stop.side_effect = lambda: calls.append("emergency_stop")
    runner.robot_controller = controller

    def fail_collection():
        runner._captures.append({"id": 0, "pose": "pose_0", "joints_l": [1] * 7,
                                 "joints_r": [2] * 7,
                                 "images": {"left": np.zeros((2, 2, 3), np.uint8)}})
        raise ValueError("collection failed")

    runner.collect_calibration_data = fail_collection
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    with pytest.raises(ValueError, match="collection failed"):
        runner.run_calibration("left", "right")
    assert runner._move_stop_event.is_set()
    assert calls == ["left", "right", "emergency_stop"]
    controller.stop_teleoperation.assert_called_once()
    controller.close.assert_called_once()
    controller.cleanup.assert_not_called()
    artifact = next(tmp_path.glob("calibration_captures/*/captures.json"))
    retained = json.loads(artifact.read_text())["poses"][0]
    assert retained["joints_r"] == [2] * 7
    assert (artifact.parent / retained["images"]["left"]).exists()
    assert not (tmp_path / "candidate.json").exists()


def test_replay_interruption_stops_worker_before_collection_returns(calibration_modules, monkeypatch):
    _, module = calibration_modules
    runner = module.CalibrationRunner()
    runner.poses_data = {"poses": [{"name": "pose_0", "follower_l": [0] * 7}]}
    runner.robot_controller = SimpleNamespace(follower_l=object(), follower_r=None)
    runner.cameras = {"left": Mock()}
    runner.camera_config = SimpleNamespace(get_role=lambda _: "left_wrist")
    started = module.threading.Event()
    finished = module.threading.Event()
    workers = []
    original_join = module.threading.Thread.join

    def move(robot, target, stop_event):
        started.set()
        stop_event.wait(10)
        finished.set()

    def interrupted_join(thread):
        if not workers:
            workers.append(thread)
            assert started.wait(1)
            raise KeyboardInterrupt
        original_join(thread, timeout=1)

    monkeypatch.setattr(module, "smooth_move_joints", move)
    monkeypatch.setattr(module.threading.Thread, "join", interrupted_join)
    try:
        with pytest.raises(KeyboardInterrupt):
            runner.collect_calibration_data()
        assert runner._move_stop_event.is_set()
        assert finished.is_set()
        runner.cameras["left"].grab.assert_not_called()
    finally:
        runner._move_stop_event.set()
        for worker in workers:
            original_join(worker, timeout=1)


def test_scene_conversion_uses_matched_valid_left_observations_and_returns_cleanly(
    calibration_modules, monkeypatch, tmp_path
):
    core, module = calibration_modules
    rng = np.random.default_rng(28)
    hand_eye = transform([0.02, -0.03, 0.04], [0.1, 0.2, -0.3])
    base_board = transform([0.5, 0.2, -0.018], [0.05, 0.1, 0.3])
    base_scene = transform([-0.2, 0.3, 0.8], [2.5, -0.2, 0.1])
    robots = [transform(rng.normal(0, 0.2, 3), rng.normal(0, 0.5, 3)) for _ in range(8)]
    observations = board_observations(robots, hand_eye, base_board)
    observations[4] = None  # Failed PnP must not shift the remaining robot poses.
    robots[1] = None       # A separate failed FK must drop its own observation.
    scene_board = np.linalg.inv(base_scene) @ base_board
    scene_observation = (cv2.Rodrigues(scene_board[:3, :3])[0], scene_board[:3, 3:4])
    runner = module.CalibrationRunner(output_file=str(tmp_path / "candidate.json"))
    runner.load_configuration = Mock()
    runner.initialize_cameras = Mock()
    runner.initialize_robots = Mock()
    runner.board_config = core.ChArUcoBoardConfig(4, 5, 0.05, 0.037)
    runner.poses_data = {"poses": [{} for _ in robots]}
    runner.camera_config = SimpleNamespace(get_camera_type=lambda _: "realsense",
                                          get_serial_by_name=lambda name: name)
    cameras = {name: Mock() for name in ("left", "scene")}
    for camera in cameras.values():
        camera.get_intrinsics.return_value = np.eye(3), np.zeros(5), (640, 360)
    runner.cameras = cameras
    runner.calibrator = core.CameraCalibrator(runner.board_config)

    def estimate(corner, ids, matrix, distortion):
        pose = observations[corner] if corner < 8 else scene_observation
        return pose if pose is not None else (None, None)

    runner.calibrator.detector.estimate_pose = estimate
    runner.collect_calibration_data = Mock(return_value={
        "left": {"corners": list(range(8)), "ids": [None] * 8, "robot_poses": robots},
        "scene": {"corners": list(range(8, 16)), "ids": [None] * 8, "robot_poses": []},
    })
    runner.robot_controller = Mock()
    # A reused runner must clear a prior cancellation before its next run.
    runner._move_stop_event.set()
    runner.initialize_cameras.side_effect = lambda _: (
        None if not runner._move_stop_event.is_set() else pytest.fail("stale cancellation")
    )
    monkeypatch.setitem(sys.modules, "raiden.db.database", SimpleNamespace(get_db=lambda: Mock()))
    result = runner.run_calibration("left", scene_camera_ids=["scene"])
    assert result["quality_metrics"]["overall_success"]
    assert result["cameras"]["left"]["num_poses_used"] == 6
    actual_scene = result["cameras"]["scene"]["extrinsics"]
    assert actual_scene["reference_frame"] == "left_arm_base"
    np.testing.assert_allclose(actual_scene["rotation_matrix"], base_scene[:3, :3], atol=1e-7)
    np.testing.assert_allclose(actual_scene["translation_vector"], base_scene[:3, 3], atol=1e-7)
    runner.robot_controller.cleanup.assert_not_called()
    runner.robot_controller.stop_teleoperation.assert_called_once()
    runner.robot_controller.close.assert_called_once()
    for camera in cameras.values():
        camera.close.assert_called_once()
    assert json.loads((tmp_path / "candidate.json").read_text()) == result


def test_recorder_cli_accepts_previous_board_and_custom_camera_config():
    # Load the actual CLI dataclass without importing driver-heavy CLI modules.
    source = Path(__file__).parents[1] / "raiden/cli.py"
    definition = next(node for node in ast.parse(source.read_text()).body
                      if isinstance(node, ast.ClassDef) and node.name == "RecordCalibrationPosesCommand")
    namespace = dict(dataclass=dataclass, Literal=Literal,
                     CALIBRATION_POSES_FILE=CALIBRATION_POSES_FILE, CAMERA_CONFIG=CAMERA_CONFIG)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)
    command = tyro.cli(namespace["RecordCalibrationPosesCommand"], args=[
        "--control", "oculus", "--min-poses", "12", "--camera-config-file", "/tmp/camera.json",
        "--squares-x", "4", "--squares-y", "5", "--square-length", "0.05",
        "--marker-length", "0.037", "--dictionary", "DICT_6X6_250",
    ])
    assert (command.control, command.min_poses, command.camera_config_file) == ("oculus", 12, "/tmp/camera.json")
    assert (command.squares_x, command.squares_y, command.square_length, command.marker_length) == (4, 5, 0.05, 0.037)


def test_invalid_canonical_transform_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        left_from_right({"bimanual_transform": {"T_left_from_right": [[float("nan")]]}})
