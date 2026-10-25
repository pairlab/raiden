"""Two Quest controllers keep motion and phase-changing button edges separate."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


@pytest.fixture
def oculus_module(monkeypatch, tmp_path):
    root = Path(__file__).parents[1] / "raiden"
    control = ModuleType("raiden.control")
    control.__path__ = [str(root / "control")]
    robot = ModuleType("raiden.robot")
    robot.__path__ = []
    modules = {
        "raiden.control": control,
        "raiden.robot": robot,
        "raiden.robot.footpedal": SimpleNamespace(
            PEDAL_LEFT=1, PEDAL_MIDDLE=2, PEDAL_RIGHT=3, try_open_footpedal=Mock()
        ),
        "raiden.robot.oculus": SimpleNamespace(OculusReader=Mock()),
    }
    spec = importlib.util.spec_from_file_location(
        "_oculus_bimanual_test", root / "control/oculus.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
        monkeypatch.setattr(module, "_CALIB_FILE", tmp_path / "oculus_calibration.json")
        yield module


def controller(*, left=True, right=True):
    return SimpleNamespace(
        follower_l=object() if left else None,
        follower_r=object() if right else None,
        start_cartesian_teleop=Mock(),
        stop_cartesian_teleop=Mock(),
    )


class Reader:
    age = 0.0

    def __init__(self):
        self.poses = {hand: np.eye(4) for hand in ("l", "r")}
        self.buttons = {}

    def get(self):
        return self.poses, self.buttons.copy()


def interface_with_reader(module, **kwargs):
    interface = module.OculusInterface(**kwargs)
    interface._reader = Reader()
    return interface


def step(interface, side):
    current = np.eye(4)
    current[:3, 3] = [0.3, 0.0, 0.2]
    return interface._target_fn(side, current, 1.0)


def calibration_move(interface, hand, delta):
    side = "left" if hand == "l" else "right"
    button = "X" if hand == "l" else "A"
    interface._reader.poses[hand] = np.eye(4)
    interface._reader.buttons = {button: True}
    step(interface, side)
    interface._reader.poses[hand][:3, 3] = delta
    interface._reader.buttons = {}
    step(interface, side)


@pytest.mark.parametrize("hand", [None, "left", "R", "", 1])
def test_invalid_hand_names_fail_before_opening_reader(oculus_module, hand):
    with pytest.raises(ValueError, match="must be 'l' or 'r'"):
        oculus_module.OculusInterface(hand_for_left_arm=hand)
    oculus_module.OculusReader.assert_not_called()


def test_duplicate_hand_override_rejected_for_both_active_arms(oculus_module):
    interface = oculus_module.OculusInterface(hand_for_left_arm="r")
    robot = controller()
    with pytest.raises(ValueError, match="distinct hand"):
        interface.setup(robot)
    with pytest.raises(ValueError, match="distinct hand"):
        interface.start_ready(robot)
    robot.start_cartesian_teleop.assert_not_called()


def test_single_left_arm_preserves_right_hand_override(oculus_module):
    interface = interface_with_reader(oculus_module, hand_for_left_arm="r")
    robot = controller(right=False)
    interface.setup(robot)
    interface.start_ready(robot)
    assert interface._active_hands == {"r"}
    interface._reader.buttons = {"A": True}
    step(interface, "left")
    assert interface.poll(robot)


def test_both_hands_drive_independent_clutch_grip_and_relative_motion(oculus_module):
    interface = interface_with_reader(oculus_module, pos_scale=1.0)
    robot = controller()
    interface.start(robot)
    interface._R = {"l": np.eye(3), "r": np.eye(3)}
    interface._alpha = 1.0
    interface._reader.buttons = {"LTr": True, "RTr": True, "LG": True}
    left = step(interface, "left")
    assert left[1] == 0.0
    assert step(interface, "right") is None
    assert interface._origin["l"] is not interface._origin["r"]
    interface._reader.poses["l"][0, 3] = 0.04
    interface._reader.poses["r"][1, 3] = -0.03
    left, right = step(interface, "left"), step(interface, "right")
    np.testing.assert_allclose(left[0][:3, 3], [0.34, 0.0, 0.2])
    np.testing.assert_allclose(right[0][:3, 3], [0.3, -0.03, 0.2])
    assert left[1] == 0.0 and right[1] == 1.0
    interface._reader.buttons = {"RTr": True}
    step(interface, "left")
    assert interface._origin["l"] is None
    assert interface._origin["r"] is not None


def test_ready_start_stop_and_fresh_verdict_edges(oculus_module):
    interface = interface_with_reader(oculus_module)
    robot = controller()
    interface.start_ready(robot)
    interface._reader.buttons = {"X": True}
    assert step(interface, "left") is None
    assert interface.poll(robot) and not interface.poll(robot)
    interface.set_active_recording(robot)
    step(interface, "left")
    assert not interface.poll(robot)  # held READY button is not a stop
    interface._reader.buttons = {"A": True}
    step(interface, "left")
    step(interface, "right")
    assert interface.poll(robot)
    interface.set_active_recording(None)
    step(interface, "right")
    assert not interface.poll_success(robot)  # held stop is not a verdict
    interface._reader.buttons = {"X": True}
    step(interface, "left")
    assert interface.poll_success(robot)


def test_simultaneous_two_hand_stop_cannot_become_success_on_phase_change(
    oculus_module,
):
    interface = interface_with_reader(oculus_module)
    robot = controller()
    interface.start_ready(robot)
    interface.set_active_recording(robot)
    interface._reader.buttons = {"X": True, "A": True}
    step(interface, "left")  # main thread sees stop before right arm's next tick
    assert interface.poll(robot)
    interface.set_active_recording(None)
    step(interface, "right")
    assert not interface.poll_success(robot)
    interface._reader.buttons = {}
    step(interface, "right")
    interface._reader.buttons = {"A": True}
    step(interface, "right")
    assert interface.poll_success(robot)


def test_failure_button_is_ignored_while_recording_then_works_for_verdict(
    oculus_module,
):
    interface = interface_with_reader(oculus_module)
    robot = controller()
    interface.start_ready(robot)
    interface.set_active_recording(robot)
    interface._reader.buttons = {"B": True}
    step(interface, "right")
    assert not interface.poll_failure(robot)
    assert not interface.calibrating
    interface.set_active_recording(None)
    step(interface, "right")
    assert not interface.poll_failure(robot)
    interface._reader.buttons = {}
    step(interface, "right")
    interface._reader.buttons = {"Y": True}
    step(interface, "left")
    assert interface.poll_failure(robot)


def test_sim_collection_success_discard_and_calibration_buttons_are_preserved(
    oculus_module,
):
    interface = interface_with_reader(oculus_module)
    interface.auto_reset = True
    robot = controller(right=False)
    interface.start(robot)
    interface.set_active_recording(robot)
    interface._reader.buttons = {"Y": True}
    step(interface, "left")
    assert interface.poll_success(robot)
    assert not interface.calibrating
    interface._reader.buttons = {"X": True}
    step(interface, "left")
    assert interface.poll_failure(robot)
    interface._reader.buttons = {"LJ": True}
    step(interface, "left")
    assert interface.calibrating


@pytest.mark.parametrize("hand", ["l", "r"])
@pytest.mark.parametrize("angles", [[25.0, -35.0, 123.0], [-70.0, 48.0, -18.0]])
def test_calibration_recovers_arbitrary_mount_rotation_and_persists(
    oculus_module, hand, angles
):
    interface = interface_with_reader(oculus_module)
    expected = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
    interface._on_calib_press(hand)
    calibration_move(interface, hand, expected.T @ [0.2, 0.0, 0.0])
    calibration_move(interface, hand, expected.T @ [0.0, 0.2, 0.0])
    assert not interface.calibrating
    np.testing.assert_allclose(interface._R[hand], expected, atol=1e-12)
    reloaded = oculus_module.OculusInterface()
    np.testing.assert_allclose(reloaded._saved_R[hand], expected, atol=1e-12)

    # Translation and relative angular motion must use that same base frame.
    interface._alpha = 1.0
    interface._pos_scale = interface._rot_scale = 1.0
    side = "left" if hand == "l" else "right"
    interface._reader.buttons = {"LTr" if hand == "l" else "RTr": True}
    interface._reader.poses[hand] = np.eye(4)
    assert step(interface, side) is None
    interface._reader.poses[hand][:3, 3] = expected.T @ [0.025, -0.03, 0.02]
    interface._reader.poses[hand][:3, :3] = Rotation.from_rotvec(
        expected.T @ [0.05, -0.03, 0.02]
    ).as_matrix()
    target, _ = step(interface, side)
    np.testing.assert_allclose(target[:3, 3], [0.325, -0.03, 0.22], atol=1e-12)
    np.testing.assert_allclose(
        target[:3, :3], Rotation.from_rotvec([0.05, -0.03, 0.02]).as_matrix(), atol=1e-12
    )


def test_calibration_fits_noisy_perpendicular_moves_to_a_rigid_rotation(oculus_module):
    interface = interface_with_reader(oculus_module)
    interface._on_calib_press("l")
    calibration_move(interface, "l", [0.2, 0.0, 0.0])
    calibration_move(interface, "l", [0.04, 0.2, 0.0])
    rotation = interface._R["l"]
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    np.testing.assert_allclose(
        rotation,
        Rotation.from_euler("z", np.arctan(0.2) / 2).as_matrix(),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "bad_delta", [[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.2, 0.01, 0.0]]
)
def test_calibration_rejects_parallel_or_antiparallel_moves_and_can_retry(
    oculus_module, bad_delta
):
    interface = interface_with_reader(oculus_module)
    interface._on_calib_press("l")
    calibration_move(interface, "l", [0.2, 0.0, 0.0])
    calibration_move(interface, "l", bad_delta)
    assert interface.calibrating
    assert interface._R["l"] is None
    assert not oculus_module._CALIB_FILE.exists()
    calibration_move(interface, "l", [0.0, 0.2, 0.0])
    assert not interface.calibrating
    np.testing.assert_allclose(interface._R["l"], np.eye(3))


@pytest.mark.parametrize("bad_delta", [[0.04, 0.0, 0.0], [np.nan, 0.2, 0.0]])
def test_invalid_or_short_calibration_move_keeps_previous_rotation(
    oculus_module, bad_delta
):
    interface = interface_with_reader(oculus_module)
    previous = Rotation.from_euler("z", 0.4).as_matrix()
    interface._R["l"] = previous.copy()
    interface._on_calib_press("l")
    calibration_move(interface, "l", bad_delta)
    assert interface._calib["l"]["stage"] == "x"
    np.testing.assert_array_equal(interface._R["l"], previous)
    interface._on_calib_press("l")
    assert not interface.calibrating
    np.testing.assert_array_equal(interface._R["l"], previous)


def test_legacy_saved_axis_rotations_load_alongside_new_arbitrary_rotations(oculus_module):
    legacy = [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    oculus_module._CALIB_FILE.write_text(json.dumps({"l": legacy}))
    interface = interface_with_reader(oculus_module)
    np.testing.assert_array_equal(interface._R["l"], legacy)
    expected = Rotation.from_euler("xyz", [0.2, -0.3, 0.5]).as_matrix()
    interface._on_calib_press("r")
    calibration_move(interface, "r", expected.T @ [0.2, 0.0, 0.0])
    calibration_move(interface, "r", expected.T @ [0.0, 0.2, 0.0])
    loaded = oculus_module.OculusInterface()
    np.testing.assert_array_equal(loaded._R["l"], legacy)
    np.testing.assert_allclose(loaded._R["r"], expected, atol=1e-12)


@pytest.mark.parametrize(
    "bad_rotation",
    [
        None,
        [[1.0]],
        np.zeros((3, 3)).tolist(),
        np.diag([1.0, 1.0, -1.0]).tolist(),
        [[1.0, 0.0, 0.0], [0.0, np.nan, 0.0], [0.0, 0.0, 1.0]],
    ],
)
def test_malformed_saved_rotation_does_not_disable_valid_other_hand(
    oculus_module, bad_rotation
):
    oculus_module._CALIB_FILE.write_text(
        json.dumps({"l": bad_rotation, "r": np.eye(3).tolist()})
    )
    interface = oculus_module.OculusInterface()
    assert interface._R["l"] is None
    np.testing.assert_array_equal(interface._R["r"], np.eye(3))


@pytest.mark.parametrize("loss", ["stale", "missing", "nonfinite", "tracking"])
def test_held_clutch_recovery_rebases_without_filter_drift(oculus_module, loss):
    interface = interface_with_reader(oculus_module, pos_scale=1.0, rot_scale=1.0)
    interface.start(controller())
    interface._R = {"l": np.eye(3), "r": np.eye(3)}
    interface._reader.buttons = {"LTr": True, "RTr": True}
    assert step(interface, "left") is None
    assert step(interface, "right") is None
    right_origin = interface._origin["r"]
    if loss == "stale":
        interface._reader.age = 1.0
    elif loss == "missing":
        del interface._reader.poses["l"]
    elif loss == "nonfinite":
        interface._reader.poses["l"][0, 3] = np.nan
    else:
        interface._tracking["l"] = "ORIENTATION"
    assert step(interface, "left") is None
    assert interface._origin["l"] is None
    assert interface._smooth["l"] is None
    assert interface._origin["r"] is right_origin
    interface._reader.age = 0.0
    interface._tracking["l"] = "POSITION"
    resumed = np.eye(4)
    resumed[:3, 3] = [0.4, -0.3, 0.2]
    resumed[:3, :3] = Rotation.from_rotvec([0.3, -0.2, 0.4]).as_matrix()
    interface._reader.poses["l"] = resumed
    assert step(interface, "left") is None
    for _ in range(5):
        target, _ = step(interface, "left")
        np.testing.assert_allclose(target[:3, 3], [0.3, 0.0, 0.2], atol=1e-12)
        np.testing.assert_allclose(target[:3, :3], np.eye(3), atol=1e-12)


@pytest.mark.parametrize("loss", ["stale", "missing", "nonfinite", "tracking"])
def test_interrupted_calibration_move_is_rejected_after_recovery(oculus_module, loss):
    interface = interface_with_reader(oculus_module)
    interface._on_calib_press("l")
    interface._reader.buttons = {"X": True}
    step(interface, "left")
    if loss == "stale":
        interface._reader.age = 1.0
    elif loss == "missing":
        del interface._reader.poses["l"]
    elif loss == "nonfinite":
        interface._reader.poses["l"][0, 3] = np.nan
    else:
        interface._tracking["l"] = "ORIENTATION"
    step(interface, "left")
    interface._reader.age = 0.0
    interface._tracking["l"] = "POSITION"
    interface._reader.poses["l"] = np.eye(4)
    interface._reader.poses["l"][0, 3] = 0.2
    interface._reader.buttons = {}
    step(interface, "left")
    assert interface._calib["l"]["stage"] == "x"
    assert interface._R["l"] is None
    calibration_move(interface, "l", [0.2, 0.0, 0.0])
    calibration_move(interface, "l", [0.0, 0.2, 0.0])
    np.testing.assert_allclose(interface._R["l"], np.eye(3))
