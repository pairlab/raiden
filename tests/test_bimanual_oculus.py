"""Two Quest controllers keep motion and phase-changing button edges separate."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest


@pytest.fixture
def oculus_module(monkeypatch):
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
        monkeypatch.setattr(
            module.OculusInterface, "_load_calibration", staticmethod(dict)
        )
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
