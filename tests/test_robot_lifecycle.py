"""Exercise real controller orchestration with drivers and IK imports replaced."""

import importlib.util
import json
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest


@pytest.fixture
def controller_module():
    class CanInterface:
        def __init__(self, *args, **kwargs):
            pass

    class GripperType(Enum):
        LINEAR_4310 = "linear_4310"
        YAM_TEACHING_HANDLE = "yam_teaching_handle"

    robot_package = ModuleType("raiden.robot")
    robot_package.__path__ = []
    jax = ModuleType("jax")
    jax.__path__ = []
    modules = {
        "jax": jax,
        "jax.numpy": np,
        "jaxlie": ModuleType("jaxlie"),
        "pyroki": ModuleType("pyroki"),
        "pyspacemouse": ModuleType("pyspacemouse"),
        "yourdfpy": ModuleType("yourdfpy"),
        "i2rt.motor_drivers.can_interface": SimpleNamespace(CanInterface=CanInterface),
        "i2rt.robots.get_robot": SimpleNamespace(get_yam_robot=Mock()),
        "i2rt.robots.motor_chain_robot": SimpleNamespace(MotorChainRobot=object),
        "i2rt.robots.robot": SimpleNamespace(Robot=object),
        "i2rt.robots.utils": SimpleNamespace(
            ARM_YAM_XML_PATH="/unused/yam.xml", GripperType=GripperType
        ),
        "raiden.robot": robot_package,
        "raiden.robot._jparse": SimpleNamespace(jparse_step=Mock()),
        "raiden.robot.footpedal": SimpleNamespace(try_open_footpedal=Mock()),
    }
    source = Path(__file__).parents[1] / "raiden/robot/controller.py"
    spec = importlib.util.spec_from_file_location("_controller_lifecycle_test", source)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
        yield module


class FakeRobot:
    def __init__(self, name, *, update_error=None, close_error=None):
        self.name = name
        self._kp = np.ones(7)
        self._kd = np.ones(7)
        self.motor_chain = object()
        self.update_error = update_error
        self.close_error = close_error
        self.closed = 0

    def update_kp_kd(self, kp, kd):
        if self.update_error is not None:
            raise self.update_error
        self._kp, self._kd = kp.copy(), kd.copy()

    def get_robot_info(self):
        return {
            "kp": self._kp,
            "kd": self._kd,
            "gripper_limits": np.array([0.1, 6.67]),
            "limit_gripper_effort": 50.0,
        }

    def close(self):
        self.closed += 1
        if self.close_error is not None:
            raise self.close_error


def followers(module, **kwargs):
    return module.RobotController(
        use_right_leader=False, use_left_leader=False, can_channels={}, **kwargs
    )


def test_one_mapping_drives_preflight_initialization_and_portable_metadata(
    controller_module, monkeypatch
):
    module = controller_module
    checked = []
    monkeypatch.setattr(
        module, "check_can_interface", lambda channel: checked.append(channel) or True
    )
    robots = {channel: FakeRobot(channel) for channel in ("can_left", "can_right")}
    factory = Mock(side_effect=lambda *, channel, **_: robots[channel])
    monkeypatch.setattr(module, "get_yam_robot", factory)
    controller = module.RobotController(
        use_right_leader=False,
        use_left_leader=False,
        can_channels={"follower_l": "can_left", "follower_r": "can_right"},
    )
    assert controller.check_can_interfaces()
    controller.initialize_robots()
    assert checked == ["can_left", "can_right"]
    assert {call.kwargs["channel"] for call in factory.call_args_list} == set(checked)
    assert all(
        call.kwargs["zero_gravity_mode"] is False for call in factory.call_args_list
    )
    config = json.loads(json.dumps(controller.recording_config(), allow_nan=False))
    assert (
        config["active_followers"]
        == config["follower_order"]
        == ["follower_l", "follower_r"]
    )
    assert config["followers"]["follower_l"]["can_channel"] == "can_left"
    assert config["followers"]["follower_r"]["gripper_limits"] == [0.1, 6.67]
    assert config["followers"]["follower_r"]["limit_gripper_force"] == 50.0
    assert config["followers"]["follower_l"]["kp"] == module.FOLLOWER_KP.tolist()
    assert config["state_order"][-1] == "gripper_aperture"
    assert config["command_order"][-1] == "gripper_intent"
    assert config["command_gripper"]["semantics"] == "binary_operator_intent"
    assert config["state_gripper"]["semantics"] == "normalized_aperture"
    assert config["motor_gripper_command_range"] == [20.0 / 95.0, 1.0]


@pytest.mark.parametrize(
    "channels",
    [
        {"unknown": "can_left"},
        {"follower_l": 1},
        {"follower_l": "same", "follower_r": "same"},
    ],
)
def test_invalid_mapping_precedes_driver_initialization(controller_module, channels):
    with pytest.raises(ValueError):
        controller_module.RobotController(can_channels=channels)
    controller_module.get_yam_robot.assert_not_called()


def test_partial_initialization_closes_every_success_even_when_close_fails(
    controller_module, monkeypatch
):
    module = controller_module
    left = FakeRobot("left", close_error=OSError("left cleanup"))
    leader = FakeRobot("leader")
    original = OSError("right initialization")

    def factory(*, channel, **kwargs):
        if channel == "can_follower_r":
            raise original
        return left if channel == "can_follower_l" else leader

    monkeypatch.setattr(module, "get_yam_robot", factory)
    controller = module.RobotController(use_right_leader=False, can_channels={})
    with pytest.raises(RuntimeError, match="Robot initialization failed") as caught:
        controller.initialize_robots()
    assert caught.value.__cause__ is original
    assert any("left cleanup" in note for note in caught.value.__notes__)
    assert left.closed == leader.closed == 1
    assert controller.follower_l is controller.follower_r is controller.leader_l is None


@pytest.mark.parametrize("gravity_failure", [False, True])
def test_post_initialization_configuration_failure_closes_all_four_arms(
    controller_module, monkeypatch, gravity_failure
):
    module = controller_module
    original = ValueError("configuration failed")
    robots = {
        name: FakeRobot(name)
        for name in module.load_can_channels(overrides={}).values()
    }
    if not gravity_failure:
        robots["can_follower_l"].update_error = original
    monkeypatch.setattr(
        module, "get_yam_robot", lambda *, channel, **_: robots[channel]
    )
    controller = module.RobotController(can_channels={})
    if gravity_failure:
        monkeypatch.setattr(
            controller, "enable_gravity_compensation", Mock(side_effect=original)
        )
    with pytest.raises(ValueError, match="configuration failed") as caught:
        controller.initialize_robots(gravity_comp_mode=gravity_failure)
    assert caught.value is original
    assert [robot.closed for robot in robots.values()] == [1, 1, 1, 1]
    assert all(
        getattr(controller, name) is None for name in controller._enabled_robot_names()
    )


def test_close_attempts_pedal_and_all_arms_and_retains_first_error(controller_module):
    module = controller_module
    controller = followers(module)
    original = OSError("pedal close")
    resources = [FakeRobot(str(index)) for index in range(5)]
    resources[0].close_error = original
    resources[1].close_error = OSError("right close")
    controller._footpedal, controller.follower_r, controller.follower_l = resources[:3]
    controller.leader_r = SimpleNamespace(_robot=resources[3])
    controller.leader_l = SimpleNamespace(_robot=resources[4])
    with pytest.raises(OSError, match="pedal close") as caught:
        controller.close()
    assert caught.value is original
    assert [resource.closed for resource in resources] == [1] * 5
    assert any("right close" in note for note in original.__notes__)
    assert controller.follower_l is controller.leader_r is controller.leader_l is None


def test_shutdown_home_failure_still_closes_and_preserves_home_error(
    controller_module, monkeypatch
):
    module = controller_module
    controller = followers(module)
    original = ValueError("home failed")
    controller.follower_l = FakeRobot("left", close_error=OSError("cleanup failed"))
    controller.follower_r = FakeRobot("right")
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    monkeypatch.setattr(controller, "return_to_home", Mock(side_effect=original))
    with pytest.raises(ValueError, match="home failed") as caught:
        controller.shutdown()
    assert caught.value is original
    assert controller.follower_r is None
    assert controller.follower_l.closed == 1
    assert any("cleanup failed" in note for note in original.__notes__)


def test_threaded_home_failure_is_propagated_after_every_arm_attempt(
    controller_module, monkeypatch
):
    module = controller_module
    controller = followers(module)
    controller.follower_l, controller.follower_r = FakeRobot("left"), FakeRobot("right")
    moved = []

    def move(robot, *args, **kwargs):
        moved.append(robot.name)
        if robot.name == "left":
            raise OSError("left home failed")

    monkeypatch.setattr(module, "smooth_move_joints", move)
    with pytest.raises(RuntimeError, match="Moving robots home failed") as caught:
        controller.move_to_home_positions()
    assert set(moved) == {"left", "right"}
    assert str(caught.value.__cause__) == "left home failed"


def test_sim_bypasses_machine_channel_file_and_records_no_physical_channel(
    controller_module, monkeypatch
):
    module = controller_module
    loader = Mock(wraps=module.load_can_channels)
    monkeypatch.setattr(module, "load_can_channels", loader)
    controller = followers(module, use_right_follower=False, sim="fake:1234")
    loader.assert_called_once_with((), overrides={})
    fake = FakeRobot("sim")
    # The simulator intentionally does not implement get_robot_info().
    sim = SimpleNamespace(update_kp_kd=fake.update_kp_kd, close=fake.close,
                          get_kp_kd=lambda: (np.full(7, 50.), np.full(7, 5.)))
    with patch.dict(
        sys.modules, {"raiden.sim": SimpleNamespace(SimFollower=lambda _, arm: sim)}
    ):
        assert controller.check_can_interfaces()
        controller.initialize_robots()
    config = controller.recording_config()
    assert config["mode"] == "sim"
    assert config["active_followers"] == ["follower_l"]
    assert config["followers"]["follower_l"]["can_channel"] is None
    assert config["followers"]["follower_l"]["gripper_limits"] is None
    module.get_yam_robot.assert_not_called()


def test_bimanual_sim_uses_two_followers_but_resets_shared_scene_once(controller_module):
    module = controller_module
    controller = followers(module, sim="fake:1234")
    handles = {}

    def factory(address, arm):
        handle = SimpleNamespace(
            get_kp_kd=lambda: (np.full(7, 50.), np.full(7, 5.)),
            update_kp_kd=Mock(), close=Mock(), reset_scene=Mock())
        handles[arm] = handle
        return handle

    with patch.dict(sys.modules, {"raiden.sim": SimpleNamespace(SimFollower=factory)}):
        controller.initialize_robots()
    assert set(handles) == {"left", "right"}
    for handle in handles.values():
        np.testing.assert_array_equal(handle.update_kp_kd.call_args.kwargs["kp"], np.full(7, 50.))
    controller.move_to_home_positions()
    assert sum(handle.reset_scene.call_count for handle in handles.values()) == 1
