"""Host timing provenance and atomic snapshots, without devices or control threads."""

import ast
import threading
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import test_recording_integrity as integrity
import test_robot_lifecycle as lifecycle

controller_module = lifecycle.controller_module


def controller(module):
    return module.RobotController(
        use_right_leader=False, use_left_leader=False, can_channels={}
    )


def test_timed_reads_keep_each_arm_and_unknown_driver_provenance(controller_module, monkeypatch):
    module = controller_module
    robot = controller(module)
    wall_times = iter([1_800_000_000_000_000_001 + i for i in range(4)])
    mono_times = iter([101, 102, 103, 104])
    monkeypatch.setattr(module, "time", SimpleNamespace(
        time_ns=lambda: next(wall_times), monotonic_ns=lambda: next(mono_times)
    ))
    measured = np.arange(6, dtype=float)
    robot.follower_r = SimpleNamespace(get_timed_observations=lambda: (
        {"joint_pos": measured, "gripper_pos": np.array([0.3])},
        {"state_timestamp_ns": 1_800_000_000_000_000_000,
         "state_timestamp_valid": True,
         "state_timestamp_source": "host_cached_feedback_read"},
    ))
    robot.follower_l = SimpleNamespace(get_observations=lambda: {"joint_pos": measured})
    observations, timing = robot.get_all_timed_observations()
    measured[:] = -5
    np.testing.assert_array_equal(observations["follower_r"]["joint_pos"], np.arange(6))
    assert timing["follower_r"]["state_timestamp_ns"] == 1_800_000_000_000_000_000
    assert timing["follower_r"]["read_start_ns"] == 1_800_000_000_000_000_001
    assert timing["follower_r"]["read_end_ns"] == 1_800_000_000_000_000_002
    assert timing["follower_l"]["read_start_monotonic_ns"] == 103
    assert timing["follower_l"]["read_end_monotonic_ns"] == 104
    assert timing["follower_l"]["state_timestamp_ns"] == -1
    assert timing["follower_l"]["state_timestamp_valid"] is False
    assert timing["follower_l"]["state_timestamp_source"] == "unavailable"


def test_command_snapshot_waits_for_api_and_keeps_target_intent_time_together(controller_module, monkeypatch):
    module = controller_module
    robot = controller(module)
    entered, release, reading, read_done = (threading.Event() for _ in range(4))
    wall_times = iter([1_800_000_000_000_000_001, 1_800_000_000_000_000_099])
    mono_times = iter([111, 199])
    monkeypatch.setattr(module, "time", SimpleNamespace(
        time_ns=lambda: next(wall_times), monotonic_ns=lambda: next(mono_times)
    ))

    def issue(target):
        entered.set()
        assert release.wait(2)

    robot.follower_r = SimpleNamespace(command_joint_pos=issue)
    target = np.r_[np.arange(6), 0.2]
    intent = np.r_[np.arange(6), 0.0]
    writer = threading.Thread(target=robot._command_follower, args=("right", target, intent))
    result = {}

    def read():
        reading.set()
        result.update(robot.get_last_commanded_snapshots())
        read_done.set()

    writer.start()
    assert entered.wait(2)
    reader = threading.Thread(target=read)
    reader.start()
    assert reading.wait(2)
    try:
        assert not read_done.wait(0.02)
    finally:
        release.set()
        writer.join(2)
        reader.join(2)
    assert not writer.is_alive() and not reader.is_alive()
    snapshot = result["follower_r"]
    target[:] = 80
    intent[:] = 90
    np.testing.assert_array_equal(snapshot["api_positions"], np.r_[np.arange(6), 0.2])
    np.testing.assert_array_equal(snapshot["positions"], np.r_[np.arange(6), 0.0])
    assert snapshot["command_timestamp_ns"] == 1_800_000_000_000_000_001
    assert snapshot["command_end_ns"] == 1_800_000_000_000_000_099
    assert snapshot["command_start_monotonic_ns"] == 111
    assert snapshot["command_end_monotonic_ns"] == 199
    snapshot["positions"][:] = -1
    assert robot.get_last_commanded_positions()["follower_r"][0] == 0
    assert result["follower_l"] == {"positions": None}


def test_failed_issue_preserves_previous_command_and_pause_hold_preserves_gripper_intent(controller_module):
    robot = controller(controller_module)
    sent = []
    robot.follower_l = SimpleNamespace(command_joint_pos=lambda command: sent.append(command.copy()))
    robot._command_follower("left", np.r_[np.ones(6), 0.1], np.r_[np.ones(6), 0.0])
    robot._command_follower("left", np.r_[np.full(6, 2), 0.4])
    before = robot.get_last_commanded_snapshots()["follower_l"]
    np.testing.assert_array_equal(before["positions"], np.r_[np.full(6, 2), 0.0])
    np.testing.assert_array_equal(sent[-1], np.r_[np.full(6, 2), 0.4])

    def fail(command):
        raise OSError("command API failed")

    robot.follower_l.command_joint_pos = fail
    with pytest.raises(OSError, match="command API failed"):
        robot._command_follower("left", np.ones(7))
    after = robot.get_last_commanded_snapshots()["follower_l"]
    assert after["command_timestamp_ns"] == before["command_timestamp_ns"]
    np.testing.assert_array_equal(after["positions"], before["positions"])


def _cached_driver():
    """Load the actual two pure snapshot methods without importing CAN/MuJoCo."""
    path = Path(__file__).parents[1] / "third_party/i2rt/i2rt/robots/motor_chain_robot.py"
    definition = next(node for node in ast.parse(path.read_text()).body
                      if isinstance(node, ast.ClassDef) and node.name == "MotorChainRobot")
    methods = [node for node in definition.body if isinstance(node, ast.FunctionDef)
               and node.name in ("get_observations", "get_timed_observations")]
    namespace = {"np": np, "Dict": dict, "Tuple": tuple, "Any": Any}
    # Only these local, reviewed methods are executed; module imports would load CAN/MuJoCo.
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    driver = SimpleNamespace(_state_lock=threading.Lock(), _gripper_index=6, temp_record_flag=False)
    for name in ("get_observations", "get_timed_observations"):
        setattr(driver, name, MethodType(namespace[name], driver))
    driver._joint_state = SimpleNamespace(
        pos=np.arange(7, dtype=float), vel=np.ones(7), eff=np.zeros(7), timestamp=1_800_000_000.125
    )
    return driver


def test_public_driver_accessor_copies_one_cached_state_and_reports_existing_host_time():
    driver = _cached_driver()
    observations, timing = driver.get_timed_observations()
    assert abs(timing["state_timestamp_ns"] - 1_800_000_000_125_000_000) <= 256
    assert timing["state_timestamp_valid"] is True
    assert timing["state_timestamp_source"] == "host_cached_feedback_read"
    driver._joint_state.pos[:] = -1
    np.testing.assert_array_equal(observations["joint_pos"], np.arange(6))
    np.testing.assert_array_equal(observations["gripper_pos"], [6])
    driver._joint_state.timestamp = float("nan")
    assert driver.get_timed_observations()[1] == {
        "state_timestamp_ns": -1, "state_timestamp_valid": False,
        "state_timestamp_source": "unavailable",
    }
    assert set(driver.get_observations()) == set(observations)


def test_cached_driver_state_replacement_during_read_cannot_mix_generations():
    driver = _cached_driver()
    driver.temp_record_flag = True
    old = driver._joint_state
    old.temp_mos = np.full(7, 31.0)
    old.temp_rotor = np.full(7, 32.0)
    replacement = SimpleNamespace(
        pos=np.full(7, 101.0), vel=np.full(7, 102.0), eff=np.full(7, 103.0),
        temp_mos=np.full(7, 104.0), temp_rotor=np.full(7, 105.0),
        timestamp=old.timestamp + 100,
    )

    class ReplacedDuringPositionRead:
        vel, eff = old.vel, old.eff
        temp_mos, temp_rotor = old.temp_mos, old.temp_rotor
        timestamp = old.timestamp

        @property
        def pos(self):
            driver._joint_state = replacement
            return old.pos

    driver._joint_state = ReplacedDuringPositionRead()
    observations, timing = driver.get_timed_observations()
    assert driver._joint_state is replacement
    np.testing.assert_array_equal(observations["joint_pos"], np.arange(6))
    np.testing.assert_array_equal(observations["joint_vel"], np.ones(6))
    np.testing.assert_array_equal(observations["joint_eff"], np.zeros(6))
    np.testing.assert_array_equal(observations["gripper_pos"], [6])
    np.testing.assert_array_equal(observations["temp_mos"], np.full(7, 31.0))
    np.testing.assert_array_equal(observations["temp_rotor"], np.full(7, 32.0))
    assert abs(timing["state_timestamp_ns"] - 1_800_000_000_125_000_000) <= 256


def test_recorder_preserves_integer_timing_unknowns_and_initial_fallback(tmp_path):
    robot = integrity.Controller(("follower_l",))
    recording = integrity.recorder.DemonstrationRecorder(
        cameras=[integrity.Camera("view")], robot_controller=robot,
        recording_dir=tmp_path, task_name="task", task_instruction="test",
        interface=SimpleNamespace(name="fake"),
    )
    origin = 1_800_000_000_000_000_001
    observation = robot.get_all_observations()
    command = np.r_[np.ones(6), 0.0]
    recording._robot_frames = [
        {"t": origin, "obs": observation, "cmd": {"follower_l": None}},
        {"t": origin + 100, "obs": observation, "cmd": {"follower_l": command},
         "readout_timestamp_ns": origin + 101, "readout_monotonic_ns": 101,
         "readout_end_ns": origin + 121, "readout_end_monotonic_ns": 121,
         "obs_timing": {"follower_l": {
             "state_timestamp_ns": origin + 11, "state_timestamp_valid": True,
             "state_timestamp_source": "host_cached_feedback_read",
             "read_start_ns": origin + 101, "read_end_ns": origin + 111,
             "read_start_monotonic_ns": 101, "read_end_monotonic_ns": 111}},
         "cmd_timing": {"follower_l": {
             "api_positions": np.r_[np.ones(6), 0.1],
             "command_timestamp_ns": origin - 20, "command_timestamp_valid": True,
             "command_timestamp_source": "host_command_api_issue", "command_end_ns": origin - 10,
             "command_start_monotonic_ns": 10, "command_end_monotonic_ns": 20}}},
    ]
    recording._save_robot_data()
    with np.load(tmp_path / "robot_data.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["follower_l_state_timestamp_ns"], [-1, origin + 11])
        np.testing.assert_array_equal(saved["follower_l_command_timestamp_ns"], [-1, origin - 20])
        np.testing.assert_array_equal(saved["follower_l_command_timestamp_valid"], [False, True])
        np.testing.assert_array_equal(saved["readout_timestamp_ns"], [-1, origin + 101])
        np.testing.assert_array_equal(saved["readout_end_ns"], [-1, origin + 121])
        np.testing.assert_array_equal(saved["readout_end_monotonic_ns"], [-1, 121])
        np.testing.assert_array_equal(saved["follower_l_joint_cmd"][0], np.r_[np.zeros(6), 1.0])
        assert np.isnan(saved["follower_l_joint_cmd_api"][0]).all()
        assert saved["follower_l_state_timestamp_ns"].dtype == np.int64
        assert saved["follower_l_command_timestamp_ns"].dtype == np.int64
        assert saved["follower_l_command_timestamp_valid"].dtype == np.bool_
        assert saved["follower_l_state_timestamp_source"].dtype.kind == "U"
        assert saved["readout_end_ns"].dtype == np.int64
        assert saved["readout_end_monotonic_ns"].dtype == np.int64
    assert "not motor acquisition" in recording.extra_metadata["robot_timing"]["state_timestamp_ns"]


def test_recorder_samples_timed_accessors_without_re_reading_legacy_positions(tmp_path, monkeypatch):
    robot = integrity.Controller(("follower_l",))
    observation = robot.get_all_observations()
    expected_state_time, expected_issue_time = 1_800_000_000_000_000_011, 1_800_000_000_000_000_007
    calls = []

    def timed_observations():
        calls.append("observations")
        return observation, {"follower_l": {
            "state_timestamp_ns": expected_state_time, "state_timestamp_valid": True,
            "state_timestamp_source": "host_cached_feedback_read"}}

    def command_snapshots():
        calls.append("commands")
        return {"follower_l": {
            "positions": np.zeros(7), "api_positions": np.zeros(7),
            "command_timestamp_ns": expected_issue_time, "command_timestamp_valid": True,
            "command_timestamp_source": "host_command_api_issue"}}

    robot.get_all_timed_observations = timed_observations
    robot.get_last_commanded_snapshots = command_snapshots
    robot.get_all_observations = lambda: pytest.fail("legacy observation re-read")
    robot.get_last_commanded_positions = lambda: pytest.fail("legacy command re-read")
    recording = integrity.recorder.DemonstrationRecorder(
        cameras=[integrity.Camera("view")], robot_controller=robot,
        recording_dir=tmp_path, task_name="task", task_instruction="test",
        interface=SimpleNamespace(name="fake"),
    )
    stop = threading.Event()
    wall_times = iter([expected_issue_time - 10, expected_state_time + 20])
    monotonic_ns = iter([10, 50])
    loop_times = iter([0.0, 0.001])

    def wall_clock():
        value = next(wall_times)
        calls.append(("wall", value))
        return value

    def monotonic_clock():
        value = next(monotonic_ns)
        calls.append(("monotonic", value))
        return value

    monkeypatch.setattr(integrity.recorder, "time", SimpleNamespace(
        time_ns=wall_clock, monotonic_ns=monotonic_clock,
        monotonic=lambda: next(loop_times),
    ))

    class OneRead:
        def is_set(self):
            return stop.is_set()

        def wait(self, delay):
            stop.set()

    recording._robot_loop(OneRead(), SimpleNamespace(get_current_timestamp_ns=lambda: expected_issue_time - 20))
    assert len(recording._robot_frames) == 1
    frame = recording._robot_frames[0]
    assert frame["obs_timing"]["follower_l"]["state_timestamp_ns"] == expected_state_time
    assert frame["cmd_timing"]["follower_l"]["command_timestamp_ns"] == expected_issue_time
    assert frame["t"] == expected_issue_time - 20
    assert frame["readout_timestamp_ns"] == expected_issue_time - 10
    assert frame["readout_end_ns"] == expected_state_time + 20
    assert frame["readout_end_monotonic_ns"] == 50
    assert calls == [
        ("wall", expected_issue_time - 10), ("monotonic", 10),
        "observations", "commands",
        ("wall", expected_state_time + 20), ("monotonic", 50),
    ]


def test_sim_reset_clears_command_position_and_timestamp_together(controller_module, monkeypatch):
    module = controller_module
    robot = module.RobotController(
        use_right_leader=False, use_left_leader=False, use_right_follower=False, sim="test:1"
    )
    robot.follower_l = SimpleNamespace(command_joint_pos=lambda _: None)
    robot._command_follower("left", np.ones(7))
    monkeypatch.setattr(robot, "stop_teleoperation", lambda: None)
    monkeypatch.setattr(robot, "move_to_home_positions", lambda: None)
    monkeypatch.setattr(module, "time", SimpleNamespace(sleep=lambda _: None))
    robot.reset_sim_episode()
    assert robot.get_last_commanded_snapshots()["follower_l"] == {"positions": None}
