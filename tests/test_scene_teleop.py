"""Arm routing, native scene calibration, and owned launcher lifetime."""

import sys
from unittest.mock import Mock

import numpy as np
import pytest

from raiden.sim.calibration import sim_calibration
from raiden.sim.client import require_arm_mode
from raiden.sim.follower import SimFollower
from raiden.sim.launch import IsaacServer


class Connection:
    def __init__(self, rig=None):
        self.rig = rig if rig is not None else {"arms": ["left", "right"]}
        self.calls = []
        self.closed = False

    def call(self, method, **kwargs):
        self.calls.append((method, kwargs))
        if method == "get_rig":
            return self.rig
        if method == "get_kp_kd":
            return np.full(6, 50.0), np.full(6, 5.0)
        if method == "get_joint_pos":
            return np.arange(7)
        if method == "get_observations":
            return {"joint_pos": np.arange(6)}
        if method == "get_timed_observations":
            return {"joint_pos": np.arange(6)}, {"state_timestamp_valid": True}

    def close(self):
        self.closed = True


@pytest.mark.parametrize("arm", ["left", "right"])
def test_follower_routes_all_arm_operations_and_preserves_scene_gains(monkeypatch, arm):
    connection = Connection()
    monkeypatch.setattr("raiden.sim.follower.SimConnection", lambda _: connection)
    follower = SimFollower(arm=arm)
    assert follower.get_kp_kd()[0][:6].tolist() == [50.0] * 6
    assert follower.get_joint_pos().shape == (7,)
    follower.get_observations()
    assert follower.get_timed_observations()[1]["state_timestamp_valid"]
    follower.command_joint_pos(np.zeros(7))
    follower.update_kp_kd(np.ones(7), np.ones(7))
    for method, kwargs in connection.calls:
        if method != "get_rig":
            assert kwargs["arm"] == arm
    follower.close()
    assert connection.closed


def test_legacy_single_arm_remains_supported_and_cannot_become_bimanual(monkeypatch):
    connection = Connection({"cameras": {}})
    monkeypatch.setattr("raiden.sim.follower.SimConnection", lambda _: connection)
    follower = SimFollower()
    assert not follower.get_timed_observations()[1]["state_timestamp_valid"]
    assert all("arm" not in kwargs for _, kwargs in connection.calls)
    with pytest.raises(ValueError, match="cannot connect"):
        SimFollower(arm="right")
    assert connection.closed


def test_failed_follower_initialization_closes_connection(monkeypatch):
    connection = Connection()
    connection.call = Mock(side_effect=RuntimeError("bad scene"))
    monkeypatch.setattr("raiden.sim.follower.SimConnection", lambda _: connection)
    with pytest.raises(RuntimeError, match="bad scene"):
        SimFollower()
    assert connection.closed


@pytest.mark.parametrize(
    "mode,arms", [("single", ["left", "right"]), ("bimanual", ["left"])]
)
def test_mode_mismatch_fails_before_input_device_startup_and_closes(
    monkeypatch, mode, arms
):
    connection = Connection({"arms": arms})
    monkeypatch.setattr("raiden.sim.client.SimConnection", lambda _: connection)
    with pytest.raises(ValueError, match="same --arms"):
        require_arm_mode("fake:5599", mode)
    assert connection.closed


def test_native_wrist_calibration_uses_recorded_poses_and_interbase_transform():
    transform = {"T_left_from_right": np.eye(4).tolist()}
    rig = {
        "backend": "isaac",
        "bimanual_transform": transform,
        "cameras": {
            "scene_camera": {"T_base_cam": np.eye(4).tolist()},
            "right_wrist_camera": {
                "dynamic": True,
                "pose_source": "native per-frame camera pose",
            },
        },
    }
    calibration = sim_calibration(rig)
    assert "table" not in calibration
    assert calibration["bimanual_transform"] == transform
    assert calibration["cameras"]["right_wrist_camera"] == {
        "pose_source": "recorded_camera_poses",
        "reference_frame": "left_arm_base",
    }


def test_launcher_preserves_venv_path_and_cleans_failed_spawn(tmp_path, monkeypatch):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/raiden_sim_server.py").touch()
    scene = tmp_path / "scene.json"
    scene.touch()
    python = tmp_path / "venv-python"
    python.symlink_to(sys.executable)
    server = IsaacServer(
        scene=scene,
        arms="bimanual",
        project=tmp_path,
        python=python,
        port=5698,
        log=tmp_path / "server.log",
    )
    assert server.command[0] == str(python)
    assert server.command[server.command.index("--arms") + 1] == "bimanual"
    probe = Mock()
    probe.__enter__ = Mock(return_value=probe)
    probe.__exit__ = Mock()
    monkeypatch.setattr("raiden.sim.launch.socket.socket", lambda: probe)
    monkeypatch.setattr(
        "raiden.sim.launch.subprocess.Popen", Mock(side_effect=OSError("spawn failed"))
    )
    with pytest.raises(OSError, match="spawn failed"):
        server.__enter__()
    assert server._stream is None
