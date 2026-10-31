"""Sim episode reset boundaries use fake controllers, cameras and recordings."""

import json
import shutil
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from raiden import recorder
from raiden.sim.action_log import ActionLog


def check_collection(tmp_path, fail_method=None):
    events = []
    starts = []
    controller = SimpleNamespace(
        setup_for_teleop_recording=lambda: events.append("setup"),
        enable_estop=lambda: None,
        shutdown=lambda: events.append("shutdown"),
        reset_count=0,
    )

    def reset():
        controller.reset_count += 1
        events.append(f"reset{controller.reset_count}")

    controller.reset_sim_episode = reset
    interface = SimpleNamespace(
        uses_leaders=False,
        name="mock",
        verdict_hint="",
        calibrating=False,
        setup=lambda _: events.append("interface_setup"),
        start=lambda _: events.append("teleop_start"),
        set_active_recording=lambda _: None,
        stop=lambda _: events.append("teleop_stop"),
    )

    class Recording(recorder.DemonstrationRecorder):
        def __init__(self, *, recording_dir, **kwargs):
            super().__init__(recording_dir=recording_dir, **kwargs)
            self.path = recording_dir

        def start_recording(self, **kwargs):
            self.is_recording = True
            self._start_time = recorder.time.monotonic()
            self._camera_start_times_ns = {}
            events.append("record_start")
            starts.append(controller.reset_count)

        def _join_threads(self):
            pass

        def _finalize_cameras(self):
            pass

        def _save_robot_data(self):
            np.savez(self.path / "robot_data.npz", timestamps=np.array([1, 2], dtype=np.int64))

        def discard(self):
            self.is_recording = False
            shutil.rmtree(self.path)

    class LogConnection:
        def call(self, method):
            events.append(method)
            if method == fail_method:
                raise EOFError("native action log unavailable")
            if method == "stop_action_log":
                metadata = tmp_path / "0000/metadata.json"
                if metadata.exists() and events.count("stop_action_log") == 1:
                    assert not json.loads(metadata.read_text())["complete"]
                return {"cmd": np.zeros((2, 7)), "sim_time": np.array([.002, .004]),
                        "t_ns": np.array([1, 2], dtype=np.int64)}
            return None  # The legacy MuJoCo start RPC returns None on success.

        def close(self):
            pass

    with patch("raiden.sim.action_log.SimConnection", return_value=LogConnection()):
        action_log = ActionLog()
    task = SimpleNamespace(
        status={"enabled": False}, poll=lambda **_: False, close=lambda: None
    )
    saved = []
    active = [None]
    write_failure = (
        patch("raiden.sim.action_log.np.savez_compressed", side_effect=OSError("disk full"))
        if fail_method == "save_action_log" else nullcontext()
    )
    with (
        patch.object(recorder, "RobotController", return_value=controller),
        patch.object(recorder, "DemonstrationRecorder", Recording),
        patch.object(
            recorder, "_wait_for_sim_action", side_effect=["save", "discard", "quit"]
        ),
        patch.object(recorder.sys, "stdin", SimpleNamespace(isatty=lambda: False)),
        patch("raiden.sim.ActionLog", return_value=action_log),
        patch("raiden.sim.TaskMonitor", return_value=task),
        write_failure,
    ):
        result = recorder._run_sim_episodes(
            [],
            interface,
            "fake",
            tmp_path,
            "new_task",
            "test",
            None,
            saved.append,
            active,
        )
    if fail_method:
        assert not saved and result is None and active == [None]
        failed = tmp_path / "0000"
        if fail_method == "start_action_log":
            assert not starts and not (failed / "metadata.json").exists()
        else:
            meta = json.loads((failed / "metadata.json").read_text())
            assert not meta["complete"] and meta["preserve_incomplete"]
            assert ("OSError" if fail_method == "save_action_log" else "EOFError") in meta["finalization_error"]
            assert (failed / "robot_data.npz").exists()
            assert recorder._next_recording_dir(tmp_path).name == "0001"
            assert (failed / "robot_data.npz").exists()
        return
    assert starts == [1, 2, 3]
    assert (
        events.index("interface_setup")
        < events.index("reset1")
        < events.index("start_action_log")
    )
    assert len(saved) == 1 and result == saved[0]
    assert active == [None]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["0000"]


class SimCollectionTest(unittest.TestCase):
    def test_operator_verdict_is_sent_only_when_the_server_supports_it(self):
        calls = []

        def call(method, **kwargs):
            calls.append((method, kwargs))
            if method == "start_action_log":
                return {"metadata": {"native_attempts": {"accepts_outcome": True}}}

        with patch("raiden.sim.action_log.SimConnection", return_value=SimpleNamespace(call=call)):
            log = ActionLog()
            log.start()
            log.discard()
        self.assertEqual(calls[-1], ("stop_action_log", {"outcome": "operator_failure"}))

    def test_physics_level_selection_restores_pause_and_rejects_hardware(self):
        from raiden.sim.client import select_physics_level

        with self.assertRaisesRegex(ValueError, "requires --sim"):
            select_physics_level("", "L2")
        calls = []
        connection = SimpleNamespace(close=lambda: calls.append("close"))

        def call(method, **kwargs):
            calls.append((method, kwargs))
            if method == "is_paused":
                return False
            if method == "set_ablation_level":
                return {"level": "L2", "values_sha256": "a" * 64}

        connection.call = call
        with patch("raiden.sim.client.SimConnection", return_value=connection):
            self.assertEqual(select_physics_level("127.0.0.1:5599", "L2")["level"], "L2")
        self.assertEqual(calls[-2:], [("set_paused", {"paused": False}), "close"])

    def test_sim_follower_preserves_profile_gains_at_controller_startup(self):
        from raiden.sim.follower import SimFollower

        gains = np.arange(6) + 1.
        calls = []

        def call(method, **kwargs):
            if method == "get_kp_kd":
                return gains, gains * .1
            if method == "get_metadata":
                return {"physics_ablation": {"level": "L3"}}
            calls.append((method, kwargs))

        with patch("raiden.sim.follower.SimConnection", return_value=SimpleNamespace(call=call)):
            follower = SimFollower()
            follower.update_kp_kd(np.ones(7) * 80, np.ones(7) * 5)
        np.testing.assert_array_equal(calls[-1][1]["kp"][:6], gains)
        np.testing.assert_array_equal(calls[-1][1]["kd"][:6], gains * .1)

    def test_first_save_and_discard_starts_capture_after_fresh_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            check_collection(Path(directory))

    def test_log_start_and_stop_failures_never_publish_success(self):
        for method in ("start_action_log", "stop_action_log", "save_action_log"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                check_collection(Path(directory), method)

    def test_action_log_rpc_failures_propagate(self):
        def fail(method):
            raise EOFError(method)

        connection = SimpleNamespace(call=fail)
        with patch("raiden.sim.action_log.SimConnection", return_value=connection):
            log = ActionLog()
        with self.assertRaises(EOFError):
            log.start()
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(EOFError):
            log.save(Path(directory))

    def test_action_log_requires_native_command_arrays(self):
        for payload in (None, {}, {"metadata": {"backend": "isaac"}}):
            with self.subTest(payload=payload):
                connection = SimpleNamespace(call=lambda method, payload=payload: payload)
                with patch("raiden.sim.action_log.SimConnection", return_value=connection):
                    log = ActionLog()
                with tempfile.TemporaryDirectory() as directory, self.assertRaises((RuntimeError, ValueError)):
                    log.save(Path(directory))


if __name__ == "__main__":
    unittest.main()
