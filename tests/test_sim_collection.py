"""Sim episode reset boundaries use fake controllers, cameras and recordings."""

import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from raiden import recorder


def check_collection(tmp_path):
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
        verdict_hint="",
        calibrating=False,
        setup=lambda _: events.append("interface_setup"),
        start=lambda _: events.append("teleop_start"),
        set_active_recording=lambda _: None,
        stop=lambda _: events.append("teleop_stop"),
    )

    class Recording:
        def __init__(self, *, recording_dir, **kwargs):
            self.path = recording_dir

        def start_recording(self, **kwargs):
            events.append("record_start")
            starts.append(controller.reset_count)

        def stop_recording(self, **kwargs):
            (self.path / "metadata.json").write_text('{"complete": true}')
            return self.path

        def discard(self):
            shutil.rmtree(self.path)

    action_log = SimpleNamespace(
        start=lambda: events.append("log_start"),
        save=lambda _: None,
        discard=lambda: None,
        close=lambda: None,
    )
    task = SimpleNamespace(
        status={"enabled": False}, poll=lambda **_: False, close=lambda: None
    )
    saved = []
    active = [None]
    with (
        patch.object(recorder, "RobotController", return_value=controller),
        patch.object(recorder, "DemonstrationRecorder", Recording),
        patch.object(
            recorder, "_wait_for_sim_action", side_effect=["save", "discard", "quit"]
        ),
        patch.object(recorder.sys, "stdin", SimpleNamespace(isatty=lambda: False)),
        patch("raiden.sim.ActionLog", return_value=action_log),
        patch("raiden.sim.TaskMonitor", return_value=task),
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
    assert starts == [1, 2, 3]
    assert (
        events.index("interface_setup")
        < events.index("reset1")
        < events.index("log_start")
    )
    assert len(saved) == 1 and result == saved[0]
    assert active == [None]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["0000"]


class SimCollectionTest(unittest.TestCase):
    def test_first_save_and_discard_starts_capture_after_fresh_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            check_collection(Path(directory))


if __name__ == "__main__":
    unittest.main()
