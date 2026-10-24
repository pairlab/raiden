"""Recording integrity checks use isolated SDK-free cameras and controllers."""

import importlib.util
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np


def _load_recorder():
    name = "_recording_integrity_recorder"
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "raiden/recorder.py"
    )
    module = importlib.util.module_from_spec(spec)
    stubs = {
        name: module,
        "boto3": SimpleNamespace(),
        "raiden.cameras": SimpleNamespace(Camera=object),
        "raiden.control": SimpleNamespace(TeleopInterface=object),
        "raiden.db.database": SimpleNamespace(get_db=lambda: None),
        "raiden.robot.controller": SimpleNamespace(
            RobotController=object, _binary_gripper=lambda value: float(value >= 0.5)
        ),
        "raiden.utils": SimpleNamespace(fzf_select=lambda *args, **kwargs: []),
    }
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


recorder = _load_recorder()


class Camera:
    recording_extension = "native"

    def __init__(self, name, *, start_error=False, grab_error=None, lose_after=None,
                 stop_error=False, close_error=False, write_artifact=True):
        self.name = name
        self.start_error = start_error
        self.grab_error = grab_error
        self.lose_after = lose_after
        self.stop_error = stop_error
        self.close_error = close_error
        self.write_artifact = write_artifact
        self.starts = self.stops = self.grabs = self.closes = 0

    def start_recording(self, path):
        self.starts += 1
        if self.write_artifact:
            path.write_bytes(b"recoverable native recording")
        if self.start_error:
            raise OSError("camera start failed after opening writer")

    def stop_recording(self):
        self.stops += 1
        if self.stop_error:
            raise OSError("camera finalization failed")

    def close(self):
        self.closes += 1
        if self.close_error:
            raise OSError("camera close failed")

    def grab(self):
        time.sleep(0.002)
        self.grabs += 1
        if self.grab_error:
            raise self.grab_error
        return self.lose_after is None or self.grabs <= self.lose_after

    def get_current_timestamp_ns(self):
        return time.time_ns()


class Controller:
    def __init__(self, arms=("follower_l", "follower_r"), *, missing=None,
                 bad_state=False, bad_command=False, no_command=None,
                 sample_error=False, sim=False):
        self.arms = list(arms)
        self.missing = missing
        self.bad_state = bad_state
        self.bad_command = bad_command
        self.no_command = no_command
        self.sample_error = sample_error
        self.sim = sim
        self.shutdowns = 0

    def recording_config(self):
        return {
            "mode": "sim" if self.sim else "real",
            "active_followers": self.arms,
            "state_order": [f"joint_{i}" for i in range(1, 7)] + ["gripper_aperture"],
            "command_order": [f"joint_{i}" for i in range(1, 7)] + ["gripper_intent"],
        }

    def get_all_observations(self):
        if self.sample_error:
            raise OSError("robot connection lost")
        return {
            arm: {"joint_pos": np.zeros(5 if self.bad_state else 6),
                  "gripper_pos": np.array([0.5])}
            for arm in self.arms if arm != self.missing
        }

    def get_last_commanded_positions(self):
        return {
            arm: None if arm == self.no_command else np.zeros(6 if self.bad_command else 7)
            for arm in self.arms
        }

    def shutdown(self):
        self.shutdowns += 1


class RecordingIntegrityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.episode = self.root / "0000"

    def tearDown(self):
        self.temp.cleanup()

    def make(self, cameras=None, controller=None, camera_config=None):
        return recorder.DemonstrationRecorder(
            cameras=cameras if cameras is not None else [Camera("custom_view")],
            robot_controller=controller if controller is not None else Controller(),
            recording_dir=self.episode,
            task_name="task", task_instruction="do the task",
            interface=SimpleNamespace(name="fake"),
            extra_metadata={"recording_config": {"cameras": camera_config or {}}},
        )

    def wait_for(self, predicate):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.002)
        self.fail("recording worker did not reach the expected state")

    def metadata(self):
        return json.loads((self.episode / "metadata.json").read_text())

    def assert_retained(self):
        meta = self.metadata()
        self.assertFalse(meta["complete"])
        self.assertTrue(meta["preserve_incomplete"])
        self.assertEqual(recorder._peek_next_episode(self.root), "0001")
        self.assertEqual(recorder._next_recording_dir(self.root).name, "0001")
        self.assertTrue(self.episode.exists())

    def test_parallel_start_failure_propagates_and_finalizes_every_attempt(self):
        cameras = [Camera("scene"), Camera("wrist", start_error=True)]
        recording = self.make(cameras)
        with self.assertRaisesRegex(RuntimeError, "wrist start.*camera start failed"):
            recording.start_recording()
        self.assertFalse(recording.is_recording)
        self.assertEqual([(c.starts, c.stops) for c in cameras], [(1, 1), (1, 1)])
        self.assertTrue((self.episode / "cameras/wrist.native").exists())
        self.assert_retained()

    def test_sampling_failure_and_camera_loss_stop_workers_and_never_complete(self):
        cases = [
            ([Camera("view", grab_error=OSError("camera disconnected"))], Controller(), "camera disconnected"),
            ([Camera("view", lose_after=2)], Controller(), "camera stream lost"),
            ([Camera("view")], Controller(sample_error=True), "robot connection lost"),
        ]
        for index, (cameras, controller, message) in enumerate(cases):
            with self.subTest(message=message):
                self.episode = self.root / f"{index:04d}"
                recording = self.make(cameras, controller)
                recording.start_recording()
                self.wait_for(recording._stop_event.is_set)
                with self.assertRaisesRegex(RuntimeError, message):
                    recording.raise_if_failed()
                with self.assertRaisesRegex(RuntimeError, message):
                    recording.stop_recording(shutdown_robots=False)
                self.assertFalse(self.metadata()["complete"])
                self.assertTrue(self.metadata()["preserve_incomplete"])
                self.assertTrue((self.episode / "robot_data.npz").exists())
                self.assertLess(cameras[0].grabs, 20)

    def test_configured_missing_arm_or_invalid_shapes_cannot_complete(self):
        cases = [
            (Controller(missing="follower_r"), "Missing observations.*follower_r"),
            (Controller(bad_state=True), "Invalid follower_l joint_pos"),
            (Controller(bad_command=True), "Invalid follower_l command"),
        ]
        for index, (controller, message) in enumerate(cases):
            with self.subTest(message=message):
                self.episode = self.root / f"{index:04d}"
                recording = self.make(controller=controller)
                recording.start_recording()
                self.wait_for(recording._stop_event.is_set)
                with self.assertRaisesRegex(RuntimeError, message):
                    recording.stop_recording(shutdown_robots=False)
                self.assertFalse(self.metadata()["complete"])

    def test_configured_camera_missing_is_rejected_before_any_writer_starts(self):
        camera = Camera("scene")
        recording = self.make([camera], camera_config={"scene": {}, "right_wrist": {}})
        with self.assertRaisesRegex(RuntimeError, "right_wrist"):
            recording.start_recording()
        self.assertEqual(camera.starts, 0)
        self.assert_retained()

    def test_missing_camera_artifact_and_stop_errors_cannot_complete(self):
        for index, camera in enumerate((Camera("view", write_artifact=False), Camera("view", stop_error=True))):
            with self.subTest(camera=camera):
                self.episode = self.root / f"{index:04d}"
                recording = self.make([camera])
                recording.start_recording()
                self.wait_for(lambda recording=recording, camera=camera: recording._robot_frames and camera.grabs)
                with self.assertRaisesRegex(RuntimeError, "missing or empty|finalization failed"):
                    recording.stop_recording(shutdown_robots=False)
                self.assertFalse(self.metadata()["complete"])

    def test_both_arms_complete_and_snapshot_is_portable(self):
        cameras = [Camera(name) for name in ("overhead", "left_tool", "right_tool")]
        snapshot = {
            name: {"role": role, "type": "realsense", "policy_crop": [0, 0, 4, 3]}
            for name, role in zip((c.name for c in cameras), ("scene", "left_wrist", "right_wrist"))
        }
        controller = Controller()
        recording = self.make(cameras, controller, snapshot)
        recording.start_recording()
        self.wait_for(lambda: len(recording._robot_frames) >= 2 and all(c.grabs for c in cameras))
        recording.stop_recording()
        meta = self.metadata()
        self.assertTrue(meta["complete"])
        self.assertNotIn("preserve_incomplete", meta)
        self.assertEqual(controller.shutdowns, 1)
        self.assertEqual(meta["recording_config"], {
            "schema_version": 1, "robot": controller.recording_config(), "cameras": snapshot,
        })
        with np.load(self.episode / "robot_data.npz") as data:
            for arm in controller.arms:
                self.assertEqual(data[f"{arm}_joint_pos_7d"].shape[1], 7)
                self.assertEqual(data[f"{arm}_joint_cmd"].shape, data[f"{arm}_joint_pos_7d"].shape)

    def test_single_arm_sim_and_custom_camera_do_not_require_bimanual_views(self):
        recording = self.make([Camera("custom_sim_view")], Controller(("follower_l",), sim=True))
        recording.start_recording()
        self.wait_for(lambda: recording._robot_frames and recording.cameras[0].grabs)
        recording.stop_recording(shutdown_robots=False)
        self.assertTrue(self.metadata()["complete"])

    def test_never_commanded_arm_retains_measured_data_and_incomplete_metadata(self):
        recording = self.make(controller=Controller(no_command="follower_r"))
        recording.start_recording()
        self.wait_for(lambda: recording._robot_frames and recording.cameras[0].grabs)
        with self.assertRaisesRegex(RuntimeError, "No commanded positions recorded for follower_r"):
            recording.stop_recording(shutdown_robots=False)
        with np.load(self.episode / "robot_data.npz") as data:
            self.assertIn("follower_l_joint_pos", data.files)
            self.assertIn("follower_r_joint_pos", data.files)
            self.assertIn("follower_l_joint_cmd", data.files)
        self.assert_retained()

    def test_camera_snapshot_uses_configured_role_and_crop_only(self):
        path = self.root / "camera.json"
        path.write_text(json.dumps({
            "named_view": {"serial": "portable-device", "type": "realsense",
                           "role": "left_wrist", "crop": [10, 20, 640, 480],
                           "color_controls": "auto", "fps": 30},
            "legacy": 1234,
        }))
        saved = recorder._camera_recording_config(recorder.CameraConfig(str(path)), ["named_view", "legacy"])
        self.assertEqual(saved, {
            "named_view": {"role": "left_wrist", "type": "realsense", "policy_crop": [10, 20, 640, 480]},
            "legacy": {"role": None, "type": "zed", "policy_crop": None},
        })

    def test_foreground_sim_wait_surfaces_worker_failure_without_operator_input(self):
        recording = self.make()
        recording._record_failure("camera view", OSError("lost"))
        with self.assertRaisesRegex(RuntimeError, "lost"):
            recorder._wait_for_sim_action(None, None, None, False, recording)

    def test_worker_stop_timeout_is_retained_and_never_complete(self):
        recording = self.make()
        recording.start_recording()
        self.wait_for(lambda: recording._robot_frames and recording.cameras[0].grabs)
        stuck = SimpleNamespace(name="stuck-worker", join=lambda timeout: None, is_alive=lambda: True)
        recording._threads.append(stuck)
        with self.assertRaisesRegex(RuntimeError, "stuck-worker.*did not stop"):
            recording.stop_recording(shutdown_robots=False)
        self.assert_retained()

    def test_interrupted_worker_join_still_shuts_down_and_saves_raw_commands(self):
        for index, error in enumerate((RuntimeError("join failed"), KeyboardInterrupt())):
            with self.subTest(error=type(error).__name__):
                self.episode = self.root / f"{index:04d}"
                controller = Controller()
                recording = self.make(controller=controller)
                recording.start_recording()
                self.wait_for(lambda recording=recording: recording._robot_frames and recording.cameras[0].grabs)
                with (
                    patch.object(recording, "_join_threads", side_effect=error),
                    self.assertRaises(type(error)) as caught,
                ):
                    recording.stop_recording()
                for thread in recording._threads:
                    thread.join()
                self.assertIs(caught.exception, error)
                self.assertEqual(controller.shutdowns, 1)
                self.assertEqual(recording.cameras[0].stops, 1)
                self.assertFalse(self.metadata()["complete"])
                self.assertTrue(self.metadata()["preserve_incomplete"])
                with np.load(self.episode / "robot_data.npz") as data:
                    for arm in controller.arms:
                        self.assertIn(f"{arm}_joint_cmd", data.files)

    def test_discard_does_not_delete_an_attempt_with_a_failed_camera_stop(self):
        recording = self.make([Camera("view", stop_error=True)])
        recording.start_recording()
        self.wait_for(lambda: recording._robot_frames and recording.cameras[0].grabs)
        with self.assertRaisesRegex(RuntimeError, "camera finalization failed"):
            recording.discard()
        self.assert_retained()

    def test_session_cleanup_attempts_each_resource_once_after_failures(self):
        for stage in ("quit", "setup", "recording_stop", "camera_close"):
            with self.subTest(stage=stage):
                cameras = [Camera("left_view", close_error=stage == "camera_close"), Camera("right_view")]
                config = self.root / f"{stage}_cameras.json"
                config.write_text(json.dumps({camera.name: 1234 for camera in cameras}))
                controller = Controller()
                controller.setup_for_teleop_recording = Mock(
                    side_effect=ValueError("initial homing failed") if stage == "setup" else None
                )
                controller.shutdown = Mock(
                    side_effect=None if stage == "camera_close" else OSError("shutdown homing failed")
                )
                controller.session_estop_requested = False
                controller.enable_estop = Mock()
                interface = SimpleNamespace(
                    name="fake", uses_leaders=False, waits_for_button_start=True,
                    ready_hint="", open=Mock(), close=Mock(), setup=Mock(),
                    start_ready=Mock(return_value=False), start=Mock(),
                    set_active_recording=Mock(), poll_success=Mock(return_value=True),
                    poll_failure=Mock(return_value=False),
                )
                ui = SimpleNamespace(
                    start=Mock(), setup_session=Mock(return_value=("task", "test", 1)),
                    monitor=Mock(return_value=None), attach_cameras=Mock(),
                    set_phase=Mock(), pause_pump=Mock(), close=Mock(),
                )
                if stage == "camera_close":
                    interface.close.side_effect = OSError("interface close failed")
                    ui.close.side_effect = OSError("UI close failed")
                db = SimpleNamespace(
                    snapshot_camera_config=Mock(return_value=1),
                    get_task_by_name=Mock(return_value=None),
                    get_latest_calibration_result=Mock(return_value=None),
                )
                with (
                    patch.object(recorder, "get_db", return_value=db),
                    patch.object(recorder, "RobotController", return_value=controller),
                    patch.object(recorder, "load_cameras_from_config", return_value=cameras),
                    patch.object(recorder, "_wait_for_start_or_quit", return_value=stage != "recording_stop"),
                    patch.object(recorder.signal, "signal"),
                    patch.object(recorder.sys, "stdin", SimpleNamespace(fileno=lambda: 0)),
                    patch.dict(sys.modules, {"raiden.record_ui": SimpleNamespace(RecordUI=lambda ui=ui, **_: ui)}),
                ):
                    arguments = {
                        "s3_bucket": None, "s3_prefix": "", "interface": interface,
                        "camera_config_file": str(config), "data_dir": str(self.root / stage), "ui": True,
                    }
                    if stage == "recording_stop":
                        recorder.run_recording(**arguments)
                    else:
                        message = "camera close failed" if stage == "camera_close" else "shutdown homing failed"
                        with self.assertRaisesRegex(OSError, message) as caught:
                            recorder.run_recording(**arguments)
                        if stage == "camera_close":
                            self.assertTrue(any("interface close failed" in note for note in caught.exception.__notes__))
                            self.assertTrue(any("UI close failed" in note for note in caught.exception.__notes__))
                controller.shutdown.assert_called_once_with()
                self.assertEqual([camera.closes for camera in cameras], [1, 1])
                interface.close.assert_called_once_with()
                ui.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
