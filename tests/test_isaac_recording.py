"""Portable Isaac recordings convert without importing an Isaac or MuJoCo runtime."""
from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from raiden.converter import (
    ConversionError,
    _extract_simrec,
    _load_sim_log,
    _sim_base_frame,
    convert_recording,
)
from raiden.sim.action_log import ActionLog
from raiden.sim.camera import SimCamera
from raiden.sim.serialization import jsonable, write_json

SPEC = importlib.util.spec_from_file_location("isaac_replay", Path(__file__).parents[1] / "scripts/isaac_replay.py")
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


class Connection:
    def __init__(self, address=""):
        self.index = 0
        self.render_arguments = []

    def close(self):
        pass

    def call(self, method, **kwargs):
        if method == "get_camera_info":
            return {"backend": "isaac", "serial": "sim-test", "width": 4, "height": 3,
                    "K": [[5, 0, 2], [0, 5, 1.5], [0, 0, 1]], "recording_mode": "frames"}
        if method == "render":
            self.render_arguments.append(kwargs)
            index = self.index
            self.index += 1
            t_ns = 1_800_000_000_000_000_000 + index * 33_333_333
            state = {"backend": "isaac", "state_version": 1, "sim_time": index / 30,
                     "physics_dt": .002, "cmd": [0.] * 6 + [1.],
                     "scene": {"articulation": {"robot": {"joint_position": [[0.] * 8]}}}}
            return {"scene_camera": {"rgb": np.full((3, 4, 3), [1, 2, 3], dtype=np.uint8),
                    "t_ns": t_ns, "frame_id": index, "T_base_cam": np.eye(4),
                    "sim": {"state": state, "objects": {"croissant": np.eye(4)},
                            "subtasks_done": [], "success": False}}}
        raise AssertionError(method)


class IsaacRecordingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.simrec = self.root / "cameras/scene_camera.simrec"

    def tearDown(self):
        self.temp.cleanup()

    def record(self, mode="frames"):
        with patch("raiden.sim.camera.SimConnection", Connection), patch("raiden.sim.camera.time.sleep"):
            camera = SimCamera("scene_camera", record_mode=mode)
            camera.open()
            camera.start_recording(self.simrec)
            for _ in range(3):
                self.assertTrue(camera.grab())
            camera.close()
        (self.root / "metadata.json").write_text(json.dumps({"complete": True, "converted": False,
            "task_name": "isaac_smoke", "task_instruction": "test", "camera_fps": 30,
            "cameras": ["scene_camera"], "duration_s": .1}))
        (self.root / "calibration_results.json").write_text(json.dumps({"cameras": {},
            "table": {"T_base_table": np.eye(4).tolist()}}))
        (self.root / "sim_metadata.json").write_text(json.dumps({"backend": "isaac", "physics_dt": .002,
            "twin_input_sha256": {"arm.xml": "original"}, "scene": {"calibration_sha256": "original"}}))

    def test_record_and_convert_exact_pixels_poses_and_full_state(self):
        self.record()
        self.assertFalse((self.simrec / "sim_state.npz").exists())
        self.assertIsNone(_sim_base_frame([self.simrec], self.root / "sim_model.xml", {}))
        counts = convert_recording(str(self.root))
        self.assertEqual(counts, {"scene_camera": 3})
        image = cv2.imread(str(self.root / "0000/rgb/scene_camera/0000000000.png"))
        np.testing.assert_array_equal(image[0, 0], [3, 2, 1])
        np.testing.assert_array_equal(np.load(self.root / "0000/rgb/scene_camera/camera_poses.npy"),
                                      np.repeat(np.eye(4)[None], 3, axis=0))
        log = _load_sim_log([self.simrec])
        self.assertEqual(log["state"][0]["backend"], "isaac")
        self.assertEqual(log["state"][0]["scene"]["articulation"]["robot"]["joint_position"], [[0.] * 8])
        self.assertTrue((self.root / "0000/sim_metadata.json").exists())

    def test_isaac_camera_waits_for_each_native_frame_id(self):
        with patch("raiden.sim.camera.SimConnection", Connection), patch("raiden.sim.camera.time.sleep"):
            camera = SimCamera("scene_camera")
            camera.open()
            self.assertEqual(camera.backend, "isaac")
            camera.start_recording(self.simrec)
            for _ in range(3):
                self.assertTrue(camera.grab())
            self.assertEqual([args["after_frame_id"] for args in camera._c.render_arguments], [None, 0, 1])
            camera.close()

    def test_state_only_conversion_requires_isaac_rerender(self):
        self.record("states")
        self.assertFalse((self.simrec / "rgb").exists())
        with self.assertRaisesRegex(ConversionError, "isaac_replay.py"):
            _extract_simrec(self.simrec, self.root / "rgb", self.root / "depth", self.root / "sim_model.xml", None)

    def test_replay_rejects_wrong_scene_or_calibration(self):
        self.record()
        metadata = json.loads((self.root / "sim_metadata.json").read_text())
        metadata["implementation_sha256"] = {"runtime.py": "original"}
        (self.root / "sim_metadata.json").write_text(json.dumps(metadata))
        class ImplementationMismatch:
            def call(self, method):
                return {**metadata, "implementation_sha256": {"runtime.py": "changed"}}
        with self.assertRaisesRegex(ValueError, "implementation_sha256"):
            REPLAY.validate_server(ImplementationMismatch(), self.root)
        class Live:
            def call(self, method):
                return {**metadata, "scene": {"calibration_sha256": "different"}}
        with self.assertRaisesRegex(ValueError, "scene"):
            REPLAY.validate_server(Live(), self.root)
        REPLAY.validate_server(Live(), self.root, allow_scene_change=True)

    def test_replay_rejects_other_physics_level(self):
        self.record()
        metadata = json.loads((self.root / "sim_metadata.json").read_text())
        metadata["physics_ablation"] = {"level": "L2", "compiled_sha256": "recorded"}
        write_json(self.root / "sim_metadata.json", metadata)

        class OtherLevel:
            def call(self, method):
                return {**metadata, "physics_ablation": {"level": "L3", "compiled_sha256": "other"}}

        with self.assertRaisesRegex(ValueError, "physics_ablation"):
            REPLAY.validate_server(OtherLevel(), self.root)

    def test_replay_does_not_assume_unbound_legacy_data_matches_a_level(self):
        self.record()
        metadata = json.loads((self.root / "sim_metadata.json").read_text())

        class BoundServer:
            def call(self, method):
                return {**metadata, "physics_ablation": {"level": "L3"}}

        with self.assertRaisesRegex(ValueError, "physics_ablation"):
            REPLAY.validate_server(BoundServer(), self.root)

    def test_replay_accepts_other_reset_realizations_but_keeps_scene_identity(self):
        self.record("states")
        metadata = json.loads((self.root / "sim_metadata.json").read_text())
        metadata["scene"].update(
            asset_physics={"manifest_sha256": "approved"},
            initial_state={"placement_randomization": {"sample_index": 1, "x": .5}},
            physical_initial_state={"croissant": {"position": [.5, .2, .9]}},
        )
        write_json(self.root / "sim_metadata.json", metadata)
        live = copy.deepcopy(metadata)
        live["scene"].update(
            initial_state={"placement_randomization": {"sample_index": 2, "x": .6}},
            physical_initial_state={"croissant": {"position": [.6, .1, .9]}},
        )

        class Server:
            def call(self, method):
                return live

        REPLAY.validate_server(Server(), self.root)
        live["scene"]["initial_state"] = {"source": "portable_state", "reset_context": {}}
        REPLAY.validate_server(Server(), self.root)
        live["scene"]["asset_physics"]["manifest_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "scene"):
            REPLAY.validate_server(Server(), self.root)

    def test_native_state_and_multiview_roundtrip(self):
        dt, initial_time = .002, .2
        tick_times = initial_time + dt * np.arange(4)
        timestamps = 1_800_000_000_000_000_000 + np.arange(4, dtype=np.int64) * 2_000_000
        commands = np.column_stack((np.arange(18).reshape(3, 6) / 100, [.75, .5, 0.]))
        states = []
        for index, (sim_time, t_ns) in enumerate(zip(tick_times, timestamps, strict=True)):
            target = commands[max(0, index - 1)] if index else np.array([0.] * 6 + [1.])
            root_pose = np.array([[0., 0., 0., 0., 0., 0., 1.]])
            states.append({
                "backend": "isaac", "state_version": 1, "physics_dt": dt,
                "sim_time": sim_time, "t_ns": t_ns, "monotonic_ns": t_ns - 100,
                "cmd": target, "requested_cmd": target, "kp": np.ones(6) * 80,
                "kd": np.ones(6) * 4, "command_sequence": index,
                "joint_names": {"robot": [f"joint_{j}" for j in range(6)] + ["finger", "finger_mirror"],
                                "oven": ["door"]},
                "reset_context": {"placement_randomization": {"seed": 7, "sample_index": 2}},
                "motor": {"computed_effort_Nm": np.arange(8), "applied_effort_Nm": np.arange(8) / 2},
                "scene": {
                    "articulation": {
                        "robot": {"root_pose": root_pose, "root_velocity": np.zeros((1, 6)),
                                  "joint_position": np.array([list(target[:6]) + [.04 * target[6], -.04 * target[6]]]),
                                  "joint_velocity": np.ones((1, 8)) * index / 100},
                        "oven": {"root_pose": root_pose + [[.7, .3, .9, 0, 0, 0, 0]],
                                 "root_velocity": np.zeros((1, 6)), "joint_position": [[index / 10]],
                                 "joint_velocity": [[.1]]},
                    },
                    "rigid_object": {"croissant": {
                        "root_pose": root_pose + [[.4 + index / 100, .2, .92, 0, 0, 0, 0]],
                        "root_velocity": np.ones((1, 6)) * index / 100,
                    }},
                },
            })
        payload = {
            "initial_state": states[0], "states": np.array(states, dtype=object),
            "state_t_ns": timestamps, "state_sim_time": tick_times,
            "cmd": commands, "requested_cmd": commands.copy(), "sim_time": tick_times[1:],
            "t_ns": timestamps[1:] - 1, "step_monotonic_ns": timestamps[1:] - 100,
            "command_sequence": np.arange(1, 4, dtype=np.int64),
            "command_t_ns": timestamps[:-1], "command_monotonic_ns": timestamps[:-1] - 100,
            "application_timestamp_phase": "host before target write and physics step; sim_time is after step",
            "state_timestamp_phase": "initial state and after every physics step; terminal inclusive",
            "command_events": [{"sequence": index + 1, "requested_cmd": command,
                                "first_apply_sim_time": tick_times[index], "t_ns": timestamps[index]}
                               for index, command in enumerate(commands)],
            "metadata": {"backend": "isaac", "physics_dt": dt},
        }

        class NativeLog:
            def call(self, method):
                return payload

        with patch("raiden.sim.action_log.SimConnection", return_value=NativeLog()):
            ActionLog().save(self.root)
        saved_states = REPLAY.load_rows(self.root / "sim_action_states.jsonl")
        self.assertEqual([row["state"] for row in saved_states], jsonable(states))
        self.assertEqual(json.loads((self.root / "sim_initial_state.json").read_text()), jsonable(states[0]))
        with np.load(self.root / "sim_action_log.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["cmd"], commands)
            np.testing.assert_allclose(saved["state_sim_time"][:-1], saved["sim_time"] - dt, atol=1e-15)
            np.testing.assert_array_equal(saved["state_t_ns"], timestamps)
            self.assertEqual(saved["cmd"].shape, (3, 7))
            self.assertEqual(saved["state_sim_time"].shape, (4,))
            self.assertEqual(saved["command_sequence"].dtype, np.int64)
            self.assertEqual(saved["cmd"].dtype, np.float64)
        events = REPLAY.load_rows(self.root / "sim_action_events.jsonl")
        self.assertEqual([row["first_apply_sim_time"] for row in events], tick_times[:-1].tolist())

        captures = [0, 1, 3]

        def pose(name, state):
            result = np.eye(4)
            result[0, 3] = .5 if name == "scene_camera" else state["scene"]["articulation"]["robot"]["joint_position"][0][0]
            return result

        class CameraConnection(Connection):
            def call(self, method, **kwargs):
                if method != "render":
                    return super().call(method, **kwargs)
                name = kwargs["cameras"][0]
                index = captures[self.index]
                self.index += 1
                return {name: {
                    "rgb": np.full((3, 4, 3), index, dtype=np.uint8), "t_ns": timestamps[index],
                    "frame_id": 100 + index, "T_base_cam": pose(name, states[index]),
                    "sim": {"state": states[index], "objects": {"oven": np.eye(4), "croissant": np.eye(4)},
                            "subtasks_done": [], "success": False},
                }}

        paths = [self.root / f"cameras/{name}.simrec" for name in ("scene_camera", "wrist_camera")]
        with patch("raiden.sim.camera.SimConnection", CameraConnection), patch("raiden.sim.camera.time.sleep"):
            for path in paths:
                camera = SimCamera(path.stem, record_mode="states")
                camera.open()
                camera.start_recording(path)
                for _ in captures:
                    self.assertTrue(camera.grab())
                camera.close()
                rows = REPLAY.load_rows(path / "sim_state.jsonl")
                self.assertEqual([row["frame_id"] for row in rows], [100, 101, 103])
                self.assertEqual([row["state"] for row in rows], jsonable([states[i] for i in captures]))
                np.testing.assert_array_equal(np.load(path / "timestamps.npy"), timestamps[captures])
                self.assertEqual(np.load(path / "camera_poses.npy").shape, (3, 4, 4))
                self.assertFalse((path / "rgb").exists())
        merged = _load_sim_log(paths)
        self.assertEqual(merged["object_poses"].shape, (6, 2, 4, 4))
        self.assertEqual(merged["subtask_done"].shape, (6, 0))
        self.assertFalse(merged["success"].any())
        self.assertEqual(merged["state"][0], jsonable(states[0]))

        class ReplayConnection:
            def call(self, method, **kwargs):
                if method == "get_metadata":
                    return {"physics_dt": dt}
                if method == "reset_to":
                    self.state = copy.deepcopy(kwargs["state"])
                    self.index = round((self.state["sim_time"] - initial_time) / dt)
                elif method == "step":
                    np.testing.assert_array_equal(kwargs["cmd"], commands[self.index])
                    self.index += 1
                    self.state = states[self.index]
                    return {"state": self.state}
                elif method == "render":
                    name = kwargs["cameras"][0]
                    return {name: {"rgb": np.full((3, 4, 3), self.index, dtype=np.uint8),
                                   "T_base_cam": pose(name, self.state)}}
                else:
                    raise AssertionError(method)

        rendered = self.root / "rendered"
        self.assertEqual(REPLAY.rerender(ReplayConnection(), self.root, output=rendered),
                         {"scene_camera": 3, "wrist_camera": 3})
        for source in paths:
            target = rendered / source.name
            loaded_times, _ = _extract_simrec(target, self.root / "extracted" / source.stem,
                                            self.root / "depth", self.root / "sim_model.xml", None)
            np.testing.assert_array_equal(loaded_times, timestamps[captures])
            np.testing.assert_array_equal(np.load(target / "camera_poses.npy"), np.load(source / "camera_poses.npy"))
            self.assertFalse((source / "rgb").exists())
        report = REPLAY.replay_commands(ReplayConnection(), self.root)[0]
        self.assertEqual(report["ticks"], 3)
        self.assertEqual(report["recorded_state_samples_compared"], 3)
        self.assertEqual(max(report["max_abs_scene_field_error"].values()), 0.)

    def test_rerender_preserves_recorded_poses_and_rejects_wrong_pose(self):
        self.record("states")
        rows = REPLAY.load_rows(self.simrec / "sim_state.jsonl")
        class Renderer:
            def __init__(self, wrong_pose=False):
                self.wrong_pose = wrong_pose
            def call(self, method, **kwargs):
                if method == "reset_to":
                    return None
                pose = np.eye(4)
                if self.wrong_pose:
                    pose[0, 3] = .01
                return {"scene_camera": {"rgb": np.full((3, 4, 3), 7, dtype=np.uint8), "T_base_cam": pose}}
        with self.assertRaisesRegex(ValueError, "optical pose"):
            REPLAY.rerender(Renderer(True), self.root)
        self.assertFalse((self.simrec / "rgb/0000000000.png").exists())
        self.assertEqual(REPLAY.rerender(Renderer(), self.root), {"scene_camera": 3})
        np.testing.assert_array_equal(np.load(self.simrec / "camera_poses.npy"),
                                      np.asarray([row["T_base_cam"] for row in rows]))
        self.assertEqual(convert_recording(str(self.root)), {"scene_camera": 3})

    def test_mujoco_recording_keeps_legacy_numeric_state_format(self):
        class Legacy(Connection):
            def call(self, method, **kwargs):
                if method == "get_camera_info":
                    return {"serial": "mujoco", "width": 4, "height": 3,
                            "K": np.eye(3), "mujoco_camera": "scene_camera"}
                if method == "get_task_status":
                    return {"subtasks": ["place"]}
                result = super().call(method, **kwargs)
                for frame in result.values():
                    frame["sim"].update(state=np.array([1., 2., 3.]), warmstart=np.array([0.]),
                                        subtasks_done=[False])
                return result
        with patch("raiden.sim.camera.SimConnection", Legacy), patch("raiden.sim.camera.time.sleep"):
            camera = SimCamera("scene_camera")
            camera.open()
            camera.start_recording(self.simrec)
            self.assertTrue(camera.grab())
            camera.close()
        self.assertFalse((self.simrec / "sim_state.jsonl").exists())
        with np.load(self.simrec / "sim_state.npz", allow_pickle=False) as log:
            np.testing.assert_array_equal(log["state"], [[1., 2., 3.]])
            np.testing.assert_array_equal(log["warmstart"], [[0.]])

    def test_native_command_replay_reports_fidelity_and_repeatability(self):
        initial = {"backend": "isaac", "physics_dt": .002, "sim_time": 0., "scene": {"q": [0.]}}
        (self.root / "sim_initial_state.json").write_text(json.dumps(initial))
        (self.root / "sim_metadata.json").write_text(json.dumps({"physics_dt": .002}))
        np.savez(self.root / "sim_action_log.npz", cmd=np.zeros((2, 7)), sim_time=[.002, .004])
        (self.root / "sim_action_states.jsonl").write_text("\n".join(json.dumps({"state": {**initial,
             "sim_time": value}, "sim_time": value}) for value in [.002, .004]))
        class Physics:
            def call(self, method, **kwargs):
                if method == "get_metadata":
                    return {"physics_dt": .002}
                if method == "reset_to":
                    self.state = dict(kwargs["state"])
                elif method == "step":
                    self.state["sim_time"] += .002
                    return {"state": {**self.state, "scene": {"q": np.asarray(self.state["scene"]["q"])}}}
        reports = REPLAY.replay_commands(Physics(), self.root, repeat=2)
        self.assertEqual(reports[0]["recorded_state_samples_compared"], 2)
        self.assertNotEqual(reports[0]["scene_sha256"],
                            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
        self.assertTrue(REPLAY._scene_arrays({"scene": {"q": np.asarray([1.])}}))
        self.assertEqual(reports[0]["max_abs_scene_field_error"], {"scene/q": 0.})
        self.assertEqual(reports[1]["max_scene_difference_from_run_0"], 0.)
        np.savez(self.root / "sim_action_log.npz", cmd=np.zeros((2, 7)), sim_time=[.002, .006])
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            REPLAY.replay_commands(Physics(), self.root)

    def test_action_log_structures_are_portable_json(self):
        with patch("raiden.sim.action_log.SimConnection", Connection):
            log = ActionLog()
        initial = {"backend": "isaac", "scene": {"q": np.array([1., 2.])}}
        log._c.call = lambda method: {"initial_state": initial, "cmd": np.zeros((2, 7)), "sim_time": np.array([.002, .004]),
            "states": np.array([{"q": np.array([1., 2.]), "t_ns": 123, "sim_time": .002}], dtype=object),
            "state_t_ns": np.array([123]), "state_sim_time": np.array([.002]),
            "command_events": [{"requested_cmd": np.ones(7), "t_ns": 123}],
            "metadata": {"backend": "isaac"}}
        log.save(self.root)
        with np.load(self.root / "sim_action_log.npz", allow_pickle=False) as data:
            self.assertEqual(data["cmd"].shape, (2, 7))
            self.assertNotIn("states", data)
        self.assertEqual(json.loads((self.root / "sim_initial_state.json").read_text())["scene"]["q"], [1, 2])
        self.assertEqual(REPLAY.load_rows(self.root / "sim_action_events.jsonl")[0]["t_ns"], 123)


if __name__ == "__main__":
    unittest.main()
