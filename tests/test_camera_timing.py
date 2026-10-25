"""Image poses follow selected capture times and measured arm sample clocks."""

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from raiden.converter import (
    ConversionError,
    _align_cameras_by_timestamp,
    _build_lowdim,
    _robot_timing_coverage,
    convert_recording,
)

ORIGIN = 1_800_000_000_000_000_000


class MovingWrist:
    _configuration = SimpleNamespace(model=SimpleNamespace(nq=7))

    def fk(self, q):
        c, s = np.cos(q[1]), np.sin(q[1])
        result = np.array([
            [c, -s, 0, q[0]], [s, c, 0, q[2]], [0, 0, 1, q[3]], [0, 0, 0, 1],
        ])
        return result


def sample_joints(progress, offset=0):
    progress = np.asarray(progress, dtype=float)
    return np.column_stack([
        offset + progress, progress * np.pi / 2, progress / 3,
        progress / 5, progress * 0, progress * 0,
    ])


def robot_data(times=(0, 100, 200), sides=("l", "r")):
    data = {"timestamps": ORIGIN + np.asarray(times, dtype=np.int64)}
    for side in sides:
        joints = sample_joints(np.asarray(times) / 200, 1 if side == "r" else 0)
        data[f"follower_{side}_joint_pos"] = joints
        data[f"follower_{side}_joint_pos_7d"] = np.column_stack([joints, [0.2, 0.4, 0.8]])
        data[f"follower_{side}_joint_cmd"] = np.column_stack([joints, [1, 0, 1]])
    return data


def timed(data, side, kind, times, valid=None):
    key = f"follower_{side}_{kind}"
    times = np.asarray(times, dtype=np.int64)
    valid = times >= 0 if valid is None else np.asarray(valid, dtype=bool)
    data[f"{key}_timestamp_ns"] = np.where(valid, ORIGIN + times, -1)
    data[f"{key}_timestamp_valid"] = valid


def hand_eye():
    result = np.eye(4)
    result[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    result[:3, 3] = [0.2, -0.1, 0.3]
    return result


def build(root, data, camera_times, wrist_keys=None):
    transform = np.eye(4)
    transform[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    transform[:3, 3] = [0.6, 0, 0]
    he = hand_eye()
    calib = {"cameras": {name: {"hand_eye_calibration": {
        "success": True, "rotation_matrix": he[:3, :3].tolist(),
        "translation_vector": he[:3, 3].tolist(),
    }} for name in (wrist_keys or {})}}
    with patch("raiden.converter._get_kinematics", return_value=MovingWrist()):
        schema = _build_lowdim(
            root, list(camera_times), len(next(iter(camera_times.values()))), {}, calib,
            data, {}, transform,
            {name: ORIGIN + np.asarray(times, dtype=np.int64) for name, times in camera_times.items()},
            wrist_camera_joint_keys=wrist_keys,
        )
    frames = [pickle.loads(path.read_bytes()) for path in sorted((root / "lowdim").glob("*.pkl"))]
    return frames, schema, transform


class CameraTimingTest(unittest.TestCase):
    def test_each_wrist_uses_own_capture_for_translation_rotation_and_hand_eye(self):
        camera_times = {"scene": [40, 100, 160], "left": [30, 110, 110], "right": [50, 90, 170]}
        data = robot_data()
        with tempfile.TemporaryDirectory() as directory:
            frames, schema, base = build(
                Path(directory), data, camera_times,
                {"left": "follower_l_joint_pos", "right": "follower_r_joint_pos"},
            )
            for i, frame in enumerate(frames):
                for name, offset in (("left", 0), ("right", 1)):
                    q = sample_joints([camera_times[name][i] / 200], offset)[0]
                    expected = MovingWrist().fk(q) @ hand_eye()
                    if name == "right":
                        expected = base @ expected
                    np.testing.assert_allclose(frame["extrinsics"][name], expected, atol=2e-7)
                # Shared state and actions still use the reference camera grid.
                self.assertAlmostEqual(frame["joints"][0], camera_times["scene"][i] / 200)
                self.assertAlmostEqual(frame["action_joints"][0], camera_times["scene"][i] / 200)
            np.testing.assert_array_equal(frames[1]["extrinsics"]["left"], frames[2]["extrinsics"]["left"])
            self.assertEqual(schema["timing"]["wrist_extrinsics_grid"], "selected_camera_capture")

    def test_repeated_native_samples_and_command_before_first_row_are_valid(self):
        data = {key: values[[0, 1, 1, 2]].copy() for key, values in robot_data(sides=("l",)).items()}
        timed(data, "l", "state", [0, 80, 80, 200])
        timed(data, "l", "command", [0, 60, 60, 120])
        data["follower_l_command_timestamp_ns"][0] = ORIGIN - 20
        with tempfile.TemporaryDirectory() as directory:
            frames, _, _ = build(Path(directory), data, {"scene": [0, 59, 60, 80, 119, 120, 200]})
            np.testing.assert_array_equal([frame["action_joints"][0] for frame in frames], [0, 0, .5, .5, .5, 1, 1])
            self.assertAlmostEqual(frames[3]["joints"][0], .5)

    def test_nanosecond_epoch_precision_and_distinct_per_arm_state_times(self):
        data = robot_data()
        timed(data, "l", "state", [0, 80, 200])
        timed(data, "r", "state", [0, 120, 200])
        with tempfile.TemporaryDirectory() as directory:
            frames, schema, base = build(
                Path(directory), data, {"scene": [49, 50, 51], "left": [50, 51, 52], "right": [48, 49, 50]},
                {"left": "follower_l_joint_pos", "right": "follower_r_joint_pos"},
            )
            self.assertAlmostEqual(frames[0]["joints"][0], 0.5 * 49 / 80)
            self.assertAlmostEqual(frames[0]["joints"][7], 1 + 0.5 * 49 / 120)
            for index, frame in enumerate(frames):
                for name, elapsed, denominator, offset in (
                    ("left", 50 + index, 80, 0), ("right", 48 + index, 120, 1),
                ):
                    q = sample_joints([0.5 * elapsed / denominator], offset)[0]
                    expected = MovingWrist().fk(q) @ hand_eye()
                    if name == "right":
                        expected = base @ expected
                    np.testing.assert_allclose(frame["extrinsics"][name], expected, atol=2e-7)
            self.assertNotEqual(frames[0]["extrinsics"]["left"][0, 3], frames[1]["extrinsics"]["left"][0, 3])
            self.assertEqual(schema["timing"]["robot_fields"]["follower_l_joint_pos"]["timestamp_field"], "follower_l_state_timestamp_ns")

    def test_native_commands_hold_issued_target_and_gripper_through_tail(self):
        data = robot_data(sides=("l",))
        timed(data, "l", "command", [-1, 60, 120])
        with tempfile.TemporaryDirectory() as directory:
            frames, schema, _ = build(Path(directory), data, {"scene": [60, 119, 120, 199, 200]})
            np.testing.assert_array_equal([frame["action_joints"][0] for frame in frames], [.5, .5, 1, 1, 1])
            np.testing.assert_array_equal([frame["action_joints"][6] for frame in frames], [0, 0, 1, 1, 1])
            self.assertEqual(schema["timing"]["robot_fields"]["follower_l_joint_cmd"]["resampling"], "zero_order_hold")
            self.assertEqual(schema["timing"]["robot_fields"]["follower_l_joint_cmd"]["unknown_rows_omitted"], 1)

    def test_unknown_native_timing_keeps_legacy_interpolation_and_gripper_hold(self):
        data = robot_data(sides=("r",))
        for kind in ("state", "command"):
            timed(data, "r", kind, [-1, -1, -1])
        with tempfile.TemporaryDirectory() as directory:
            frames, schema, _ = build(Path(directory), data, {"scene": [49, 99, 100, 200]})
            np.testing.assert_allclose([frame["action_joints"][0] for frame in frames], [1.245, 1.495, 1.5, 2])
            np.testing.assert_array_equal([frame["action_joints"][6] for frame in frames], [1, 1, 0, 1])
            self.assertEqual(schema["timing"]["robot_fields"]["follower_r_joint_cmd"]["timestamp_field"], "timestamps")

    def test_partial_state_timing_uses_known_samples_and_rejects_uncovered_wrist(self):
        data = robot_data(sides=("l",))
        timed(data, "l", "state", [-1, 80, 180])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames, _, _ = build(root / "valid", data, {"scene": [100], "wrist": [150]}, {"wrist": "follower_l_joint_pos"})
            expected = MovingWrist().fk(sample_joints([.85])[0]) @ hand_eye()
            np.testing.assert_allclose(frames[0]["extrinsics"]["wrist"], expected, atol=2e-7)
            for time in (79, 181):
                with self.subTest(time=time), self.assertRaisesRegex(ConversionError, "refusing to extrapolate"):
                    build(root / str(time), data, {"scene": [100], "wrist": [time]}, {"wrist": "follower_l_joint_pos"})

    def test_malformed_timestamps_and_mixed_domains_are_rejected(self):
        for case in ("negative", "nonmonotonic", "float", "shape", "domain", "validity"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                data = robot_data(sides=("l",))
                timed(data, "l", "state", [0, 100, 200])
                key = "follower_l_state_timestamp_ns"
                if case == "negative":
                    data[key][0] = -1
                elif case == "nonmonotonic":
                    data[key][1] = ORIGIN + 201
                elif case == "float":
                    data[key] = data[key].astype(float)
                elif case == "shape":
                    data[key] = data[key][:2]
                elif case == "domain":
                    data[key] -= ORIGIN
                else:
                    del data["follower_l_state_timestamp_valid"]
                with self.assertRaises(ConversionError):
                    build(Path(directory), data, {"scene": [100]})

    def test_sim_pose_remains_exact_even_with_different_capture_times(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            camera = root / "rgb/wrist"
            camera.mkdir(parents=True)
            pose = np.eye(4)[None]
            pose[0, :3, 3] = [7, 8, 9]
            np.save(camera / "camera_poses.npy", pose)
            frames, _, _ = build(root, robot_data(), {"scene": [100], "wrist": [110]}, {"wrist": "follower_r_joint_pos"})
            np.testing.assert_array_equal(frames[0]["extrinsics"]["wrist"], pose[0])

    def test_alignment_is_truly_nearest_and_report_preserves_signed_times(self):
        # At 20, the old nominal-period lookup chose 35 despite 10 being nearer.
        times = {"scene": [0, 20, 40, 60, 80, 100], "wrist": [0, 10, 35, 80, 100]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, values in times.items():
                camera = root / "rgb" / name
                camera.mkdir(parents=True)
                for i in range(len(values)):
                    (camera / f"{i:010d}.png").write_bytes(bytes([i]))
            selected, counts = _align_cameras_by_timestamp(
                root, {name: ORIGIN + np.asarray(values, dtype=np.int64) * 1_000_000 for name, values in times.items()},
                {name: len(values) for name, values in times.items()}, camera_fps=30,
            )
            np.testing.assert_array_equal((selected["wrist"] - ORIGIN) // 1_000_000, [0, 10, 35, 80, 80, 100])
            self.assertEqual(counts, {"scene": 6, "wrist": 6})
            report = json.loads((root / "camera_alignment.json").read_text())
            stream = report["streams"]["wrist"]
            self.assertEqual(stream["selected_source_indices"], [0, 1, 2, 3, 3, 4])
            self.assertEqual(stream["residual_ns"], [0, -10_000_000, -5_000_000, 20_000_000, 0, 0])
            self.assertEqual(stream["held_slots"], 1)
            self.assertEqual(stream["max_abs_residual_ns"], 20_000_000)
            self.assertIsNone(report["physical_sync_tolerance_ns"])
            self.assertFalse(report["physical_sync_verified"])
            self.assertEqual((root / "rgb/wrist/0000000003.png").read_bytes(), bytes([3]))

    def test_alignment_trims_to_both_arm_sample_coverage_and_first_command(self):
        data = robot_data()
        timed(data, "l", "state", [5, 100, 195])
        timed(data, "r", "state", [10, 100, 190])
        timed(data, "r", "command", [-1, 20, 120])
        coverage = _robot_timing_coverage(data)
        np.testing.assert_array_equal(coverage - ORIGIN, [20, 190])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            times = {"scene": [0, 20, 100, 190, 200], "wrist": [0, 25, 105, 185, 200]}
            with patch("raiden.converter._apply_camera_selection"):
                selected, _ = _align_cameras_by_timestamp(
                    root, {name: ORIGIN + np.asarray(values, dtype=np.int64) for name, values in times.items()},
                    {name: len(values) for name, values in times.items()}, robot_timestamps=coverage,
                )
            np.testing.assert_array_equal(selected["scene"] - ORIGIN, [20, 100, 190])
            np.testing.assert_array_equal(selected["wrist"] - ORIGIN, [25, 105, 185])

    def test_nearest_ties_choose_earlier_capture_and_drop_guard_remains(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("raiden.converter._apply_camera_selection"):
                selected, _ = _align_cameras_by_timestamp(
                    root, {"scene": ORIGIN + np.array([0, 20, 40]), "wrist": ORIGIN + np.array([0, 10, 30, 40])},
                    {"scene": 3, "wrist": 4},
                )
            np.testing.assert_array_equal(selected["wrist"] - ORIGIN, [0, 10, 40])
            with self.assertRaisesRegex(ConversionError, "dropped .* frames in a row"):
                _align_cameras_by_timestamp(
                    root, {"scene": ORIGIN + np.array([0, 200_000_000]), "wrist": ORIGIN + np.array([0, 200_000_000])},
                    {"scene": 2, "wrist": 2}, camera_fps=30,
                )

    def test_native_realsense_bag_and_db3_discovery_preserve_conversion(self):
        for suffix in ("bag", "db3"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                cameras = root / "cameras"
                cameras.mkdir()
                source = cameras / f"scene.{suffix}"
                source.touch()
                (root / "metadata.json").write_text(json.dumps({"recording_config": {"cameras": {}}}))
                data = robot_data(sides=("l",))
                np.savez(root / "robot_data.npz", **data)

                def extract(path, rgb_dir, depth_dir, **kwargs):
                    self.assertEqual(path, source)
                    rgb_dir.mkdir(parents=True)
                    for i in range(3):
                        (rgb_dir / f"{i:010d}.png").write_bytes(bytes([i]))
                    np.save(rgb_dir / "timestamps.npy", data["timestamps"])
                    return data["timestamps"], {"fx": 1, "fy": 1, "cx": 0, "cy": 0, "width": 1, "height": 1}

                with patch("raiden.converter._extract_bag", side_effect=extract) as reader, patch(
                    "raiden.converter._get_kinematics", return_value=MovingWrist()
                ):
                    counts = convert_recording(str(root))
                reader.assert_called_once()
                self.assertEqual(counts, {"scene": 3})
                frame = pickle.loads((root / "0000/lowdim/0000000001.pkl").read_bytes())
                np.testing.assert_array_equal(frame["action_joints"], data["follower_l_joint_cmd"][1].astype(np.float32))
                np.testing.assert_array_equal(frame["extrinsics"]["scene"], np.eye(4))
                self.assertTrue((root / "0000/camera_alignment.json").is_file())

    def test_duplicate_bag_db3_camera_name_fails_before_extracting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cameras").mkdir()
            for suffix in ("bag", "db3"):
                (root / "cameras" / f"wrist.{suffix}").touch()
            with patch("raiden.converter._extract_bag") as reader:
                with self.assertRaisesRegex(ConversionError, "Duplicate RealSense recordings"):
                    convert_recording(str(root))
            reader.assert_not_called()
            self.assertFalse((root / "0000").exists())


if __name__ == "__main__":
    unittest.main()
