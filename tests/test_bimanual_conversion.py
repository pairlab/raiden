"""Conversion preserves arm order, causal gripper intent, and episode geometry."""

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from raiden.converter import (
    ConversionError,
    _build_lowdim,
    _build_sequence_metadata,
    _recording_camera_settings,
    convert_recording,
)

_ORIGIN = 1_800_000_000_000_000_000


class Kinematics:
    """Distinct synthetic joints map directly to EE translation."""

    _configuration = SimpleNamespace(model=SimpleNamespace(nq=7))

    def fk(self, q):
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = q[:3]
        return pose


def robot_data(sides=("l", "r")):
    result = {"timestamps": _ORIGIN + np.array([0, 100, 200], dtype=np.int64)}
    for side in sides:
        offset = 0 if side == "l" else 30
        arm = np.tile((np.array([0, 10, 20]) + offset)[:, None], (1, 6))
        grip = [1, 0, 1] if side == "l" else [0, 1, 0]
        measured = [0.2, 0.4, 0.8] if side == "l" else [0.9, 0.6, 0.1]
        result[f"follower_{side}_joint_cmd"] = np.column_stack([arm, grip])
        result[f"follower_{side}_joint_pos"] = arm
        result[f"follower_{side}_joint_pos_7d"] = np.column_stack([arm, measured])
    return result


def build(root, data, times=None, cameras=None, **kwargs):
    if times is None:
        times = [0, 49, 50, 99, 100, 150, 199, 200]
    cameras = cameras or ["scene"]
    with patch("raiden.converter._get_kinematics", return_value=Kinematics()):
        schema = _build_lowdim(
            root,
            cameras,
            len(times),
            {},
            kwargs.pop("calib", None),
            data,
            {},
            kwargs.pop("transform", None),
            {name: _ORIGIN + np.asarray(times, dtype=np.int64) for name in cameras},
            **kwargs,
        )
    frames = []
    for frame in sorted((root / "lowdim").glob("*.pkl")):
        with frame.open("rb") as stream:
            frames.append(pickle.load(stream))
    return frames, schema


class BimanualConversionTest(unittest.TestCase):
    def test_same_timestamp_uses_last_command_and_one_sample_episode_is_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = robot_data(("l",))
            data = {key: values[[0, 1, 1, 2]].copy() for key, values in data.items()}
            data["follower_l_joint_cmd"][:, 6] = [1, 1, 0, 1]
            frames, _ = build(root / "duplicates", data, times=[99, 100, 101, 200])
            np.testing.assert_array_equal(
                [frame["action_joints"][6] for frame in frames], [1, 0, 0, 1]
            )
            single = {key: values[-1:] for key, values in data.items()}
            frames, schema = build(root / "single", single, times=[200])
            self.assertEqual(frames[0]["action_joints"][6], 1)
            self.assertEqual(schema["action"]["dims"], 7)

    def test_binary_intent_holds_until_each_edge_and_includes_terminal_command(self):
        for sides in (("l",), ("r",), ("l", "r")):
            with self.subTest(sides=sides), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                frames, schema = build(root, robot_data(sides))
                commands = np.stack([frame["action_joints"] for frame in frames])
                poses = np.stack([frame["action"] for frame in frames])
                measured = np.stack([frame["joints"] for frame in frames])
                actual = np.stack([frame["actual_poses"] for frame in frames])
                for index, side in enumerate(sides):
                    expected = [1, 1, 1, 1, 0, 0, 0, 1]
                    if side == "r":
                        expected = 1 - np.asarray(expected)
                    np.testing.assert_array_equal(commands[:, index * 7 + 6], expected)
                    np.testing.assert_array_equal(poses[:, index * 13 + 12], expected)
                    # Joints keep linear interpolation, while measured openings remain continuous.
                    offset = 0 if side == "l" else 30
                    np.testing.assert_allclose(
                        commands[:, index * 7],
                        np.array([0, 4.9, 5, 9.9, 10, 15, 19.9, 20]) + offset,
                    )
                    self.assertAlmostEqual(
                        measured[3, index * 7 + 6], 0.398 if side == "l" else 0.603
                    )
                    np.testing.assert_allclose(
                        actual[:, index * 13 + 12], measured[:, index * 7 + 6]
                    )
                _build_sequence_metadata(
                    root, ["scene"], {"scene": len(frames)}, {}, {}, schema
                )
                metadata = json.loads((root / "metadata.json").read_text())
                self.assertEqual(metadata["action"]["dims"], 7 * len(sides))
                self.assertEqual(metadata["action_ee"]["dims"], 13 * len(sides))
                self.assertEqual(metadata["actual_poses"]["dims"], 13 * len(sides))
                self.assertEqual(metadata["action"]["lowdim_key"], "action_joints")
                self.assertEqual(metadata["action_ee"]["lowdim_key"], "action")
                self.assertEqual(
                    metadata["action_ee"]["frames"],
                    {
                        (
                            "left" if side == "l" else "right"
                        ): f"{('left' if side == 'l' else 'right')}_arm_base"
                        for side in sides
                    },
                )

    def test_right_wrist_requires_transform_while_ee_poses_keep_own_base(self):
        hand_eye = {
            "success": True,
            "rotation_matrix": np.eye(3).tolist(),
            "translation_vector": [1, 2, 3],
        }
        calibration = {"cameras": {"wrist": {"hand_eye_calibration": hand_eye}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = {
                "cameras": ["wrist"],
                "calib": calibration,
                "wrist_camera_joint_keys": {"wrist": "follower_r_joint_pos"},
            }
            with self.assertRaisesRegex(ConversionError, "bimanual base transform"):
                build(root, robot_data(), **arguments)
            transform = np.eye(4, dtype=np.float32)
            transform[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
            transform[0, 3] = 100
            frames, schema = build(root, robot_data(), transform=transform, **arguments)
            # right-base wrist [30,30,30]+hand-eye [1,2,3] -> left-base [68,31,33].
            np.testing.assert_allclose(
                frames[0]["extrinsics"]["wrist"][:3, 3], [68, 31, 33]
            )
            np.testing.assert_allclose(frames[0]["action"][13:16], [30, 30, 30])
            np.testing.assert_allclose(frames[0]["actual_poses"][13:16], [30, 30, 30])
            self.assertEqual(schema["action_ee"]["frames"]["right"], "right_arm_base")

    def test_native_sim_camera_pose_needs_no_bimanual_transform(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "rgb/wrist"
            image_root.mkdir(parents=True)
            poses = np.repeat(np.eye(4)[None], 3, axis=0)
            poses[:, 0, 3] = [1, 2, 3]
            np.save(image_root / "camera_poses.npy", poses)
            frames, _ = build(
                root,
                robot_data(),
                times=[0, 100, 200],
                cameras=["wrist"],
                wrist_camera_joint_keys={"wrist": "follower_r_joint_pos"},
            )
            np.testing.assert_array_equal(frames[-1]["extrinsics"]["wrist"], poses[-1])

    def test_episode_roles_and_capture_crop_survive_live_config_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cameras").mkdir()
            (root / "cameras/wrist.bag").touch()
            np.savez(root / "robot_data.npz", **robot_data())
            (root / "calibration_results.json").write_text(
                json.dumps(
                    {
                        "cameras": {
                            "wrist": {
                                "hand_eye_calibration": {
                                    "success": True,
                                    "rotation_matrix": np.eye(3).tolist(),
                                    "translation_vector": [0, 0, 0],
                                }
                            }
                        }
                    }
                )
            )
            recorded = {
                "recording_config": {
                    "schema_version": 1,
                    "cameras": {
                        "wrist": {
                            "role": "left_wrist",
                            "type": "realsense",
                            "policy_crop": [1, 2, 3, 4],
                        }
                    },
                },
                "camera_capture_receipts": {"wrist": {"policy_crop": [5, 6, 7, 8]}},
            }
            (root / "metadata.json").write_text(json.dumps(recorded))
            live = root / "live_camera.json"
            crops = []

            def extract(path, rgb_dir, depth_dir, **kwargs):
                crops.append(kwargs["crop"])
                rgb_dir.mkdir(parents=True)
                times = _ORIGIN + np.array([0, 100, 200], dtype=np.int64)
                for index in range(3):
                    cv2.imwrite(
                        str(rgb_dir / f"{index:010d}.png"),
                        np.zeros((8, 7, 3), np.uint8),
                    )
                np.save(rgb_dir / "timestamps.npy", times)
                return times, {
                    "fx": 1,
                    "fy": 2,
                    "cx": 3,
                    "cy": 4,
                    "width": 7,
                    "height": 8,
                }

            for index, role in enumerate(("right_wrist", "scene")):
                live.write_text(
                    json.dumps(
                        {
                            "wrist": {
                                "serial": "fake",
                                "type": "realsense",
                                "role": role,
                                "crop": [9, 9, 9, 9],
                            }
                        }
                    )
                )
                output = root / f"output{index}"
                with (
                    patch("raiden.converter.CAMERA_CONFIG", str(live)),
                    patch("raiden.converter._extract_bag", side_effect=extract),
                    patch(
                        "raiden.converter._get_kinematics", return_value=Kinematics()
                    ),
                ):
                    convert_recording(str(root), episode_dir=str(output))
                with (output / "lowdim/0000000001.pkl").open("rb") as stream:
                    frame = pickle.load(stream)
                np.testing.assert_array_equal(
                    frame["extrinsics"]["wrist"][:3, 3], [10, 10, 10]
                )
                metadata = json.loads((output / "metadata.json").read_text())
                self.assertEqual(
                    metadata["camera_settings"]["wrist"]["role"], "left_wrist"
                )
                self.assertEqual(
                    metadata["camera_settings"]["wrist"]["policy_crop"], [5, 6, 7, 8]
                )
            self.assertEqual(crops, [(5, 6, 7, 8)] * 2)

    def test_old_episode_falls_back_but_recorded_nulls_are_authoritative(self):
        with tempfile.TemporaryDirectory() as directory:
            live = Path(directory) / "camera.json"
            live.write_text(
                json.dumps(
                    {
                        "wrist": {
                            "serial": "fake",
                            "type": "realsense",
                            "role": "right_wrist",
                            "crop": [1, 2, 3, 4],
                        }
                    }
                )
            )
            with patch("raiden.converter.CAMERA_CONFIG", str(live)):
                old = _recording_camera_settings({})
                self.assertEqual(old["wrist"]["role"], "right_wrist")
                self.assertEqual(old["wrist"]["policy_crop"], [1, 2, 3, 4])
                saved = _recording_camera_settings(
                    {
                        "recording_config": {
                            "cameras": {
                                "wrist": {
                                    "role": None,
                                    "type": "realsense",
                                    "policy_crop": None,
                                }
                            }
                        }
                    }
                )
                self.assertIsNone(saved["wrist"]["role"])
                self.assertIsNone(saved["wrist"]["policy_crop"])
                readback = _recording_camera_settings(
                    {
                        "recording_config": {
                            "cameras": {"wrist": {"policy_crop": [1, 2, 3, 4]}}
                        },
                        "camera_capture_receipts": {"wrist": {"policy_crop": None}},
                    }
                )
                self.assertIsNone(readback["wrist"]["policy_crop"])


if __name__ == "__main__":
    unittest.main()
