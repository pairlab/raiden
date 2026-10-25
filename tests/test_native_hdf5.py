"""Native RGB policy export: wire clocks, arm semantics and geometry regressions."""

import importlib.util
import json
import sqlite3
import struct
import tempfile
import unittest
from pathlib import Path

try:
    import h5py
except ImportError:
    h5py = None
import cv2
import numpy as np

from raiden.native_hdf5 import export_episodes, image_header, image_pixels

EPOCH = 1_792_965_000_000_000_000


def cdr_string(text):
    raw = text.encode() + b"\0"
    return b"\x00\x01\0\0" + struct.pack("<I", len(raw)) + raw


def image_message(image, timestamp, *, cdr=True, encoding="bgr8"):
    data = bytearray(b"\x00\x01\0\0" if cdr else struct.pack("<I", 1))

    def number(code, value, alignment=4):
        if cdr:
            data.extend(b"\0" * ((-len(data)) % alignment))
        data.extend(struct.pack("<" + code, value))

    def string(text):
        raw = text.encode() + (b"\0" if cdr else b"")
        number("I", len(raw))
        data.extend(raw)

    number("I", timestamp // 1_000_000_000)
    number("I", timestamp % 1_000_000_000)
    string("Color")
    number("I", image.shape[0])
    number("I", image.shape[1])
    string(encoding)
    number("B", 0, 1)
    number("I", image.shape[1] * 3)
    number("I", image.nbytes)
    data.extend(image.tobytes())
    return bytes(data)


def compressed_message(image, timestamp):
    prefix = image_message(image, timestamp)
    # CompressedImage shares the CDR Header but replaces all following fields.
    frame_end = 22  # four-byte encapsulation,sec,nsec,string length,Color\0
    data = bytearray(prefix[:frame_end])
    data.extend(b"\0" * ((-len(data)) % 4))
    format_name = b"bgr8; png compressed bgr8\0"
    data.extend(struct.pack("<I", len(format_name)))
    data.extend(format_name)
    data.extend(b"\0" * ((-len(data)) % 4))
    ok, png = cv2.imencode(".png", image)
    assert ok
    data.extend(struct.pack("<I", png.nbytes))
    data.extend(png.tobytes())
    return bytes(data)


def make_episode(
    root, *, bimanual=True, complete=True, modern=True, compressed_scene=False
):
    episode = root / "0005"
    (episode / "cameras").mkdir(parents=True)
    names = ["scene_camera", "left_wrist_camera"] + (
        ["right_wrist_camera"] if bimanual else []
    )
    meta = {
        "complete": complete,
        "task_name": "stand_bottle",
        "task_instruction": "stand the bottles up",
        "cameras": names,
        "camera_capture_receipts": {},
        "recording_config": {"cameras": {}},
    }
    rows = EPOCH + np.arange(35, dtype=np.int64) * 10_000_000
    robot = {"timestamps": rows}
    for j, arm in enumerate(
        ["follower_l", "follower_r"] if bimanual else ["follower_l"]
    ):
        positions = np.repeat((np.arange(len(rows)) * 0.01 + j)[:, None], 7, axis=1)
        positions[:, 6] = 0.3 + np.arange(len(rows)) * 0.01
        commands = positions + 0.1
        commands[:, 6] = np.arange(len(rows)) >= 13
        robot[arm + "_joint_pos_7d"] = positions
        robot[arm + "_joint_cmd"] = commands
        if modern:
            robot[arm + "_state_timestamp_ns"] = rows + 3_000_000
            robot[arm + "_state_timestamp_valid"] = np.ones(len(rows), bool)
            robot[arm + "_command_timestamp_ns"] = rows - 1_000_000
            robot[arm + "_command_timestamp_valid"] = np.ones(len(rows), bool)
    np.savez(episode / "robot_data.npz", **robot)
    transform = {
        "success": True,
        "rotation_matrix": np.eye(3).tolist(),
        "translation_vector": [0, 0, 0],
    }
    calibration = {
        "coordinate_frame": "left_arm_base",
        "cameras": {},
        "bimanual_transform": {"T_left_from_right": np.eye(4).tolist()},
    }
    source_pixels = {}
    for j, camera in enumerate(names):
        meta["camera_capture_receipts"][camera] = {
            "native_width": 8,
            "native_height": 4,
            "width": 8,
            "height": 4,
            "policy_crop": None,
            "fx": 100.0,
            "fy": 100.0,
            "cx": 4.0,
            "cy": 2.0,
            "distortion_model": "distortion.inverse_brown_conrady",
            "k1": 0.1,
        }
        meta["recording_config"]["cameras"][camera] = {
            "role": ["scene", "left_wrist", "right_wrist"][j],
            "policy_crop": None,
        }
        calibration["cameras"][camera] = {
            "extrinsics" if j == 0 else "hand_eye_calibration": transform
        }
        path = episode / "cameras" / (camera + ".db3")
        with sqlite3.connect(path) as db:
            db.execute(
                "CREATE TABLE topics (id INTEGER PRIMARY KEY,name TEXT,type TEXT,serialization_format TEXT)"
            )
            db.execute(
                "CREATE TABLE messages (id INTEGER PRIMARY KEY,topic_id INTEGER,timestamp INTEGER,data BLOB)"
            )
            prefix = f"/device_0/sensor_{0 if j else 1}/Color_0/image/"
            compressed = compressed_scene and j == 0
            db.execute(
                "INSERT INTO topics VALUES (1,?,?,?)",
                (
                    prefix + ("data/compressed" if compressed else "data"),
                    "sensor_msgs/msg/CompressedImage"
                    if compressed
                    else "sensor_msgs/msg/Image",
                    "cdr",
                ),
            )
            db.execute(
                "INSERT INTO topics VALUES (2,?,?,?)",
                (prefix + "metadata", "std_msgs/msg/String", "cdr"),
            )
            images = []
            for i in range(10):
                # The ROS2 header/log time is RELATIVE. SDK timing metadata is
                # epoch/global; accidentally using the header cannot pass.
                record_time = 100_000_000 + i * 33_333_333
                actual = EPOCH + 17_000_000 + i * 33_333_333 + j * 2_000_000
                image = np.zeros((4, 8, 3), dtype=np.uint8)
                image[..., 0], image[..., 1], image[..., 2] = np.arange(8), i, j + 40
                image[0, 2] = [1, 2, 250]
                images.append(image)
                body = (
                    compressed_message(image, record_time)
                    if compressed
                    else image_message(image, record_time)
                )
                db.execute(
                    "INSERT INTO messages (topic_id,timestamp,data) VALUES (1,?,?)",
                    (record_time, body),
                )
                timing = f"Frame number={i};Frame Counter={i};timestamp_domain=Global Time;timestamp={actual / 1e6:.6f};"
                db.execute(
                    "INSERT INTO messages (topic_id,timestamp,data) VALUES (2,?,?)",
                    (record_time, cdr_string(timing)),
                )
            source_pixels[camera] = images
    (episode / "metadata.json").write_text(json.dumps(meta))
    (episode / "calibration_results.json").write_text(json.dumps(calibration))
    return episode, robot, source_pixels


class NativeWireTest(unittest.TestCase):
    def test_ros1_and_ros2_pixels_keep_orientation_and_rgb_channels(self):
        bgr = np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3)
        for cdr in (True, False):
            with self.subTest(cdr=cdr):
                message = image_message(bgr, 1_234_567_890, cdr=cdr)
                self.assertEqual(
                    image_header(message, cdr=cdr)["timestamp_ns"], 1_234_567_890
                )
                np.testing.assert_array_equal(
                    image_pixels(message, cdr=cdr), bgr[:, :, ::-1]
                )
                if not cdr:
                    np.testing.assert_array_equal(
                        image_pixels(message + b"\0" * 4, cdr=False), bgr[:, :, ::-1]
                    )

    def test_malformed_wire_headers_are_rejected(self):
        raw = image_message(np.zeros((4, 8, 3), np.uint8), 100)
        for value in (raw[:-1], raw[:15], b"\0\x02\0\0" + raw[4:], raw + b"x"):
            with self.subTest(length=len(value)), self.assertRaises(ValueError):
                image_pixels(value, cdr=True)
        # Invalid is_bigendian is rejected even for one-byte color pixels.
        changed = bytearray(raw)
        header = image_header(raw, cdr=True)
        changed[header["offset"] - 12] = 9
        with self.assertRaises(ValueError):
            image_pixels(bytes(changed), cdr=True)

    def test_png_compressed_color_decode_and_invalid_payload(self):
        image = np.arange(96, dtype=np.uint8).reshape(4, 8, 3)
        raw = compressed_message(image, 100)
        np.testing.assert_array_equal(
            image_pixels(raw, cdr=True, compressed=True), image[:, :, ::-1]
        )
        changed = bytearray(raw)
        header = image_header(raw, cdr=True, compressed=True)
        changed[header["offset"]] = 0
        with self.assertRaisesRegex(ValueError, "lossless color PNG"):
            image_pixels(bytes(changed), cdr=True, compressed=True)
        with self.assertRaises(ValueError):
            image_pixels(raw[:-1], cdr=True, compressed=True)


@unittest.skipIf(h5py is None, "install the hdf5 extra to run exporter tests")
class NativeHDF5Test(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_bimanual_db3_uses_sdk_times_causal_commands_and_recorded_intrinsics(self):
        episode, robot, pixels = make_episode(self.root, compressed_scene=True)
        original = (episode / "metadata.json").read_bytes()
        target = self.root / "policy.hdf5"
        report = export_episodes(self.root, target, ["0005"], sizes=(4, 2))
        self.assertEqual((episode / "metadata.json").read_bytes(), original)
        self.assertFalse(report["depth_exported"])
        self.assertEqual(report["episodes"]["0005"]["action_dims"], 14)
        with h5py.File(target, "r") as f:
            g = f["data/demo_0005"]
            grid = g["obs/policy_timestamp_ns"][()]
            self.assertTrue(np.all(grid >= EPOCH))
            self.assertTrue(np.all(np.diff(grid) >= 33_333_333))
            for j, arm in enumerate(["follower_l", "follower_r"]):
                ts = robot[arm + "_command_timestamp_ns"]
                ix = np.searchsorted(ts, grid, side="right") - 1
                np.testing.assert_array_equal(
                    g["actions_joint_pos"][:, j * 7 : j * 7 + 7],
                    robot[arm + "_joint_cmd"][ix].astype(np.float32),
                )
                self.assertIn(f"robot{j}_joint_pos", g["obs"])
            for camera, images in pixels.items():
                ix = g[camera + "_source_frame_index"][()]
                np.testing.assert_array_equal(
                    g[f"obs/{camera}_image_4"][0], images[ix[0]][:, 2:6, ::-1]
                )
                np.testing.assert_array_equal(
                    g[f"calibration/{camera}/intrinsic_2"][()],
                    [[50, 0, 1], [0, 50, 1], [0, 0, 1]],
                )
            self.assertNotIn("T_base_table", g["calibration"])
            self.assertNotIn("depth", g["obs"])
            self.assertEqual(f["mask/train"][()].tolist(), [b"demo_0005"])
        with self.assertRaises(FileExistsError):
            export_episodes(self.root, target, ["0005"])

    def test_legacy_single_arm_keeps_linear_targets_and_binary_hold(self):
        _, robot, _ = make_episode(self.root, bimanual=False, modern=False)
        target = self.root / "single.hdf5"
        export_episodes(
            self.root, target, ["0005"], sizes=(4,), validation_episodes=["0005"]
        )
        with h5py.File(target, "r") as f:
            g = f["data/demo_0005"]
            grid = g["obs/policy_timestamp_ns"][()]
            times = robot["timestamps"]
            expected = np.interp(
                grid - EPOCH, times - EPOCH, robot["follower_l_joint_cmd"][:, 0]
            )
            np.testing.assert_allclose(
                g["actions_joint_pos"][:, 0], expected, atol=1e-7
            )
            self.assertEqual(g["actions_joint_pos"].shape[1], 7)
            self.assertEqual(set(np.unique(g["actions_joint_pos"][:, 6])), {0.0, 1.0})
            self.assertNotIn("robot1_joint_pos", g["obs"])
            self.assertEqual(f["mask/valid"][()].tolist(), [b"demo_0005"])

    @unittest.skipUnless(
        importlib.util.find_spec("rosbags"), "ROS1 reader optional dependency is absent"
    )
    def test_ros1_bag_single_arm_export(self):
        from rosbags.rosbag1 import Writer

        episode, _, pixels = make_episode(self.root, bimanual=False, modern=False)
        for j, (camera, images) in enumerate(pixels.items()):
            path = episode / "cameras" / (camera + ".bag")
            with Writer(path) as writer:
                connection = writer.add_connection(
                    "/device_0/sensor_1/Color_0/image/data",
                    "sensor_msgs/msg/Image",
                    msgdef="std_msgs/Header header\nuint32 height\nuint32 width\nstring encoding\nuint8 is_bigendian\nuint32 step\nuint8[] data\n",
                    md5sum="060021388200f6f0f447d0fcd9c64743",
                )
                for i, image in enumerate(images):
                    timestamp = EPOCH + 17_000_000 + i * 33_333_333 + j * 2_000_000
                    writer.write(
                        connection,
                        timestamp,
                        image_message(image, timestamp, cdr=False) + b"\0" * 4,
                    )
            (episode / "cameras" / (camera + ".db3")).unlink()
        target = self.root / "ros1.hdf5"
        report = export_episodes(self.root, target, ["0005"], sizes=(4,))
        self.assertEqual(report["episodes"]["0005"]["action_dims"], 7)
        self.assertEqual(
            report["episodes"]["0005"]["cameras"]["scene_camera"]["timestamp_domains"],
            ["ros1_epoch_header"],
        )

    def test_incomplete_episode_and_missing_native_metadata_leave_no_output(self):
        episode, _, _ = make_episode(self.root, complete=False)
        target = self.root / "bad.hdf5"
        with self.assertRaisesRegex(ValueError, "incomplete"):
            export_episodes(self.root, target, ["0005"])
        self.assertFalse(target.exists())
        self.assertEqual(list(self.root.glob("*.tmp")), [])
        meta = json.loads((episode / "metadata.json").read_text())
        meta["complete"] = True
        (episode / "metadata.json").write_text(json.dumps(meta))
        with sqlite3.connect(episode / "cameras/left_wrist_camera.db3") as db:
            db.execute("DELETE FROM messages WHERE topic_id=2")
        with self.assertRaisesRegex(ValueError, "timing association"):
            export_episodes(self.root, target, ["0005"])
        self.assertFalse(target.exists())

    def test_explicit_selection_and_validation_ids_are_checked(self):
        for episodes, valid in (
            ([], []),
            (["../escape"], []),
            (["0005"], ["0006"]),
            (["0005", "0005"], []),
        ):
            with self.subTest(episodes=episodes, valid=valid), self.assertRaises(ValueError):
                export_episodes(
                    self.root,
                    self.root / "bad.hdf5",
                    episodes,
                    validation_episodes=valid,
                )

    def test_interpolation_gap_masks_preserve_episode_and_identify_affected_anchors(
        self,
    ):
        episode, robot, _ = make_episode(self.root)
        # Delete recorder rows in a100ms interval, retaining all camera frames.
        keep = np.r_[np.arange(10), np.arange(20, 35)]
        np.savez(episode / "robot_data.npz", **{k: v[keep] for k, v in robot.items()})
        target = self.root / "gap.hdf5"
        report = export_episodes(self.root, target, ["0005"], sizes=(4,))
        self.assertEqual(report["episodes"]["0005"]["robot_row_gaps_over_50ms"], 1)
        with h5py.File(target, "r") as f:
            g = f["data/demo_0005"]
            grid = g["obs/policy_timestamp_ns"][()]
            intervals = g["obs/robot0_state_interpolation_interval_ns"][()]
            mask = g["robot_state_gap_over_50ms"][()]
            np.testing.assert_array_equal(mask, intervals > 50_000_000)
            self.assertTrue(mask.any())
            self.assertTrue(np.all(grid[mask] > EPOCH + 93_000_000))
            self.assertTrue(np.all(grid[mask] < EPOCH + 203_000_000))


if __name__ == "__main__":
    unittest.main()
