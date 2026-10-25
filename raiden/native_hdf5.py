"""Export explicitly selected native RealSense episodes to policy HDF5.

This file-only exporter needs numpy, opencv, h5py and (for ROS1 bags) rosbags;
it does not import a camera SDK or initialize a robot. Images remain in their
recorded orientation. Depth stays in the native recordings; this RGB/joint
policy export does not advertise or produce depth pixels.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import struct
import tempfile
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from raiden.calibration_utils import left_from_right, validate_transform
from raiden.realsense_files import list_realsense_recordings

SCHEMA = "raiden.native_policy_hdf5/1"
WALL_CLOCK_MIN_NS = 100_000_000_000_000_000


class _Wire:
    def __init__(self, raw: bytes, *, cdr: bool):
        self.raw, self.cdr, self.pos = raw, cdr, 0
        self.endian = "<"
        if cdr:
            if len(raw) < 4 or raw[:2] not in (b"\x00\x00", b"\x00\x01"):
                raise ValueError("Unsupported CDR encapsulation")
            self.endian = "<" if raw[1] == 1 else ">"
            self.pos = 4

    def number(self, code: str, alignment: int = 4):
        if self.cdr:
            self.pos += (-self.pos) % alignment
        size = struct.calcsize(code)
        if self.pos + size > len(self.raw):
            raise ValueError("Truncated image/message header")
        value = struct.unpack_from(self.endian + code, self.raw, self.pos)[0]
        self.pos += size
        return value

    def string(self) -> str:
        size = self.number("I")
        if self.pos + size > len(self.raw) or (
            self.cdr and (size == 0 or self.raw[self.pos + size - 1] != 0)
        ):
            raise ValueError("Invalid serialized string length or terminator")
        data = self.raw[self.pos : self.pos + size]
        self.pos += size
        return data.rstrip(b"\0").decode("utf-8")


def image_header(
    raw: bytes, *, cdr: bool, total_length: int | None = None, compressed=False
) -> dict:
    """Validate an Image body, optionally from its prefix for a metadata scan."""
    wire = _Wire(raw, cdr=cdr)
    seq = None if cdr else wire.number("I")
    sec, nsec = wire.number("i" if cdr else "I"), wire.number("I")
    if sec < 0 or nsec >= 1_000_000_000:
        raise ValueError("Invalid Image header timestamp")
    wire.string()  # frame_id
    if compressed:
        format_name = wire.string().lower()
        if format_name not in (
            "bgr8; png compressed bgr8",
            "rgb8; png compressed rgb8",
        ):
            raise ValueError(
                f"Unsupported compressed color format {format_name!r}; expected lossless RGB8 PNG"
            )
        count = wire.number("I")
        if len(raw) != wire.pos + count or count < 33:
            raise ValueError("Invalid compressed Image data length")
        png = raw[wire.pos :]
        if (
            png[:8] != b"\x89PNG\r\n\x1a\n"
            or png[12:16] != b"IHDR"
            or png[24:26] != b"\x08\x02"
        ):
            raise ValueError("Expected an eight-bit, three-channel lossless color PNG")
        width, height = struct.unpack_from(">II", png, 16)
        if min(width, height) <= 0:
            raise ValueError("Invalid compressed Image dimensions")
        return {
            "timestamp_ns": sec * 1_000_000_000 + nsec,
            "sequence": seq,
            "height": height,
            "width": width,
            "encoding": "png",
            "offset": wire.pos,
            "count": count,
        }
    height, width = wire.number("I"), wire.number("I")
    encoding = wire.string().lower()
    endian, step, count = wire.number("B", 1), wire.number("I"), wire.number("I")
    if encoding not in ("bgr8", "rgb8") or endian not in (0, 1):
        raise ValueError(
            f"Unsupported color Image encoding/endian: {encoding}/{endian}"
        )
    if height <= 0 or width <= 0 or step < width * 3 or count != height * step:
        raise ValueError("Invalid color Image dimensions, stride or data length")
    length = len(raw) if total_length is None else total_length
    suffix = length - wire.pos - count
    if suffix not in ((0,) if cdr else (0, 4)):
        raise ValueError("Unexpected Image body length")
    if total_length is None and suffix and raw[-suffix:] != b"\0" * suffix:
        raise ValueError("Unqualified ROS1 Image suffix")
    return {
        "timestamp_ns": sec * 1_000_000_000 + nsec,
        "sequence": seq,
        "height": height,
        "width": width,
        "encoding": encoding,
        "offset": wire.pos,
        "step": step,
        "count": count,
    }


def image_pixels(raw: bytes, *, cdr: bool, compressed=False) -> np.ndarray:
    header = image_header(raw, cdr=cdr, compressed=compressed)
    if compressed:
        png = np.frombuffer(raw, np.uint8, header["count"], header["offset"])
        image = cv2.imdecode(png, cv2.IMREAD_UNCHANGED)
        if (
            image is None
            or image.dtype != np.uint8
            or image.shape != (header["height"], header["width"], 3)
        ):
            raise ValueError(
                "Compressed color PNG could not be decoded with the advertised dimensions"
            )
        return image[:, :, ::-1]
    rows = np.frombuffer(raw, np.uint8, header["count"], header["offset"])
    image = rows.reshape(header["height"], header["step"])[:, : header["width"] * 3]
    image = image.reshape(header["height"], header["width"], 3)
    return image[:, :, ::-1] if header["encoding"] == "bgr8" else image


def _strict_times(values, label, *, duplicates=False) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if values.ndim != 1 or not len(values):
        raise ValueError(f"{label}: missing timestamps")
    if np.any(np.diff(values) < (0 if duplicates else 1)):
        raise ValueError(f"{label}: timestamps must increase")
    return values


def _sdk_time(raw: bytes, metadata: dict, camera: str) -> tuple[int, str, int]:
    text = _Wire(raw, cdr=True).string()
    fields = dict(item.split("=", 1) for item in text.split(";") if "=" in item)
    try:
        stamp = float(fields["timestamp"])
        if not np.isfinite(stamp):
            raise ValueError("nonfinite timestamp")
        timestamp = int(stamp * 1_000_000)
        domain = fields["timestamp_domain"].lower().replace(" ", "_")
        counter = int(
            fields["Frame Counter"]
            if "Frame Counter" in fields
            else fields["Frame number"]
        )
    except (KeyError, ValueError, OverflowError) as error:
        raise ValueError(f"{camera}: invalid native frame timing metadata") from error
    if domain in ("global_time", "system_time"):
        if timestamp < WALL_CLOCK_MIN_NS:
            raise ValueError(f"{camera}: non-epoch {domain} timestamp")
    elif domain == "hardware_clock":
        receipt = metadata.get("camera_capture_receipts", {}).get(camera, {})
        measurement = receipt.get("clock_mapping_receipt") or {}
        if not str(measurement.get("frame_timestamp_domain", "")).endswith(
            "hardware_clock"
        ):
            raise ValueError(
                f"{camera}: hardware clock lacks a matching recorded host mapping"
            )
        timestamp += int(metadata["realsense_clock_offsets"][camera])
    else:
        raise ValueError(f"{camera}: unsupported timestamp domain {domain!r}")
    return timestamp, domain, counter


class NativeColor:
    """Reiterable color stream; metadata scans do not decode pixel arrays."""

    def __init__(self, path: Path, metadata: dict):
        self.path, self.metadata = path, metadata
        self.camera, self.cdr = path.stem, path.suffix == ".db3"
        if self.cdr:
            with self._database() as con:
                topics = con.execute(
                    "SELECT id,name,type,serialization_format FROM topics"
                ).fetchall()
                matches = [
                    row
                    for row in topics
                    if row[1].endswith(
                        ("/Color_0/image/data", "/Color_0/image/data/compressed")
                    )
                ]
                if len(matches) != 1:
                    raise ValueError(f"{path}: expected one CDR color Image topic")
                self.compressed = matches[0][1].endswith("/compressed")
                msgtype = (
                    "sensor_msgs/msg/CompressedImage"
                    if self.compressed
                    else "sensor_msgs/msg/Image"
                )
                if matches[0][2:] != (msgtype, "cdr"):
                    raise ValueError(
                        f"{path}: color topic type/serialization does not match its path"
                    )
                self.topic = matches[0][0]
                name = matches[0][1].split("/image/")[0] + "/image/metadata"
                meta_topics = [row for row in topics if row[1] == name]
                if len(meta_topics) != 1 or meta_topics[0][2:] != (
                    "std_msgs/msg/String",
                    "cdr",
                ):
                    raise ValueError(f"{path}: missing native color timing metadata")
                rows = con.execute(
                    "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY id",
                    (meta_topics[0][0],),
                ).fetchall()
                self.native_times = {}
                for record_time, raw in rows:
                    if record_time in self.native_times:
                        raise ValueError(f"{path}: duplicate native timing metadata")
                    self.native_times[record_time] = _sdk_time(
                        raw, metadata, self.camera
                    )
        else:
            self.compressed = False

    @contextmanager
    def _database(self):
        connection = sqlite3.connect(
            self.path.resolve().as_uri() + "?mode=ro", uri=True
        )
        try:
            yield connection
        finally:
            connection.close()

    def messages(self, *, scan=False):
        if self.cdr:
            with self._database() as con:
                field = (
                    "substr(data,1,1024)" if scan and not self.compressed else "data"
                )
                query = f"SELECT timestamp,length(data),{field} FROM messages WHERE topic_id=? ORDER BY id"
                for record_time, length, raw in con.execute(query, (self.topic,)):
                    header = image_header(
                        raw,
                        cdr=True,
                        total_length=length if scan else None,
                        compressed=self.compressed,
                    )
                    if (
                        record_time not in self.native_times
                        or header["timestamp_ns"] != record_time
                    ):
                        raise ValueError(
                            f"{self.path}: Image/native timing association is ambiguous"
                        )
                    timestamp, domain, counter = self.native_times[record_time]
                    yield timestamp, counter, domain, header, raw
        else:
            from rosbags.rosbag1 import Reader

            with Reader(self.path) as reader:
                matches = [
                    c
                    for c in reader.connections
                    if c.topic.endswith("/Color_0/image/data")
                ]
                if len(matches) != 1 or matches[0].msgtype != "sensor_msgs/msg/Image":
                    raise ValueError(f"{self.path}: expected one color Image topic")
                if matches[0].digest != "060021388200f6f0f447d0fcd9c64743":
                    raise ValueError(f"{self.path}: unexpected ROS1 Image digest")
                for _, _, raw in reader.messages(connections=matches):
                    header = image_header(raw, cdr=False)
                    if header["timestamp_ns"] < WALL_CLOCK_MIN_NS:
                        raise ValueError(
                            f"{self.path}: ROS1 Image timestamp is not host epoch time"
                        )
                    yield (
                        header["timestamp_ns"],
                        header["sequence"],
                        "ros1_epoch_header",
                        header,
                        raw,
                    )


def _nearest(times, grid):
    right = np.searchsorted(times, grid).clip(0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(
        np.abs(times[left] - grid) <= np.abs(times[right] - grid), left, right
    )


def _transform(value: dict) -> np.ndarray:
    if not value.get("success"):
        raise ValueError("Camera calibration is not successful")
    matrix = np.eye(4)
    matrix[:3, :3] = value["rotation_matrix"]
    matrix[:3, 3] = np.asarray(value["translation_vector"]).reshape(3)
    return validate_transform(matrix)


def _crop_intrinsics(receipt, settings, sizes):
    width, height = int(receipt["native_width"]), int(receipt["native_height"])
    crop = settings.get("policy_crop", receipt.get("policy_crop"))
    x, y, w, h = tuple(map(int, crop)) if crop is not None else (0, 0, width, height)
    if x < 0 or y < 0 or min(w, h) <= 0 or x + w > width or y + h > height:
        raise ValueError("Recorded camera crop exceeds native image")
    if receipt.get("policy_crop") != crop:
        raise ValueError("Recorded camera config and capture receipt crops disagree")
    if (w, h) != (int(receipt["width"]), int(receipt["height"])):
        raise ValueError("Recorded crop and capture receipt dimensions disagree")
    side = min(w, h)
    square_x, square_y = (w - side) // 2, (h - side) // 2
    # fx/fy/cx/cy in capture receipts already include the recorded policy crop.
    intrinsic = np.array(
        [
            [receipt["fx"], 0, receipt["cx"] - square_x],
            [0, receipt["fy"], receipt["cy"] - square_y],
            [0, 0, 1],
        ],
        float,
    )
    if not np.isfinite(intrinsic).all() or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ValueError("Invalid recorded camera intrinsics")
    matrices = {}
    for size in sizes:
        k = intrinsic.copy()
        k[:2] *= size / side
        matrices[size] = k
    return (x + square_x, y + square_y, side, side), matrices


def _timed_field(data, arm, kind):
    key = f"{arm}_{'state' if kind == 'state' else 'command'}_timestamp_ns"
    valid_key = key.removesuffix("_ns") + "_valid"
    field = f"{arm}_{'joint_pos_7d' if kind == 'state' else 'joint_cmd'}"
    values = np.asarray(data[field], dtype=np.float64)
    if values.shape != (len(data["timestamps"]), 7) or not np.isfinite(values).all():
        raise ValueError(f"{field}: expected finite N x 7 data")
    valid = np.ones(len(values), dtype=bool)
    if key in data:
        if valid_key not in data or data[valid_key].shape != valid.shape:
            raise ValueError(f"{key}: missing validity mask")
        valid = data[valid_key].astype(bool) & (data[key] >= 0)
        times = data[key][valid]
    else:
        times = data["timestamps"]
    values = values[valid]
    times = _strict_times(times, key, duplicates=True)
    # Recorder rows can repeat one cached feedback/command snapshot. Keep the
    # final coherent snapshot for each timestamp, without interpolating intent.
    keep = np.r_[np.diff(times) != 0, True]
    return (
        times[keep],
        values[keep],
        np.flatnonzero(valid)[keep],
        key if key in data else "timestamps",
    )


def _state_brackets(times, grid):
    """Intervals actually spanned by interpolation, zero at a measured sample."""
    left = np.searchsorted(times, grid, side="right") - 1
    right = np.searchsorted(times, grid, side="left")
    if left.min() < 0 or right.max() >= len(times):
        raise ValueError("State interpolation would extrapolate")
    return left, right, times[right] - times[left]


def _prepare_episode(path: Path, sizes, hz, max_skew_ms):
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata.get("complete") is not True:
        raise ValueError(f"{path}: incomplete episode cannot be exported")
    with np.load(path / "robot_data.npz", allow_pickle=False) as archive:
        data = dict(archive)
    row_times = _strict_times(data["timestamps"], f"{path}/robot timestamps")
    arms = [
        arm for arm in ("follower_l", "follower_r") if f"{arm}_joint_pos_7d" in data
    ]
    if not arms or (arms == ["follower_r"]):
        raise ValueError(
            "Expected the left follower, optionally followed by the right follower"
        )
    configured = (
        metadata.get("recording_config", {})
        .get("robot", {})
        .get("active_followers", arms)
    )
    if set(configured) != set(arms):
        raise ValueError("Active follower config and saved robot streams disagree")
    robot = {
        arm: {kind: _timed_field(data, arm, kind) for kind in ("state", "command")}
        for arm in arms
    }
    files = {p.stem: p for p in list_realsense_recordings(path / "cameras")}
    names = metadata["cameras"]
    if not names or len(set(names)) != len(names) or set(files) != set(names):
        raise ValueError("Episode camera list and native camera files disagree")
    banks = {}
    for camera in names:
        source = NativeColor(files[camera], metadata)
        records = [
            (t, counter, domain, h)
            for t, counter, domain, h, _ in source.messages(scan=True)
        ]
        times = _strict_times([r[0] for r in records], f"{camera} frame timestamps")
        receipt = metadata["camera_capture_receipts"][camera]
        if any(
            (r[3]["width"], r[3]["height"])
            != (receipt["native_width"], receipt["native_height"])
            for r in records
        ):
            raise ValueError(f"{camera}: image dimensions differ from capture receipt")
        settings = dict(
            metadata.get("recording_config", {}).get("cameras", {}).get(camera, {})
        )
        settings.setdefault(
            "role",
            {
                "scene_camera": "scene",
                "left_wrist_camera": "left_wrist",
                "right_wrist_camera": "right_wrist",
            }.get(camera),
        )
        if settings["role"] not in ("scene", "left_wrist", "right_wrist"):
            raise ValueError(f"{camera}: recording lacks a recognized camera role")
        if settings["role"] == "right_wrist" and "follower_r" not in arms:
            raise ValueError(f"{camera}: right wrist camera has no right arm state")
        crop, intrinsics = _crop_intrinsics(receipt, settings, sizes)
        banks[camera] = {
            "source": source,
            "times": times,
            "crop": crop,
            "intrinsics": intrinsics,
            "counters": np.array([r[1] for r in records]),
            "timestamp_domains": sorted({r[2] for r in records}),
            "settings": settings,
        }
    low = max(
        [int(row_times[0])]
        + [int(b["times"][0]) for b in banks.values()]
        + [int(s[0][0]) for arm in robot.values() for s in arm.values()]
    )
    high = min(
        [int(row_times[-1])]
        + [int(b["times"][-1]) for b in banks.values()]
        + [int(arm["state"][0][-1]) for arm in robot.values()]
    )
    reference = next(
        (n for n in names if banks[n]["settings"].get("role") == "scene"),
        "scene_camera" if "scene_camera" in names else names[0],
    )
    scene_times = banks[reference]["times"]
    anchor_index = np.searchsorted(scene_times, low)
    if anchor_index == len(scene_times) or scene_times[anchor_index] > high:
        raise ValueError("Camera and robot timing ranges do not overlap")
    anchor = int(scene_times[anchor_index])
    period = 1e9 / hz
    grid = anchor + np.rint(
        np.arange(int((high - anchor) / period) + 1) * period
    ).astype(np.int64)
    if len(grid) < 2:
        raise ValueError("Episode has fewer than two overlapping policy frames")
    diagnostics = {
        "robot_row_max_gap_ms": float(np.diff(row_times).max() / 1e6),
        "robot_row_gaps_over_50ms": int(np.count_nonzero(np.diff(row_times) > 50e6)),
        "cameras": {},
    }
    for camera, bank in banks.items():
        times = bank["times"]
        bank["selection"] = indices = _nearest(times, grid)
        skew = (times[indices] - grid) / 1e6
        if np.abs(skew).max() > max_skew_ms:
            raise ValueError(
                f"{camera}: selected image skew {np.abs(skew).max():.3f}ms exceeds {max_skew_ms:.3f}ms"
            )
        counters = bank["counters"]
        diagnostics["cameras"][camera] = {
            "source_frames": len(times),
            "selected_minus_grid_ms_min": float(skew.min()),
            "selected_minus_grid_ms_max": float(skew.max()),
            "max_absolute_skew_ms": float(np.abs(skew).max()),
            "repeated_policy_frames": int(np.count_nonzero(np.diff(indices) == 0)),
            "max_source_interval_ms": float(np.diff(times).max() / 1e6)
            if len(times) > 1
            else 0.0,
            "counter_missing_frames": int(np.maximum(np.diff(counters) - 1, 0).sum()),
            "timestamp_domains": bank["timestamp_domains"],
        }
    return metadata, robot, banks, grid, diagnostics


def export_episodes(
    raw_root: Path,
    output: Path,
    episodes: list[str],
    *,
    sizes=(128, 256),
    hz=30.0,
    validation_episodes=(),
    max_skew_ms=None,
) -> dict:
    """Write a new HDF5 atomically; explicit IDs are the reviewed success list."""
    import h5py

    raw_root, output = Path(raw_root), Path(output)
    sizes = tuple(dict.fromkeys(sizes))
    if not sizes or any(not isinstance(s, int) or s <= 0 for s in sizes):
        raise ValueError("Image sizes must be positive integers")
    if not np.isfinite(hz) or hz <= 0 or hz > 1000:
        raise ValueError("Policy frequency must be finite and between 0 and 1000 Hz")
    max_skew_ms = 1000.0 / hz if max_skew_ms is None else max_skew_ms
    if not np.isfinite(max_skew_ms) or max_skew_ms <= 0:
        raise ValueError("Maximum image skew must be finite and positive")
    if (
        not episodes
        or len(set(episodes)) != len(episodes)
        or any(Path(e).name != e or e in (".", "..") for e in episodes)
    ):
        raise ValueError("Provide unique episode directory names explicitly")
    if not set(validation_episodes) <= set(episodes):
        raise ValueError("Validation episodes must belong to the explicit episode list")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=output.name + ".", suffix=".tmp", dir=output.parent
    )
    os.close(fd)
    temp = Path(temp_name)
    report = {
        "schema": SCHEMA,
        "output": str(output.resolve()),
        "episodes": {},
        "depth_exported": False,
        "orientation": "native",
        "policy_hz": hz,
        "image_sizes": list(sizes),
        "command_gripper": "binary intent:0 closed,1 open",
        "arm_order": ["left", "right"],
        "selection": "explicit reviewed successful episode IDs; no local database read or modified",
    }
    try:
        with h5py.File(temp, "w") as f:
            f.attrs.update(
                schema=SCHEMA,
                depth_exported=False,
                image_orientation="native",
                image_color="RGB",
                action_type="abs_joint_pos",
                policy_hz=hz,
            )
            root, masks = f.create_group("data"), {"train": [], "valid": []}
            contract = None
            for episode in episodes:
                metadata, robot, banks, grid, diagnostics = _prepare_episode(
                    raw_root / episode, sizes, hz, max_skew_ms
                )
                current = (
                    tuple(robot),
                    tuple(banks),
                    metadata["task_name"],
                    metadata["task_instruction"],
                )
                if contract is not None and current != contract:
                    raise ValueError(
                        "Episodes in one export must share arms, cameras, task and instruction"
                    )
                contract = current
                split = "valid" if episode in validation_episodes else "train"
                name = "demo_" + episode
                masks[split].append(name.encode())
                g = root.create_group(name)
                g.attrs.update(
                    num_samples=len(grid),
                    source_episode=episode,
                    policy_hz=hz,
                    task=metadata["task_name"],
                    prompt=metadata["task_instruction"],
                    arm_order=json.dumps(list(robot)),
                    diagnostics=json.dumps(diagnostics),
                )
                obs = g.create_group("obs")
                obs.create_dataset("policy_timestamp_ns", data=grid)
                actions, arm_reports, state_gap_masks = [], {}, []
                for index, (arm, streams) in enumerate(robot.items()):
                    times, states, _, state_clock = streams["state"]
                    origin = int(times[0])
                    q = np.column_stack(
                        [
                            np.interp(grid - origin, times - origin, states[:, j])
                            for j in range(7)
                        ]
                    )
                    left, right, gap_ns = _state_brackets(times, grid)
                    obs.create_dataset(
                        f"robot{index}_state_source_left_timestamp_ns", data=times[left]
                    )
                    obs.create_dataset(
                        f"robot{index}_state_source_right_timestamp_ns",
                        data=times[right],
                    )
                    obs.create_dataset(
                        f"robot{index}_state_interpolation_interval_ns", data=gap_ns
                    )
                    state_gap_masks.append(gap_ns > 50_000_000)
                    obs.create_dataset(
                        f"robot{index}_joint_pos", data=q[:, :6].astype(np.float32)
                    )
                    obs.create_dataset(
                        f"robot{index}_gripper_jaw_width",
                        data=q[:, 6:].astype(np.float32),
                    )
                    times, commands, source_rows, command_clock = streams["command"]
                    indices = np.searchsorted(times, grid, side="right") - 1
                    if np.any(indices < 0):
                        raise ValueError(
                            "Command selection precedes first available command"
                        )
                    if not np.isin(commands[:, 6], [0.0, 1.0]).all():
                        raise ValueError(
                            f"{arm}: command gripper is not binary operator intent"
                        )
                    action = commands[indices].copy()
                    if command_clock == "timestamps":
                        # Older single-arm rows only sample targets. Preserve the
                        # established linear arm-target interpolation plus causal
                        # gripper hold. New recordings use causal API snapshots.
                        origin = int(times[0])
                        action[:, :6] = np.column_stack(
                            [
                                np.interp(grid - origin, times - origin, commands[:, j])
                                for j in range(6)
                            ]
                        )
                    actions.append(action)
                    g.create_dataset(
                        f"{arm}_command_source_timestamp_ns", data=times[indices]
                    )
                    g.create_dataset(
                        f"{arm}_command_source_robot_index", data=source_rows[indices]
                    )
                    arm_reports[arm] = {
                        "state_clock": state_clock,
                        "command_clock": command_clock,
                        "state_max_gap_ms": float(
                            np.diff(streams["state"][0]).max() / 1e6
                        )
                        if len(streams["state"][0]) > 1
                        else 0.0,
                        "command_max_gap_ms": float(np.diff(times).max() / 1e6)
                        if len(times) > 1
                        else 0.0,
                        "max_selected_command_age_ms": float(
                            np.max(grid - times[indices]) / 1e6
                        ),
                    }
                action = np.concatenate(actions, axis=1).astype(np.float32)
                g.create_dataset("actions_joint_pos", data=action)
                g.create_dataset("actions", data=action)
                g.create_dataset("task_emb", data=np.zeros(512, np.float32))
                g.create_dataset(
                    "robot_state_gap_over_50ms", data=np.any(state_gap_masks, axis=0)
                )
                calibration = json.loads(
                    (raw_root / episode / "calibration_results.json").read_text()
                )
                if calibration.get("coordinate_frame") != "left_arm_base":
                    raise ValueError("Expected camera calibration in left_arm_base")
                cal = g.create_group("calibration")
                cal.attrs["source_json"] = json.dumps(calibration)
                cal.attrs["coordinate_frame"] = "left_arm_base"
                transform = left_from_right(calibration)
                if len(robot) == 2 and transform is None:
                    raise ValueError(
                        "Bimanual export requires the inter-base calibration"
                    )
                if transform is not None:
                    cal.create_dataset("T_left_from_right", data=transform)
                if "table" in calibration and "T_base_table" in calibration["table"]:
                    cal.create_dataset(
                        "T_base_table",
                        data=validate_transform(calibration["table"]["T_base_table"]),
                    )
                for camera, bank in banks.items():
                    selections = bank["selection"]
                    g.create_dataset(f"{camera}_source_frame_index", data=selections)
                    obs.create_dataset(
                        f"{camera}_timestamp_ns", data=bank["times"][selections]
                    )
                    c = cal.create_group(camera)
                    c.attrs["recorded_settings"] = json.dumps(bank["settings"])
                    c.attrs["capture_receipt"] = json.dumps(
                        metadata["camera_capture_receipts"][camera]
                    )
                    c.create_dataset("native_square_crop", data=bank["crop"])
                    camera_cal = calibration.get("cameras", {}).get(camera, {})
                    role = bank["settings"]["role"]
                    if role.endswith("_wrist") and "hand_eye_calibration" in camera_cal:
                        c.attrs["arm_base_frame"] = (
                            "right_arm_base"
                            if role == "right_wrist"
                            else "left_arm_base"
                        )
                        c.create_dataset(
                            "T_ee_camera",
                            data=_transform(camera_cal["hand_eye_calibration"]),
                        )
                    elif role == "scene" and "extrinsics" in camera_cal:
                        if (
                            camera_cal["extrinsics"].get(
                                "reference_frame", "left_arm_base"
                            )
                            != "left_arm_base"
                        ):
                            raise ValueError(
                                f"{camera}: scene extrinsics are not in left_arm_base"
                            )
                        c.create_dataset(
                            "T_left_base_camera",
                            data=_transform(camera_cal["extrinsics"]),
                        )
                    else:
                        raise ValueError(f"{camera}: missing camera calibration")
                    datasets = {}
                    for size in sizes:
                        c.create_dataset(
                            f"intrinsic_{size}", data=bank["intrinsics"][size]
                        )
                        datasets[size] = obs.create_dataset(
                            f"{camera}_image_{size}",
                            shape=(len(grid), size, size, 3),
                            dtype=np.uint8,
                            chunks=(1, size, size, 3),
                            compression="lzf",
                        )
                    obs[f"{camera}_image"] = datasets[sizes[0]]
                    cursor = 0
                    for source_index, (_, _, _, _, raw) in enumerate(
                        bank["source"].messages()
                    ):
                        if cursor == len(grid):
                            break
                        if source_index != selections[cursor]:
                            continue
                        image = image_pixels(
                            raw,
                            cdr=bank["source"].cdr,
                            compressed=bank["source"].compressed,
                        )
                        x, y, w, h = bank["crop"]
                        image = image[y : y + h, x : x + w]
                        end = int(
                            np.searchsorted(selections, source_index, side="right")
                        )
                        for size, dataset in datasets.items():
                            rgb = cv2.resize(
                                image, (size, size), interpolation=cv2.INTER_AREA
                            )
                            dataset[cursor:end] = rgb
                        cursor = end
                    if cursor != len(grid):
                        raise ValueError(
                            f"{camera}: stream ended before all selected frames were written"
                        )
                diagnostics["arms"] = arm_reports
                g.attrs["diagnostics"] = json.dumps(diagnostics)
                report["episodes"][episode] = dict(
                    frames=len(grid),
                    split=split,
                    action_dims=action.shape[1],
                    **diagnostics,
                )
                if diagnostics["robot_row_gaps_over_50ms"]:
                    print(
                        f"WARNING {episode}: {diagnostics['robot_row_gaps_over_50ms']} robot sampling gap(s) exceed50ms; max{diagnostics['robot_row_max_gap_ms']:.1f}ms",
                        flush=True,
                    )
            mask = f.create_group("mask")
            for split, names in masks.items():
                mask.create_dataset(split, data=np.asarray(names, dtype="S"))
            root.attrs["total"] = sum(
                value["frames"] for value in report["episodes"].values()
            )
            f.attrs["export_report"] = json.dumps(report)
            f.flush()
        with h5py.File(temp, "r") as f:
            for episode, info in report["episodes"].items():
                g = f["data"]["demo_" + episode]
                if g["actions_joint_pos"].shape != (
                    info["frames"],
                    info["action_dims"],
                ):
                    raise ValueError("HDF5 action readback failed")
                if not np.isfinite(g["actions_joint_pos"][()]).all():
                    raise ValueError("HDF5 contains nonfinite actions")
        with temp.open("rb") as stream:
            os.fsync(stream.fileno())
        # Hard-link publication refuses a concurrently created destination.
        os.link(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        required=True,
        type=Path,
        help="task directory containing raw episode folders",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="new HDF5 file; existing outputs are never overwritten",
    )
    parser.add_argument(
        "--episodes",
        required=True,
        nargs="+",
        help="explicit reviewed successful episode IDs",
    )
    parser.add_argument(
        "--validation-episodes",
        nargs="*",
        default=[],
        help="explicit episode holdout; all other selected episodes train",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[128, 256],
        help="center-square RGB sizes; first also has unsuffixed image keys",
    )
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument(
        "--max-image-skew-ms", type=float, help="default one policy period"
    )
    args = parser.parse_args(argv)
    report = export_episodes(
        args.raw_root,
        args.output,
        args.episodes,
        sizes=args.sizes,
        hz=args.hz,
        validation_episodes=args.validation_episodes,
        max_skew_ms=args.max_image_skew_ms,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
