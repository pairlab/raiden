"""Saved timing audit checks run entirely on fake playback and temporary files."""

import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from raiden import timing_audit as audit


def sample(timestamp_ns, counter, domain="global_time"):
    return {"timestamp_ns": timestamp_ns, "frame_number": counter,
            "timestamp_domain": domain, "native_metadata": {"frame_counter": counter}}


class Frame:
    def __init__(self, stream, timestamp_ms=10, counter=1, fail=False):
        self.stream, self.timestamp_ms, self.counter, self.fail = stream, timestamp_ms, counter, fail

    def get_profile(self):
        return SimpleNamespace(stream_type=lambda: self.stream)

    def get_timestamp(self):
        if self.fail:
            raise RuntimeError("bad saved timestamp")
        return self.timestamp_ms

    def get_frame_timestamp_domain(self):
        return "hardware_clock"

    def get_frame_number(self):
        return self.counter

    def supports_frame_metadata(self, key):
        return key == "frame_counter"

    def get_frame_metadata(self, key):
        return self.counter


class Sensor:
    def __init__(self, stream, fail_start=False):
        self.stream, self.fail_start = stream, fail_start
        self.opened = self.stopped = self.closed = False
        self.callback = None

    def get_stream_profiles(self):
        return [SimpleNamespace(stream_type=lambda: self.stream)]

    def open(self, profiles):
        self.opened = True

    def start(self, callback):
        self.callback = callback
        if self.fail_start:
            raise RuntimeError("saved stream start failed")

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class Playback:
    def __init__(self, sensors, frames):
        self.sensors, self.frames = sensors, frames
        self.real_time = None
        self.paused = False
        self.status = "stopped"

    def set_real_time(self, value):
        self.real_time = value

    def pause(self):
        self.paused = True

    def query_sensors(self):
        return self.sensors

    def resume(self):
        assert self.paused
        assert all(sensor.callback is not None for sensor in self.sensors)
        for frame in self.frames:
            next(s for s in self.sensors if s.stream == frame.stream).callback(frame)
        self.status = "stopped"

    def current_status(self):
        return self.status


def fake_sdk(sensors, frames):
    playback = Playback(sensors, frames)
    context = SimpleNamespace(unloaded=[], loaded=[], settings=None)

    def load(path):
        context.loaded.append(path)
        return playback

    context.load_device = load
    context.unload_device = lambda path: context.unloaded.append(path)

    def make_context(settings):
        context.settings = settings
        return context

    sdk = SimpleNamespace(
        context=make_context, stream=SimpleNamespace(color="color", depth="depth"),
        product_line=SimpleNamespace(sw_only=256),
        frame_metadata_value=SimpleNamespace(frame_counter="frame_counter"),
        playback_status=SimpleNamespace(stopped="stopped"),
    )
    return sdk, playback, context


class TimingAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bag = self.root / "test.bag"
        self.bag.write_bytes(b"saved recording remains unchanged")

    def tearDown(self):
        self.temp.cleanup()

    def test_native_reader_preserves_independent_stream_frames_and_closes_at_eof(self):
        sensors = [Sensor("color"), Sensor("depth")]
        sdk, playback, context = fake_sdk(sensors, [
            Frame("color", 10, 1), Frame("depth", 12, 1), Frame("depth", 45, 2),
            Frame("color", 76, 3),
        ])
        result = audit.read_bag(self.bag, rs=sdk)
        self.assertEqual(result["read"]["stop_reason"], "eof")
        self.assertTrue(result["read"]["complete"])
        self.assertEqual([r["frame_number"] for r in result["samples"]["color"]], [1, 3])
        self.assertEqual([r["frame_number"] for r in result["samples"]["depth"]], [1, 2])
        self.assertFalse(playback.real_time)
        self.assertTrue(all(sensor.stopped and sensor.closed for sensor in sensors))
        self.assertEqual(context.loaded, [str(self.bag)])
        self.assertEqual(context.settings, {"device-mask": 256})
        self.assertEqual(context.unloaded, [str(self.bag)])
        self.assertEqual(self.bag.read_bytes(), b"saved recording remains unchanged")

    def test_callback_failure_and_partial_start_still_stop_close_and_unload(self):
        for failure in ("metadata", "start"):
            with self.subTest(failure=failure):
                sensors = [Sensor("color"), Sensor("depth", fail_start=failure == "start")]
                sdk, _, context = fake_sdk(sensors, [Frame("color", fail=True)])
                result = audit.read_bag(self.bag, rs=sdk)
                self.assertFalse(result["read"]["complete"])
                self.assertEqual(result["read"]["stop_reason"],
                                 "metadata_error" if failure == "metadata" else "playback_error")
                self.assertTrue(all(sensor.stopped and sensor.closed for sensor in sensors))
                self.assertEqual(context.unloaded, [str(self.bag)])

    def test_playback_with_profiles_but_no_frames_is_incomplete_and_closes(self):
        sensors = [Sensor("color"), Sensor("depth")]
        sdk, _, context = fake_sdk(sensors, [])
        result = audit.read_bag(self.bag, rs=sdk)
        self.assertEqual(result["read"]["stop_reason"], "no_frames")
        self.assertEqual(result["read"]["frames_read"], 0)
        self.assertFalse(result["read"]["complete"])
        self.assertTrue(all(sensor.stopped and sensor.closed for sensor in sensors))
        self.assertEqual(context.unloaded, [str(self.bag)])

    def test_color_only_playback_remains_complete(self):
        sdk, _, _ = fake_sdk([Sensor("color")], [Frame("color")])
        result = audit.read_bag(self.bag, rs=sdk)
        self.assertTrue(result["read"]["complete"])
        self.assertEqual(result["read"]["stop_reason"], "eof")
        self.assertEqual(len(result["samples"]["color"]), 1)
        self.assertEqual(result["samples"]["depth"], [])

    def test_frame_and_time_limits_report_partial_reads_and_release_resources(self):
        sensors = [Sensor("color")]
        sdk, _, context = fake_sdk(sensors, [Frame("color", counter=i) for i in range(4)])
        result = audit.read_bag(self.bag, max_frames=2, rs=sdk)
        self.assertEqual(result["read"]["stop_reason"], "frame_limit")
        self.assertEqual(result["read"]["frames_read"], 2)
        self.assertFalse(result["read"]["complete"])
        self.assertEqual(context.unloaded, [str(self.bag)])
        sensors = [Sensor("color")]
        sdk, playback, _ = fake_sdk(sensors, [])
        playback.resume = lambda: setattr(playback, "status", "playing")
        with patch.object(audit.time, "monotonic", side_effect=[0.0, 2.0]):
            result = audit.read_bag(self.bag, max_seconds=1, rs=sdk)
        self.assertEqual(result["read"]["stop_reason"], "time_limit")
        self.assertTrue(sensors[0].stopped and sensors[0].closed)

    def test_context_initialization_failure_is_a_visible_partial_read(self):
        sdk = SimpleNamespace(context=lambda settings: (_ for _ in ()).throw(RuntimeError("udev monitor unavailable")),
                              product_line=SimpleNamespace(sw_only=256))
        result = audit.read_bag(self.bag, rs=sdk)
        self.assertEqual(result["read"]["stop_reason"], "playback_error")
        self.assertFalse(result["read"]["complete"])
        self.assertIn("udev monitor", result["read"]["errors"][0])

    def test_native_summary_detects_gap_duplicate_and_rgb_depth_offset(self):
        color = [sample(0, 1, "hardware_clock"), sample(30_000_000, 2, "hardware_clock"),
                 sample(90_000_000, 4, "hardware_clock"), sample(90_000_000, 4, "hardware_clock")]
        depth = [sample(t + 2_000_000, i + 1, "hardware_clock")
                 for i, t in enumerate((0, 30_000_000, 60_000_000, 90_000_000))]
        summary = audit._stream_summary(color)
        self.assertEqual(summary["inferred_missing_frames"], 1)
        self.assertEqual(summary["frame_counter_gap_events"], 1)
        self.assertEqual(summary["duplicate_frame_counters"], 1)
        self.assertEqual(summary["inter_frame_ms"]["max"], 60)
        pair = audit._rgb_depth_summary(color, depth)["by_timestamp_domain"][0]
        self.assertEqual(pair["selected_minus_reference_ms"]["p50"], 2)
        self.assertEqual(pair["absolute_residual_ms"]["max"], 2)

    def test_host_mapping_preserves_sdk_global_and_reports_relative_unknown_provenance(self):
        records = [sample(1_000_000, 1, "hardware_clock"), sample(2_000_000, 2, "unknown")]
        mapped, report = audit._map_host_times(records, None, {})
        self.assertEqual(len(mapped), 0)
        self.assertEqual(report["unmapped_frames"], 2)
        mapped, report = audit._map_host_times(records, 5_000_000, {})
        np.testing.assert_array_equal(mapped, [6_000_000])
        self.assertFalse(report["measurement_provenance_available"])
        self.assertIsNone(report["physical_clock_error_bound_ms"])
        mapped, report = audit._map_host_times([sample(10_000_000, 1)], 99_000_000, {})
        np.testing.assert_array_equal(mapped, [10_000_000])
        self.assertEqual(report["methods"], {"sdk_global_time": 1})
        measurement = {"host_wait_start_ns": 100, "host_receive_ns": 2_000_100}
        _, report = audit._map_host_times(records, 5_000_000,
                                         {"clock_mapping_receipt": measurement})
        self.assertEqual(report["measurement_wait_width_ms"], 2)
        self.assertTrue(report["measurement_provenance_available"])

    def test_timestamp_domain_change_does_not_create_false_interval_or_mapping(self):
        records = [sample(10_000_000, 1, "hardware_clock"),
                   sample(20_000_000, 2, "hardware_clock"),
                   sample(1_000_000_000_000, 3), sample(1_000_010_000_000, 4)]
        summary = audit._stream_summary(records)
        self.assertEqual(summary["timestamp_domain_transitions"], 1)
        self.assertIsNone(summary["time_span_ms"])
        self.assertEqual(summary["inter_frame_ms"]["max"], 10)
        self.assertEqual(summary["inter_frame_ms"]["count"], 2)
        times, mapping = audit._map_host_times(records, 123, {
            "clock_mapping_receipt": {"frame_timestamp_domain": "global_time"}})
        np.testing.assert_array_equal(times, [1_000_000_000_000, 1_000_010_000_000])
        self.assertEqual(mapping["unmapped_frames"], 2)

    def test_episode_known_cross_camera_offset_without_exposing_all_frames(self):
        (self.root / "cameras").mkdir()
        for name in ("scene", "wrist", "unknown"):
            (self.root / "cameras" / f"{name}.bag").write_bytes(b"saved")
        (self.root / "metadata.json").write_text(json.dumps({
            "realsense_clock_offsets": {"wrist": 1_000_000_000},
        }))

        def reader(path, **kwargs):
            base = [0, 30_000_000, 60_000_000]
            if path.stem == "scene":
                records = [sample(t + 1_000_000_000, i) for i, t in enumerate(base)]
            else:
                records = [sample(t + 4_000_000, i, "hardware_clock") for i, t in enumerate(base)]
            return {"samples": {"color": records, "depth": []},
                    "read": {"complete": True, "stop_reason": "eof"}}

        result = audit.audit_episode(self.root, reference_camera="scene", bag_reader=reader)
        known = result["cross_camera_host_timestamp_nearest_selection"]["wrist"]
        self.assertEqual(known["selected_minus_reference_ms"]["p50"], 4)
        self.assertEqual(known["absolute_residual_ms"]["max"], 4)
        self.assertEqual(known["reference_frames_outside_target_coverage"], 1)
        self.assertIn("missing", result["cross_camera_host_timestamp_nearest_selection"]["unknown"])
        encoded = json.dumps(result, allow_nan=False)
        self.assertNotIn('"samples"', encoded)
        self.assertNotIn('"selected_source_indices"', encoded)

    def test_robot_feedback_read_age_brackets_command_age_and_held_issue(self):
        np.savez(self.root / "robot_data.npz", timestamps=[1_000_000, 11_000_000, 21_000_000],
                 readout_end_ns=[4_000_000, 14_000_000, 24_000_000],
                 follower_l_joint_pos=np.zeros((3, 6)),
                 follower_l_joint_cmd=np.zeros((3, 7)),
                 follower_l_read_start_ns=[1_000_000, 11_000_000, 21_000_000],
                 follower_l_read_end_ns=[3_000_000, 13_000_000, 23_000_000],
                 follower_l_read_start_monotonic_ns=[100, 200, 300],
                 follower_l_read_end_monotonic_ns=[1_000_100, 1_000_200, 1_000_300],
                 follower_l_state_timestamp_ns=[2_000_000, 2_000_000, -1],
                 follower_l_state_timestamp_valid=[True, True, False],
                 follower_l_state_timestamp_source=["host_cached_feedback_read"] * 2 + ["unavailable"],
                 follower_l_command_timestamp_ns=[500_000, 500_000, -1],
                 follower_l_command_timestamp_valid=[True, True, False],
                 follower_l_command_timestamp_source=["host_command_api_issue"] * 2 + ["unavailable"])
        result = audit.audit_robot(self.root / "robot_data.npz")["arms"]["follower_l"]
        self.assertEqual(result["feedback_read_bracket_ms"]["max"], 1)
        self.assertEqual(result["feedback_read_bracket_ms"]["clock"], "host_monotonic")
        self.assertEqual(result["feedback_read_age_ms"]["max"], 11)
        self.assertEqual(result["feedback_read_age_ms"]["unavailable_rows"], 1)
        self.assertEqual(result["command_age_ms"]["max"], 13.5)
        self.assertEqual(result["command_age_ms"]["reference"], "row_capture_completion")
        self.assertEqual(result["rows_reusing_previous_command_issue"], 1)
        self.assertEqual(result["unchanged_consecutive_command_vectors"], 2)

    def test_legacy_robot_does_not_infer_freshness_from_unchanged_joints(self):
        np.savez(self.root / "robot_data.npz", timestamps=[1, 2],
                 follower_l_joint_pos=np.zeros((2, 6)))
        result = audit.audit_robot(self.root / "robot_data.npz")["arms"]["follower_l"]
        self.assertIn("feedback_read_age_ms", result["missing_metrics"])
        self.assertIn("command_age_ms", result["missing_metrics"])
        self.assertNotIn("feedback_read_age_ms", result)

    def test_conversion_report_residuals_keep_relative_domain_visible(self):
        path = self.root / "camera_alignment.json"
        path.write_text(json.dumps({"schema_version": 1, "clock_domain": "relative_camera_start",
                                   "grid_timestamps_ns": [0, 10, 20],
                                   "streams": {"wrist": {
                                       "selected_source_indices": [0, 0, 1],
                                       "selected_timestamps_ns": [0, 0, 25_000_000],
                                       "residual_ns": [-2_000_000, 1_000_000, 5_000_000],
                                       "clock_mapping": {"mapping": "first_frame_relative"}}}}))
        result = audit.audit_alignment(path)
        self.assertEqual(result["clock_domain"], "relative_camera_start")
        self.assertEqual(result["streams"]["wrist"]["selected_minus_grid_ms"]["min"], -2)
        self.assertEqual(result["streams"]["wrist"]["absolute_residual_ms"]["max"], 5)
        self.assertEqual(result["streams"]["wrist"]["repeated_source_selections"], 1)
        self.assertNotIn("grid_timestamps_ns", result)

    def test_standard_conversion_report_and_db3_are_detected(self):
        (self.root / "cameras").mkdir()
        (self.root / "cameras" / "scene.db3").write_bytes(b"native recording")
        converted = self.root / "0000"
        converted.mkdir()
        alignment = converted / "camera_alignment.json"
        alignment.write_text(json.dumps({"streams": {}, "clock_domain": "host_wall_clock"}))

        def reader(path, **kwargs):
            self.assertEqual(path.suffix, ".db3")
            return {"samples": {"color": [sample(0, 1)], "depth": []},
                    "read": {"complete": True, "stop_reason": "eof"}}

        result = audit.audit_episode(self.root, bag_reader=reader)
        self.assertEqual(result["conversion_alignment"]["path"], str(alignment))
        self.assertEqual(result["reference_camera"], "scene")
        result = audit.audit_episode(self.root, alignment_report=self.root / "absent.json", bag_reader=reader)
        self.assertIn("missing", result["conversion_alignment"])

    def test_cli_can_print_json_without_importing_or_opening_cameras(self):
        np.savez(self.root / "robot_data.npz", timestamps=[1, 2],
                 follower_l_joint_pos=np.zeros((2, 6)))
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code = audit.main([str(self.root), "--json-out", "-"])
        self.assertEqual(code, 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["cameras"], {})
        self.assertEqual(result["robot"]["row_count"], 2)
        target = self.root / "existing.json"
        target.write_text("retained")
        with self.assertRaises(FileExistsError):
            audit.main([str(self.root), "--json-out", str(target)])
        self.assertEqual(target.read_text(), "retained")

    def test_empty_directory_is_rejected_as_raw_episode(self):
        with self.assertRaisesRegex(ValueError, "expected a saved raw episode"):
            audit.audit_episode(self.root)
        with contextlib.redirect_stderr(io.StringIO()) as error_output:
            with self.assertRaises(SystemExit) as error:
                audit.main([str(self.root)])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("expected a saved raw episode", error_output.getvalue())


if __name__ == "__main__":
    unittest.main()
