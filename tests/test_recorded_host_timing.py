"""Measured API sends are a distinct input experiment, without fitted lag."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from raiden.sim.command_replay import JOINT_NAMES, schedule, with_recorded_host_timing


class RecordedHostTimingTest(unittest.TestCase):
    def test_measured_send_holds_preserve_targets_start_and_actual_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, rollout = root / "native", root / "rollout"
            bundle.mkdir()
            rollout.mkdir()
            commands = np.zeros((6, 7))
            commands[:, 6] = 1
            commands[3:, 0] = 0.1
            q, dq = np.r_[np.arange(6) * 0.01, 0.8], np.r_[np.zeros(6), -0.02]
            np.savez(
                bundle / "commands.npz",
                cmd=commands,
                requested_cmd=commands,
                offset_ns=np.arange(6, dtype=np.int64) * 2_000_000,
                initial_q=q,
                initial_dq=dq,
            )
            metadata = {
                "schema": "raiden.held_joint_commands.v1",
                "joint_names": JOINT_NAMES,
                "source_kind": "isaac_applied_tick_targets",
                "duration_ns": 12_000_000,
                "physics_dt": 0.002,
                "initial_state_override": {"kind": "synthetic measured start"},
                "initial_state": {"declared": "retained portable physical state"},
            }
            (bundle / "commands.json").write_text(json.dumps(metadata))
            events = [
                {
                    "requested_cmd": commands[0].tolist(),
                    "send_before_monotonic_ns": 100_000_009,
                    "send_after_monotonic_ns": 100_000_029,
                },
                {
                    "requested_cmd": commands[3].tolist(),
                    "send_before_monotonic_ns": 107_000_010,
                    "send_after_monotonic_ns": 107_000_040,
                },
            ]
            execution = {
                "status": "complete",
                "events_sent": 2,
                "stream_start_monotonic_ns": 100_000_000,
                "stream_end_monotonic_ns": 114_000_000,
            }
            (rollout / "execution.json").write_text(json.dumps(execution))
            trace = rollout / "send_observation_trace.json"
            trace.write_text(json.dumps(events))
            arrays, saved = with_recorded_host_timing(bundle, rollout, root / "host")
            np.testing.assert_array_equal(arrays["offset_ns"], [0, 7_000_001])
            np.testing.assert_array_equal(arrays["initial_q"], q)
            np.testing.assert_array_equal(arrays["initial_dq"], dq)
            self.assertEqual(saved["initial_state"], metadata["initial_state"])
            self.assertEqual(
                saved["recorded_host_timing"]["origin_after_stream_start_ns"], 9
            )
            self.assertEqual(saved["source_kind"], "hardware_host_request_events")
            indices, _ = schedule(arrays, saved, 0.002)
            np.testing.assert_array_equal(indices, [0, 0, 0, 0, 1, 1, 1])
            # A changed target cannot be excused as a timing-only derivative.
            events[1]["requested_cmd"][0] = 0.2
            trace.write_text(json.dumps(events))
            with self.assertRaisesRegex(
                ValueError, "differ from the native held targets"
            ):
                with_recorded_host_timing(bundle, rollout, root / "bad")


if __name__ == "__main__":
    unittest.main()
