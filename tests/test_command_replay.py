"""Timing/convention tests; these run without simulator or robot imports."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from raiden.sim.command_replay import (
    JOINT_NAMES,
    measured_initial_state,
    schedule,
    validate_bundle,
)


class HeldReplayTest(unittest.TestCase):
    def bundle(self, offsets):
        arrays = {
            "offset_ns": np.array(offsets, dtype=np.int64),
            "cmd": np.zeros((len(offsets), 7)),
            "initial_q": np.zeros(7),
            "initial_dq": np.zeros(7),
        }
        metadata = {
            "schema": "raiden.held_joint_commands.v1",
            "joint_names": JOINT_NAMES,
            "source_kind": "real_observation_clock_command_snapshots",
            "duration_ns": offsets[-1],
        }
        return arrays, metadata

    def test_native_ticks_preserve_targets_exactly(self):
        arrays, metadata = self.bundle([0, 2_000_000, 4_000_000, 6_000_000])
        metadata.update(
            source_kind="isaac_applied_tick_targets",
            physics_dt=0.002,
            duration_ns=8_000_000,
        )
        arrays["cmd"][:, 0] = np.arange(4)
        indices, report = schedule(arrays, metadata, 0.002)
        np.testing.assert_array_equal(indices, np.arange(4))
        self.assertEqual(report["max_event_round_forward_ns"], 0)
        with self.assertRaises(ValueError):
            schedule(arrays, metadata, 0.004)

    def test_irregular_snapshots_round_forward_and_hold(self):
        arrays, metadata = self.bundle([0, 3_000_001, 10_000_001])
        arrays["cmd"][:, 0] = [1, 2, 3]
        indices, report = schedule(arrays, metadata, 0.002)
        np.testing.assert_array_equal(indices, [0, 0, 1, 1, 1, 1])
        self.assertEqual(report["max_event_round_forward_ns"], 999_999)
        self.assertEqual(report["duration_overrun_ns"], 1_999_999)
        self.assertEqual(report["last_boundary_events_without_hold"], 1)

    def test_nonmonotonic_and_invalid_aperture_rejected(self):
        arrays, metadata = self.bundle([0, 5, 5])
        with self.assertRaises(ValueError):
            validate_bundle(arrays, metadata)
        arrays["offset_ns"][-1] = 6
        metadata["duration_ns"] = 6
        arrays["cmd"][0, 6] = 2
        with self.assertRaises(ValueError):
            validate_bundle(arrays, metadata)

    def test_measured_initial_q_is_not_first_requested_q(self):
        arrays, metadata = self.bundle([0, 10])
        arrays["initial_q"] = np.r_[np.arange(6) * 0.1, 0.8]
        arrays["initial_dq"] = np.r_[np.ones(6) * 0.2, 0.3]
        arrays["requested_cmd"] = arrays["cmd"].copy()
        names = JOINT_NAMES + [
            "gripper0_right_left_finger",
            "gripper0_right_right_finger",
        ]
        template = {
            "joint_names": {"robot": names},
            "scene": {
                "articulation": {
                    "robot": {
                        "joint_position": [[0.0] * 8],
                        "joint_velocity": [[0.0] * 8],
                    }
                }
            },
        }
        state = measured_initial_state(
            template, arrays, metadata, {"finger_stops_m": [0.0, 0.0475]}
        )
        q = state["scene"]["articulation"]["robot"]["joint_position"][0]
        np.testing.assert_allclose(q, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.038, 0.038])
        self.assertEqual(
            template["scene"]["articulation"]["robot"]["joint_position"][0][-1], 0.0
        )

    def test_failed_rollout_saves_partial_response_and_restores_state_and_pause(self):
        spec = importlib.util.spec_from_file_location(
            "parity_replay", Path(__file__).parents[1] / "scripts/parity_replay.py"
        )
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        arrays, metadata = self.bundle([0, 2_000_000, 4_000_000])
        live = {
            "backend": "isaac",
            "physics_dt": 0.002,
            "joint_limits_rad": [[-4, 4]] * 6,
        }
        metadata.update(
            source_kind="isaac_applied_tick_targets",
            physics_dt=0.002,
            duration_ns=6_000_000,
            initial_state={"initial": True},
            source_sim_metadata=live,
            timing_uncertainty="test",
        )

        class FailingConnection:
            def __init__(self):
                self.calls = []
                self.steps = 0

            def call(self, name, **kwargs):
                self.calls.append((name, kwargs))
                if name == "get_metadata":
                    return live
                if name == "is_paused":
                    return False
                if name == "get_state":
                    return {"original": True}
                if name == "step":
                    self.steps += 1
                    if self.steps == 2:
                        raise RuntimeError("synthetic solver failure")
                    return {
                        "state": {
                            "observations": {
                                "joint_pos": [0.0] * 6,
                                "joint_vel": [0.0] * 6,
                                "joint_eff": [0.0] * 6,
                                "gripper_pos": [1.0],
                                "gripper_eff": [0.0],
                            },
                            "motor": {
                                "computed_effort_Nm": [0.0] * 6,
                                "applied_effort_Nm": [0.0] * 6,
                            },
                            "scene": {
                                "articulation": {"oven": {"joint_position": [[0.0]]}},
                                "rigid_object": {
                                    "croissant": {
                                        "root_pose": [
                                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                                        ]
                                    }
                                },
                            },
                        }
                    }

            def close(self):
                self.calls.append(("close", {}))

        connection = FailingConnection()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "trace"
            args = SimpleNamespace(
                out=out,
                sim="unused",
                bundle=Path(directory),
                repeat=1,
                allow_scene_change=False,
                allow_unmeasured_start=False,
            )
            with patch("raiden.sim.client.SimConnection", return_value=connection):
                with self.assertRaisesRegex(RuntimeError, "solver failure"):
                    script.replay(args, arrays, metadata)
            with np.load(out / "run_00.npz") as saved:
                self.assertEqual(saved["observation"].shape, (1, 7))
            self.assertEqual(
                json.loads((out / "replay.json").read_text())["status"], "failed"
            )
            self.assertEqual(
                connection.calls[-3:],
                [
                    ("reset_to", {"state": {"original": True}}),
                    ("set_paused", {"paused": False}),
                    ("close", {}),
                ],
            )


if __name__ == "__main__":
    unittest.main()
