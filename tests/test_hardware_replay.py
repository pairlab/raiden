"""Native hold integrity and motion guards, using a virtual clock/follower only."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from raiden.sim.command_replay import (
    JOINT_NAMES,
    MIN_APERTURE,
    YAM_NOMINAL_LIMITS,
    sha256,
    with_measured_hardware_start,
)
from raiden.sim.hardware_replay import (
    dry_run_hardware_plan,
    execute_prepared_hardware,
    prepare_hardware_plan,
)

KP = [80.0, 80.0, 80.0, 40.0, 10.0, 10.0, 20.0]
KD = [5.0, 5.0, 5.0, 1.5, 1.5, 1.5, 0.5]


class VirtualClock:
    def __init__(self, oversleep_ns=0):
        self.now = 10_000_000_000
        self.sleeps = []
        self.oversleep_ns = oversleep_ns

    def monotonic_ns(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += round(seconds * 1e9) + self.oversleep_ns


class FakeFollower:
    def __init__(self):
        self.q = np.r_[np.zeros(6), 1.0]
        self.sent = []
        self.info = {
            "kp": KP.copy(),
            "kd": KD.copy(),
            "gravity_comp_factor": 1.3,
            "limit_gripper_effort": 50.0,
            "joint_limits": YAM_NOMINAL_LIMITS.copy(),
        }

    def get_robot_info(self):
        return self.info

    def get_observations(self):
        return {
            "joint_pos": self.q[:6].copy(),
            "joint_vel": np.zeros(6),
            "joint_eff": np.zeros(6),
            "gripper_pos": self.q[6:].copy(),
            "gripper_vel": np.zeros(1),
            "gripper_eff": np.zeros(1),
        }

    def command_joint_pos(self, command):
        self.sent.append(command.copy())
        self.q = command.copy()


class TimedFollower(FakeFollower):
    def __init__(self, clock, send_duration_ns):
        super().__init__()
        self.clock = clock
        self.send_duration_ns = send_duration_ns

    def command_joint_pos(self, command):
        super().command_joint_pos(command)
        self.clock.now += self.send_duration_ns


class HardwarePreparationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        self.episode = self.root / "episode"
        self.episode.mkdir()
        self.plan = self.root / "plan"
        self.cmd = np.zeros((4, 7))
        self.cmd[:, 6] = [1.0, 1.0, MIN_APERTURE, MIN_APERTURE]
        self.cmd[2:, 0] = 0.1
        np.savez(
            self.episode / "robot_data.npz",
            timestamps=np.arange(4, dtype=np.int64) * 10_000_000,
            follower_l_joint_cmd=np.c_[np.zeros((4, 6)), [1.0, 1.0, 0.0, 0.0]],
        )
        (self.episode / "metadata.json").write_text("{}")
        self.metadata = {
            "schema": "raiden.held_joint_commands.v1",
            "joint_names": JOINT_NAMES,
            "source_kind": "isaac_applied_tick_targets",
            "source_episode": str(self.episode),
            "duration_ns": 8_000_000,
            "physics_dt": 0.002,
            "initial_state": {"kp": KP, "kd": KD, "scene": {"test": "physical start"}},
            "source_sim_metadata": {
                "controller": {"gravity_comp_factor": 1.3},
                "scene": {"initial_state": {"initial_state_id": "parity_s2"}},
            },
            "source_sha256": {
                name: sha256(self.episode / name)
                for name in ("robot_data.npz", "metadata.json")
            },
        }
        self.arrays = {
            "offset_ns": np.arange(4, dtype=np.int64) * 2_000_000,
            "cmd": self.cmd,
            "requested_cmd": self.cmd.copy(),
            "initial_q": np.r_[np.zeros(6), 1.0],
            "initial_dq": np.zeros(7),
        }
        self.write_bundle()

    def tearDown(self):
        self.temp.cleanup()

    def write_bundle(self):
        np.savez(self.bundle / "commands.npz", **self.arrays)
        (self.bundle / "commands.json").write_text(json.dumps(self.metadata))

    def prepare(self):
        return prepare_hardware_plan(self.bundle, self.plan)

    def run_ready(self, follower, **kwargs):
        return execute_prepared_hardware(
            self.plan,
            follower,
            self.root / "executed",
            execute=True,
            attended=True,
            matched_physical_start=True,
            exclusive_control=True,
            controller_history_ready=True,
            **kwargs,
        )

    def test_compacts_only_identical_holds_and_preserves_sampled_intent(self):
        arrays, plan = self.prepare()
        np.testing.assert_array_equal(arrays["offset_ns"], [0, 4_000_000])
        np.testing.assert_array_equal(arrays["source_tick_index"], [0, 2])
        np.testing.assert_array_equal(arrays["sampled_operator_intent"], [1, 1, 0, 0])
        self.assertEqual(plan["duration_ns"], 8_000_000)
        self.assertTrue(dry_run_hardware_plan(self.plan)["software_stream_valid"])
        self.assertFalse(plan["hardware_executed"])

    def test_constant_native_stream_retains_full_final_hold(self):
        self.arrays["cmd"][:] = self.arrays["cmd"][0]
        self.write_bundle()
        arrays, _ = self.prepare()
        self.assertEqual(len(arrays["cmd"]), 1)
        self.assertEqual(dry_run_hardware_plan(self.plan)["duration_ns"], 8_000_000)

    def test_source_change_and_plan_start_or_controller_tampering_fail(self):
        self.prepare()
        path = self.plan / "hardware_plan.json"
        plan = json.loads(path.read_text())
        plan["expected_controller"]["kp"][0] = 0
        path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "controller differs"):
            dry_run_hardware_plan(self.plan)
        plan["expected_controller"]["kp"][0] = KP[0]
        path.write_text(json.dumps(plan))
        arrays_path = self.plan / "hardware_plan.npz"
        with np.load(arrays_path) as original:
            arrays = {key: original[key].copy() for key in original.files}
        arrays["initial_q"][0] = 0.5
        np.savez(arrays_path, **arrays)
        plan["plan_arrays_sha256"] = sha256(arrays_path)
        path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "initial_q"):
            dry_run_hardware_plan(self.plan)
        (self.episode / "metadata.json").write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, "source content changed"):
            dry_run_hardware_plan(self.plan)

    def test_explicit_readiness_gain_state_and_finite_limit_guards_precede_commands(
        self,
    ):
        self.prepare()
        follower = FakeFollower()
        with self.assertRaisesRegex(ValueError, "must all be explicit"):
            execute_prepared_hardware(self.plan, follower, self.root / "none")
        follower.info["kp"][0] = 0
        with self.assertRaisesRegex(ValueError, "setting differs"):
            self.run_ready(follower)
        follower.info["kp"][0] = KP[0]
        follower.q[0] = 0.2
        with self.assertRaisesRegex(ValueError, "start differs"):
            self.run_ready(follower)
        follower.q[0] = 0
        follower.info["joint_limits"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "joint limits"):
            self.run_ready(follower)
        self.assertEqual(follower.sent, [])

    def test_success_streams_exact_native_targets_then_observed_pose_hold_without_home(
        self,
    ):
        self.prepare()
        follower, clock = FakeFollower(), VirtualClock()
        result = self.run_ready(follower, clock=clock)
        np.testing.assert_array_equal(
            follower.sent, [self.cmd[0], self.cmd[2], self.cmd[2]]
        )
        trace = json.loads(
            (self.root / "executed/send_observation_trace.json").read_text()
        )
        self.assertEqual([r["intended_offset_ns"] for r in trace], [0, 4_000_000])
        self.assertEqual([r["lateness_ns"] for r in trace], [0, 0])
        self.assertEqual(clock.now, 10_008_000_000)
        self.assertEqual(
            result["final_observed_pose_hold"]["purpose"], "completion hold"
        )
        self.assertTrue(result["hardware_executed"])

    def test_deadline_miss_aborts_without_burst_and_logs_abort_hold(self):
        self.prepare()
        follower, clock = FakeFollower(), VirtualClock(oversleep_ns=3_000_000)
        with self.assertRaisesRegex(RuntimeError, "deadline missed"):
            self.run_ready(follower, clock=clock)
        self.assertEqual(len(follower.sent), 2)  # first event plus same-pose abort hold
        result = json.loads((self.root / "executed/execution.json").read_text())
        self.assertEqual(result["events_sent"], 1)
        self.assertEqual(result["final_observed_pose_hold"]["purpose"], "abort hold")
        self.assertEqual(result["deadline_miss"]["event"], 1)
        self.assertEqual(result["deadline_miss"]["lateness_ns"], 3_000_000)
        self.assertTrue(result["hardware_executed"])

    def test_stop_remains_responsive_during_final_long_hold(self):
        self.arrays["cmd"][:] = self.arrays["cmd"][0]
        self.metadata["duration_ns"] = 1_000_000_000
        self.write_bundle()
        self.prepare()
        follower, clock = FakeFollower(), VirtualClock()
        with self.assertRaises(InterruptedError):
            self.run_ready(
                follower,
                clock=clock,
                stop_requested=lambda: clock.now >= 10_025_000_000,
            )
        self.assertLessEqual(clock.now, 10_035_000_000)
        self.assertLessEqual(max(clock.sleeps), 0.010)
        self.assertEqual(len(follower.sent), 2)

    def test_best_effort_preserves_every_hold_and_records_cumulative_dilation(self):
        # Adjacent 2 ms requests plus a 1 ms API operation and 3 ms wake jitter:
        # an absolute scheduler would burst/catch up. Best effort must not.
        self.arrays["cmd"][:, 0] = [0.0, 0.1, 0.2, 0.3]
        self.write_bundle()
        self.prepare()
        clock = VirtualClock(oversleep_ns=3_000_000)
        follower = TimedFollower(clock, send_duration_ns=1_000_000)
        result = self.run_ready(follower, clock=clock, timing_profile="best_effort")
        trace = json.loads(
            (self.root / "executed/send_observation_trace.json").read_text()
        )
        np.testing.assert_array_equal(follower.sent[:4], self.arrays["cmd"])
        self.assertEqual(
            [r["intended_offset_ns"] for r in trace],
            [0, 2_000_000, 4_000_000, 6_000_000],
        )
        self.assertEqual([r["source_hold_ns"] for r in trace], [2_000_000] * 4)
        self.assertEqual(
            [r["observed_hold_after_send_completion_ns"] for r in trace],
            [5_000_000] * 4,
        )
        self.assertEqual(
            [r["schedule_delay_after_ns"] for r in trace],
            [1_000_000, 5_000_000, 9_000_000, 13_000_000],
        )
        self.assertTrue(all(r["full_hold_observed"] for r in trace))
        self.assertEqual(result["max_lateness_ns"], 50_000_000)
        self.assertEqual(result["stream_wall_duration_ns"], 24_000_000)
        self.assertEqual(result["completed_timing_dilation_ns"], 16_000_000)
        self.assertEqual(result["final_hold_end_lateness_ns"], 3_000_000)
        self.assertEqual(result["request_hold_audit"]["shorter_than_source_holds"], 0)
        self.assertFalse(result["exact_native_timing_claimed"])
        self.assertEqual(
            result["final_observed_pose_hold"]["send_before_monotonic_ns"],
            10_024_000_000,
        )

    def test_best_effort_accepts_50ms_wake_lateness_boundary(self):
        self.prepare()
        follower, clock = FakeFollower(), VirtualClock(oversleep_ns=50_000_000)
        result = self.run_ready(follower, clock=clock, timing_profile="best_effort")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["successful_send_lateness_ns"]["max"], 50_000_000)
        self.assertEqual(result["request_hold_audit"]["shorter_than_source_holds"], 0)

    def test_unbounded_preserves_holds_through_long_wakes_and_sends(self):
        self.arrays["cmd"][:, 0] = [0.0, 0.1, 0.2, 0.3]
        self.write_bundle()
        self.prepare()
        clock = VirtualClock(oversleep_ns=75_000_000)
        follower = TimedFollower(clock, send_duration_ns=100_000_000)
        result = self.run_ready(
            follower, clock=clock, timing_profile="best_effort_unbounded"
        )
        np.testing.assert_array_equal(follower.sent[:4], self.arrays["cmd"])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["events_sent"], 4)
        self.assertIsNone(result["max_lateness_ns"])
        self.assertEqual(result["successful_send_lateness_ns"]["max"], 75_000_000)
        self.assertEqual(result["request_hold_audit"]["shorter_than_source_holds"], 0)
        self.assertGreater(result["completed_timing_dilation_ns"], 500_000_000)

    def test_best_effort_long_host_stall_aborts_without_sending_overdue_event(self):
        self.prepare()
        follower, clock = FakeFollower(), VirtualClock(oversleep_ns=51_000_000)
        with self.assertRaisesRegex(RuntimeError, "host stall"):
            self.run_ready(follower, clock=clock, timing_profile="best_effort")
        result = json.loads((self.root / "executed/execution.json").read_text())
        self.assertEqual(result["events_sent"], 1)
        self.assertEqual(
            len(follower.sent), 2
        )  # first native target, then same-pose abort hold
        self.assertEqual(result["host_stall"]["unexpected_duration_ns"], 51_000_000)
        self.assertIsNone(result["completed_timing_dilation_ns"])
        self.assertEqual(result["final_observed_pose_hold"]["purpose"], "abort hold")

    def test_best_effort_blocked_send_is_not_hidden_by_moving_next_deadline(self):
        self.prepare()
        clock = VirtualClock()
        follower = TimedFollower(clock, send_duration_ns=51_000_000)
        with self.assertRaisesRegex(RuntimeError, "command_joint_pos"):
            self.run_ready(follower, clock=clock, timing_profile="best_effort")
        result = json.loads((self.root / "executed/execution.json").read_text())
        self.assertEqual(result["events_sent"], 1)
        self.assertEqual(result["host_stall"]["operation"], "command_joint_pos")
        self.assertTrue(result["hardware_executed"])

    def test_best_effort_hold_stop_remains_bounded_during_final_hold(self):
        self.arrays["cmd"][:] = self.arrays["cmd"][0]
        self.metadata["duration_ns"] = 1_000_000_000
        self.write_bundle()
        self.prepare()
        follower, clock = FakeFollower(), VirtualClock()
        with self.assertRaises(InterruptedError):
            self.run_ready(
                follower,
                clock=clock,
                timing_profile="best_effort",
                stop_requested=lambda: clock.now >= 10_025_000_000,
            )
        self.assertLessEqual(clock.now, 10_035_000_000)
        self.assertEqual(len(follower.sent), 2)
        result = json.loads((self.root / "executed/execution.json").read_text())
        self.assertEqual(result["final_observed_pose_hold"]["purpose"], "abort hold")

    def test_invalid_timing_profile_is_rejected_before_motion(self):
        self.prepare()
        follower = FakeFollower()
        for profile in ("", "best-effort", None, []):
            with self.assertRaisesRegex(ValueError, "timing_profile"):
                self.run_ready(follower, timing_profile=profile)
        self.assertEqual(follower.sent, [])

    def test_measured_shared_start_keeps_inputs_and_nonrobot_poses_unchanged(self):
        names = JOINT_NAMES + [
            "gripper0_right_left_finger",
            "gripper0_right_right_finger",
        ]
        initial = self.metadata["initial_state"]
        initial.update(
            joint_names={"robot": names},
            observations={
                "joint_pos": [0.0] * 6,
                "joint_vel": [0.0] * 6,
                "gripper_pos": [1.0],
                "gripper_vel": [0.0],
            },
            scene={
                "articulation": {
                    "robot": {
                        "joint_position": [[0.0] * 8],
                        "joint_velocity": [[0.0] * 8],
                    },
                    "oven": {"root_pose": [[0.775, 0.06, 0.9, 0, 0, 0, 1]]},
                },
                "rigid_object": {
                    "croissant": {"root_pose": [[0.52, 0.175, 0.9215, 0, 0, 0, 1]]}
                },
            },
        )
        self.metadata["source_sim_metadata"]["finger_stops_m"] = [0.0, 0.0475]
        self.write_bundle()
        original_hashes = {
            name: sha256(self.bundle / name)
            for name in ("commands.npz", "commands.json")
        }
        q = [0.008, 0.001, 0.006, -0.071, 0.006, -0.014, 0.999]
        snapshot = self.root / "measured.json"
        snapshot.write_text(
            json.dumps(
                {
                    "health": {"healthy": True},
                    "controller_matches": True,
                    "initial_q_rad_aperture": q,
                    "initial_dq_rad_s_aperture_s": [0.0] * 7,
                    "host_time_ns": 123,
                    "host_monotonic_ns": 456,
                }
            )
        )
        arrays, metadata = with_measured_hardware_start(
            self.bundle, snapshot, self.root / "shared"
        )
        for key in ("offset_ns", "cmd", "requested_cmd"):
            np.testing.assert_array_equal(arrays[key], self.arrays[key])
        np.testing.assert_array_equal(arrays["initial_q"], q)
        physical = metadata["initial_state"]["scene"]
        self.assertEqual(
            physical["articulation"]["robot"]["joint_position"][0][3], -0.071
        )
        self.assertAlmostEqual(
            physical["articulation"]["robot"]["joint_position"][0][6], 0.0475 * 0.999
        )
        self.assertEqual(physical["rigid_object"], initial["scene"]["rigid_object"])
        self.assertEqual(
            physical["articulation"]["oven"], initial["scene"]["articulation"]["oven"]
        )
        self.assertEqual(
            {name: sha256(self.bundle / name) for name in original_hashes},
            original_hashes,
        )
        self.assertIn("live measured", metadata["initial_state_origin"])

    def test_invalid_tolerances_do_not_issue_commands(self):
        self.prepare()
        follower = FakeFollower()
        for kwargs in [
            {"arm_start_tolerance_rad": np.nan},
            {"aperture_start_tolerance": -1},
            {"max_lateness_ns": float("nan")},
            {"max_lateness_ns": -1},
        ]:
            with self.assertRaises(ValueError):
                self.run_ready(follower, **kwargs)
        self.assertEqual(follower.sent, [])


if __name__ == "__main__":
    unittest.main()
