"""Offline source selection, native evidence retention and capture failure guards."""

import gzip
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import termios
import threading
import tty
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from raiden.sim.command_replay import YAM_NOMINAL_LIMITS
from raiden.sim.hardware_session import NativeControllerRecorder
from raiden.sim.real_replay import (
    NativeTraceWriter,
    controller_processes,
    prepare_selection,
    run_episode,
    select_episodes,
    validate_prepared,
)
from raiden.sim.replay_capture import CAMERAS, DualCameraCapture

KP, KD = [80.0, 80.0, 80.0, 40.0, 10.0, 10.0, 20.0], [5.0, 5.0, 5.0, 1.5, 1.5, 1.5, 0.5]


class Clock:
    def __init__(self):
        self.now = 10_000_000_000

    def monotonic_ns(self):
        return self.now

    def time_ns(self):
        return self.now + 1_000_000_000_000

    def sleep(self, seconds):
        self.now += round(seconds * 1e9)


class Follower:
    def __init__(self, clock=None):
        self.robot = self
        self.clock = clock
        self.q = np.r_[np.zeros(6), 1.0]
        self.sent = []

    def get_robot_info(self):
        return {
            "kp": KP,
            "kd": KD,
            "gravity_comp_factor": 1.3,
            "limit_gripper_effort": 50.0,
            "joint_limits": YAM_NOMINAL_LIMITS,
        }

    def get_observations(self):
        return {
            "joint_pos": self.q[:6].copy(),
            "joint_vel": np.zeros(6),
            "joint_eff": np.arange(6),
            "gripper_pos": self.q[6:].copy(),
            "gripper_vel": np.zeros(1),
            "gripper_eff": np.ones(1),
        }

    def command_joint_pos(self, q):
        self.sent.append(q.copy())
        self.q = q.copy()
        if self.clock:
            self.clock.now += 1234


class Recorder(NativeControllerRecorder):
    def health(self, *_):
        return {"healthy": True}


class FakeTrace:
    def __init__(self, *args):
        self.closed = False

    def start(self):
        pass

    def check(self):
        pass

    def close(self):
        self.closed = True


class FakeCapture:
    fail_start = False

    def __init__(self, *args):
        self.stopped = False

    def start(self, **kwargs):
        if self.fail_start:
            raise RuntimeError("wrist camera failed")

    def check(self):
        pass

    def snapshot(self):
        return {name: {"status": "recording", "frames": 1} for name in CAMERAS}

    def stop(self):
        self.stopped = True


class RealReplayTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "raw"
        self.episode = self.source / "0005"
        self.episode.mkdir(parents=True)
        self.config = {
            name: {
                "serial": str(i),
                "type": "realsense",
                "color_controls": {"white_balance": 2800, "exposure": 250, "gain": 22},
            }
            for i, name in enumerate(CAMERAS)
        }
        self.config_path = self.root / "camera.json"
        self.config_path.write_text(json.dumps(self.config))
        cmd = np.c_[np.zeros((4, 6)), [1.0, 1.0, 0.5, 0.5]]
        np.savez(
            self.episode / "robot_data.npz",
            timestamps=np.arange(4, dtype=np.int64) * 10_000_000,
            follower_l_joint_cmd=cmd,
            follower_l_joint_pos_7d=np.tile(cmd[0], (4, 1)),
            follower_l_joint_vel=np.zeros((4, 6)),
            follower_l_gripper_vel=np.zeros((4, 1)),
        )
        np.savez(
            self.episode / "sim_action_log.npz",
            sim_time=np.arange(1, 5) * 0.002,
            cmd=cmd,
            requested_cmd=cmd,
        )
        (self.episode / "metadata.json").write_text(
            json.dumps({"complete": True, "task_name": "full_oven"})
        )
        initial = {
            "physics_dt": 0.002,
            "sim_time": 0.0,
            "kp": KP,
            "kd": KD,
            "observations": {
                "joint_pos": [0.0] * 6,
                "gripper_pos": [1.0],
                "joint_vel": [0.0] * 6,
                "gripper_vel": [0.0],
            },
            "scene": {"preserved": "objects"},
        }
        (self.episode / "sim_initial_state.json").write_text(json.dumps(initial))
        sim = {
            "scene": {"initial_state": {"initial_state_id": "parity_s2"}},
            "controller": {"gravity_comp_factor": 1.3},
            "calibration": {"source": "accepted 20261018_175242"},
            "calibration_sha256": "test",
        }
        (self.episode / "sim_metadata.json").write_text(json.dumps(sim))
        (self.episode / "calibration_results.json").write_text(
            json.dumps({"source": "transformed accepted 20261018_175242"})
        )
        self.prepared = self.root / "prepared"

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self):
        return prepare_selection(
            self.source, ["0005"], self.prepared, camera_config=self.config_path
        )

    def test_explicit_selection_rejects_missing_duplicate_incomplete_and_old_state(
        self,
    ):
        for ids in ([], ["0005", "0005"], ["0002"], ["../0005"]):
            with self.assertRaises(ValueError):
                select_episodes(self.source, ids)
        (self.episode / "metadata.json").write_text('{"complete":false}')
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.prepare()
        self.assertFalse(self.prepared.exists())
        (self.episode / "metadata.json").write_text('{"complete":true}')
        meta = json.loads((self.episode / "sim_metadata.json").read_text())
        meta["scene"]["initial_state"]["initial_state_id"] = "old_drop_start"
        (self.episode / "sim_metadata.json").write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "source start"):
            self.prepare()

    def test_prepare_preserves_native_holds_and_rejects_changed_bindings(self):
        manifest = self.prepare()
        self.assertEqual(manifest["episodes"][0]["request_events"], 2)
        self.assertEqual(manifest["episodes"][0]["duration_ns"], 8_000_000)
        with np.load(self.prepared / "0005/plan/hardware_plan.npz") as plan:
            np.testing.assert_array_equal(plan["offset_ns"], [0, 4_000_000])
        self.assertEqual(validate_prepared(self.prepared)["status"], "prepared")
        self.config_path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            validate_prepared(self.prepared)

    def test_offline_cli_preparation_cannot_import_devices(self):
        script = Path(__file__).parents[1] / "scripts/replay_sim_in_real.py"
        bootstrap = """import builtins,runpy,sys
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in {'i2rt','pyrealsense2','cv2'}:
        raise AssertionError('offline CLI imported device module '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
sys.argv=sys.argv[1:]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                bootstrap,
                str(script),
                "prepare",
                "--source-root",
                str(self.source),
                "--episodes",
                "0005",
                "--out",
                str(self.prepared),
                "--camera-config",
                str(self.config_path),
            ],
            cwd=script.parents[1],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout)["hardware_initialized"])

    def test_partial_capture_failure_prevents_any_task_or_hold_command(self):
        self.prepare()
        follower, capture, trace = Follower(), FakeCapture(), FakeTrace()
        capture.fail_start = True
        with self.assertRaisesRegex(RuntimeError, "wrist camera failed"):
            run_episode(
                self.prepared / "0005/plan",
                follower,
                Recorder(),
                self.root / "trial",
                self.config,
                camera_factory=lambda *_: capture,
                trace_factory=lambda *_: trace,
                enforce_initial_state=False,
            )
        self.assertFalse(follower.sent)
        self.assertTrue(capture.stopped)
        self.assertTrue(trace.closed)
        self.assertEqual(
            json.loads((self.root / "trial/trial.json").read_text())["status"], "failed"
        )

    def test_camera_loss_during_stream_aborts_remaining_events_and_retains_hold(self):
        self.prepare()
        follower, clock, capture = Follower(), Clock(), FakeCapture()
        follower.clock = clock
        checks = []

        def camera_check():
            checks.append(True)
            if len(checks) >= 3:
                raise RuntimeError("scene camera became stale")

        capture.check = camera_check
        with self.assertRaisesRegex(RuntimeError, "camera became stale"):
            run_episode(
                self.prepared / "0005/plan",
                follower,
                Recorder(),
                self.root / "loss",
                self.config,
                camera_factory=lambda *_: capture,
                trace_factory=FakeTrace,
                clock=clock,
            )
        execution = json.loads((self.root / "loss/rollout/execution.json").read_text())
        self.assertEqual(execution["events_sent"], 1)
        self.assertEqual(execution["status"], "failed")
        self.assertIn("final_observed_pose_hold", execution)
        self.assertEqual(len(follower.sent), 2)
        self.assertTrue(capture.stopped)

    def test_start_mismatch_refuses_task_and_effective_tolerance_is_recorded(self):
        self.prepare()
        follower = Follower()
        follower.q[-1] = 0.9
        with self.assertRaisesRegex(ValueError, "hardware start differs"):
            run_episode(
                self.prepared / "0005/plan",
                follower,
                Recorder(),
                self.root / "strict",
                self.config,
                camera_factory=FakeCapture,
                trace_factory=FakeTrace,
            )
        self.assertFalse(follower.sent)
        clock = Clock()
        follower.clock = clock
        result = run_episode(
            self.prepared / "0005/plan",
            follower,
            Recorder(),
            self.root / "tolerant",
            self.config,
            aperture_start_tolerance=0.12,
            camera_factory=FakeCapture,
            trace_factory=FakeTrace,
            clock=clock,
        )
        snap = json.loads((self.root / "tolerant/measured_start.json").read_text())
        self.assertFalse(snap["default_tolerance_start_ready"])
        self.assertTrue(snap["numerical_start_ready"])
        self.assertAlmostEqual(snap["initial_error_rad_aperture"][-1], -0.1)
        self.assertEqual(result["execution"]["events_sent"], 2)
        self.assertEqual(
            result["execution"]["request_hold_audit"]["shorter_than_source_holds"], 0
        )
        self.assertEqual(
            len(follower.sent), 3
        )  # two original events, final observed pose hold
        self.assertFalse(result["device_application_timestamps_available"])

    def test_user_disabled_start_rejection_retains_measurements_and_controller_guards(
        self,
    ):
        self.prepare()
        clock, follower = Clock(), Follower()
        follower.clock = clock
        follower.q[:] = [0.4, 0.3, 0.2, -0.5, 0.2, 0.3, 0.6]
        original_obs = follower.get_observations

        def moving_observations():
            result = original_obs()
            result["joint_vel"] = np.full(6, 0.2)
            result["gripper_vel"] = np.array([0.1])
            return result

        follower.get_observations = moving_observations
        result = run_episode(
            self.prepared / "0005/plan",
            follower,
            Recorder(),
            self.root / "operator_reset",
            self.config,
            camera_factory=FakeCapture,
            trace_factory=FakeTrace,
            clock=clock,
            enforce_initial_state=False,
        )
        execution = result["execution"]
        self.assertFalse(execution["numerical_initial_state_check_enforced"])
        self.assertEqual(execution["events_sent"], 2)
        self.assertEqual(
            execution["measured_initial_q_rad_aperture"],
            [0.4, 0.3, 0.2, -0.5, 0.2, 0.3, 0.6],
        )
        self.assertEqual(
            execution["measured_initial_dq_rad_s_aperture_s"], [0.2] * 6 + [0.1]
        )
        np.testing.assert_array_equal(follower.sent[0], np.r_[np.zeros(6), 1.0])
        measured = json.loads(
            (self.root / "operator_reset/measured_start.json").read_text()
        )
        self.assertFalse(measured["numerical_start_ready"])
        self.assertFalse(measured["numerical_initial_state_check_enforced"])
        self.assertIn("user requested", result["initial_state_policy"])
        bad = Follower()
        original_info = bad.get_robot_info()
        original_info["kp"] = np.zeros(7)
        bad.get_robot_info = lambda: original_info
        with self.assertRaisesRegex(ValueError, "follower setting differs"):
            run_episode(
                self.prepared / "0005/plan",
                bad,
                Recorder(),
                self.root / "bad_gains",
                self.config,
                camera_factory=FakeCapture,
                trace_factory=FakeTrace,
                enforce_initial_state=False,
            )
        self.assertFalse(bad.sent)
        limits = Follower()
        info = limits.get_robot_info()
        info["joint_limits"] = np.tile([-0.001, 0.001], (6, 1))
        # The source first arm target is zero; move the lower bound above it.
        info["joint_limits"][0] = [0.1, 0.2]
        limits.get_robot_info = lambda: info
        with self.assertRaisesRegex(ValueError, "joint limits would clamp"):
            run_episode(
                self.prepared / "0005/plan",
                limits,
                Recorder(),
                self.root / "bad_limits",
                self.config,
                camera_factory=FakeCapture,
                trace_factory=FakeTrace,
                enforce_initial_state=False,
            )
        self.assertFalse(limits.sent)

    def test_process_audit_catches_native_and_module_owners_on_newer_pythons(self):
        proc = self.root / "proc"
        proc.mkdir()
        cases = {
            11: ["python3.13", "scripts/parity_native_hardware.py"],
            12: ["python3.12", "-m", "raiden", "record"],
            13: ["python", "scripts/replay_sim_in_real.py", "run", "--execute"],
            14: ["rd", "record", "--sim", "127.0.0.1:5599"],
            15: ["python", "scripts/calibration_hardware.py", "--channel", "other_can"],
        }
        for pid, argv in cases.items():
            folder = proc / str(pid)
            folder.mkdir()
            (folder / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in argv))
        self.assertEqual(
            {
                row["pid"]
                for row in controller_processes("can_follower_l", proc_root=proc)
            },
            {11, 12, 13},
        )

    def test_camera_import_failure_keeps_both_failed_receipts(self):
        capture = DualCameraCapture(self.config, self.root / "cameras")
        with patch.dict(sys.modules, {"pyrealsense2": None}):
            with self.assertRaisesRegex(RuntimeError, "camera startup failed"):
                capture.start(timeout=1.0)
            with self.assertRaisesRegex(RuntimeError, "capture incomplete"):
                capture.stop()
        for name in CAMERAS:
            receipt = json.loads(
                (self.root / "cameras" / name / "receipt.json").read_text()
            )
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(receipt["frames"], 0)
        aggregate = json.loads((self.root / "cameras/capture.json").read_text())
        self.assertFalse(aggregate["camera_settings_written"])

    def test_empty_or_stale_camera_fails_health_guard(self):
        clock = Clock()
        capture = DualCameraCapture(self.config, self.root / "cameras", clock=clock)
        with self.assertRaises(RuntimeError):
            capture.check()
        capture.status = {
            name: {
                "status": "recording",
                "frames": 1,
                "last_frame_monotonic_ns": clock.now,
            }
            for name in CAMERAS
        }
        capture.check()
        clock.now += 500_000_001
        with self.assertRaisesRegex(RuntimeError, "stale"):
            capture.check()

    def test_native_rows_persist_per_trial_and_overflow_is_explicit(self):
        recorder = Recorder(capacity=3)
        for trial in range(2):
            writer = NativeTraceWriter(
                recorder, self.root / f"trace{trial}", interval=10.0
            )
            writer.start()
            with recorder.lock:
                recorder.can_rows.append({"trial": trial, "motor_id": 1, "q": 1.0})
                recorder.controller_rows.append(
                    {"trial": trial, "dq": 0.0, "effort": 2.0}
                )
                recorder.total_can_rows += 1
                recorder.total_controller_rows += 1
            writer.close()
        with gzip.open(
            self.root / "trace0/native_can_transactions.jsonl.gz", "rt"
        ) as stream:
            self.assertEqual(json.loads(stream.readline())["trial"], 0)
        overflow = NativeTraceWriter(recorder, self.root / "overflow", interval=10.0)
        overflow.start()
        with recorder.lock:
            for i in range(4):
                recorder.can_rows.append({"motor_id": i})
                recorder.total_can_rows += 1
            recorder.controller_rows.append({"effort": 2.0})
            recorder.total_controller_rows += 1
        with self.assertRaisesRegex(RuntimeError, "overflowed"):
            overflow.close()
        receipt = json.loads((self.root / "overflow/instrumentation.json").read_text())
        self.assertEqual(receipt["status"], "failed")
        with gzip.open(
            self.root / "overflow/native_can_transactions.jsonl.gz", "rt"
        ) as stream:
            self.assertEqual(len(stream.readlines()), 3)

    def test_new_native_logging_error_cannot_be_reported_complete(self):
        recorder = Recorder()
        writer = NativeTraceWriter(recorder, self.root / "trace", interval=10.0)
        writer.start()
        with recorder.lock:
            recorder.can_rows.append({"q": 0.0})
            recorder.controller_rows.append({"q": 0.0})
            recorder.total_can_rows += 1
            recorder.total_controller_rows += 1
        recorder.logging_error_count += 1
        with self.assertRaisesRegex(RuntimeError, "new logging failure"):
            writer.close()
        self.assertEqual(
            json.loads((self.root / "trace/instrumentation.json").read_text())[
                "status"
            ],
            "failed",
        )


class OwnerFailureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parents[1] / "scripts/replay_sim_in_real.py"
        spec = importlib.util.spec_from_file_location("replay_sim_in_real_test", path)
        cls.script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.script)

    def test_inherited_raw_terminal_accepts_echoed_enter_and_restores_original_state(
        self,
    ):
        master, slave = os.openpty()
        stream = os.fdopen(os.dup(slave), "r")
        try:
            tty.setraw(slave)
            raw = termios.tcgetattr(slave)
            with patch.object(self.script.sys, "stdin", stream):
                with self.script.attended_terminal():
                    current = termios.tcgetattr(slave)
                    self.assertEqual(
                        current[3] & (termios.ICANON | termios.ECHO | termios.ISIG),
                        termios.ICANON | termios.ECHO | termios.ISIG,
                    )
                    self.assertTrue(current[0] & termios.ICRNL)
                    self.assertFalse(current[0] & (termios.IGNCR | termios.INLCR))
                    os.write(master, b"RUN 0005 MATCHED_START\r")
                    ready, _, _ = self.script.select.select([stream], [], [], 1.0)
                    self.assertTrue(
                        ready, "Enter did not deliver a complete command line"
                    )
                    self.assertEqual(stream.readline(), "RUN 0005 MATCHED_START\n")
            self.assertEqual(termios.tcgetattr(slave), raw)
        finally:
            stream.close()
            os.close(master)
            os.close(slave)

    def test_broken_output_and_failed_diagnostic_do_not_skip_power_off_wait(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            stop = threading.Event()

            def ready(*args):
                self.assertTrue(stop.is_set())
                (output / "POWER_OFF_CONFIRMED").touch()
                return [], [], []

            with (
                patch.object(
                    self.script, "write_json", side_effect=OSError("disk full")
                ),
                patch.object(self.script.select, "select", side_effect=ready),
                patch("builtins.print", side_effect=BrokenPipeError),
            ):
                self.script.wait_for_power_off(output, stop, "input failure")

    def test_owner_failure_retains_controller_and_power_off_cleanup_survives_flush_failure(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            args = types.SimpleNamespace(
                out=Path(directory) / "owner",
                channel="fake_can",
                prepared=Path(directory),
            )
            args._active_action = types.SimpleNamespace(
                is_alive=lambda: True, join=unittest.mock.Mock()
            )
            recorder = types.SimpleNamespace(
                install=lambda *_: None,
                flush=unittest.mock.Mock(side_effect=OSError("disk full")),
                stop_threads_after_confirmed_power_off=unittest.mock.Mock(),
            )
            follower = Follower()
            follower._check = lambda: None
            modules = {
                "i2rt.motor_drivers.dm_driver": types.SimpleNamespace(
                    DMChainCanInterface=object, DMSingleMotorCanInterface=object
                ),
                "i2rt.robots.get_robot": types.SimpleNamespace(
                    get_yam_robot=lambda **_: object()
                ),
                "i2rt.robots.motor_chain_robot": types.SimpleNamespace(
                    MotorChainRobot=object
                ),
                "i2rt.robots.utils": types.SimpleNamespace(
                    GripperType=types.SimpleNamespace(LINEAR_4310=object())
                ),
            }
            with (
                patch.dict(sys.modules, modules),
                patch.object(self.script, "require_no_controller"),
                patch.object(self.script.sys, "stdin", io.StringIO("\n")),
                patch.object(
                    self.script, "owner_loop", side_effect=OSError("input failed")
                ),
            ):
                # StringIO.isatty is patched on the instance; no native constructor is reachable.
                self.script.sys.stdin.isatty = lambda: True
                with (
                    patch.object(
                        self.script, "NativeControllerRecorder", return_value=recorder
                    ),
                    patch.object(self.script, "HealthyFollower", return_value=follower),
                    patch.object(self.script, "wait_for_power_off") as wait,
                    patch.object(self.script.time, "sleep"),
                    patch.object(self.script.signal, "signal"),
                ):
                    with self.assertRaisesRegex(OSError, "disk full"):
                        self.script.execute(args, {})
                    wait.assert_called_once()
                    self.assertEqual(
                        self.script.sys.stdin.tell(),
                        0,
                        "--execute must not consume a typed initialization confirmation",
                    )
                    args._active_action.join.assert_called_once_with(timeout=10.0)
                    recorder.stop_threads_after_confirmed_power_off.assert_called_once()


if __name__ == "__main__":
    unittest.main()
