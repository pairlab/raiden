"""Controller logging tests use fake classes and never construct native drivers."""

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from raiden.sim.hardware_session import NativeControllerRecorder


class InstrumentationTest(unittest.TestCase):
    def test_wrappers_preserve_arguments_results_and_log_seven_feedbacks(self):
        @dataclass
        class Feedback:
            position: float
            velocity: float = 0.0

        class Motor:
            def set_control(self, motor_id, motor_type, pos, vel, kp, kd, torque):
                self.last = (motor_id, motor_type, pos, vel, kp, kd, torque)
                return Feedback(pos, vel)

            def motor_on(self, *args, **kwargs):
                return "original on"

            motor_off = motor_on
            clean_error = motor_on

        class Robot:
            def __init__(self):
                self.updates = 0
                self._joint_state = SimpleNamespace(
                    pos=np.zeros(7), vel=np.zeros(7), eff=np.zeros(7)
                )
                self.motor_chain = SimpleNamespace(running=True)
                self._server_thread = SimpleNamespace(is_alive=lambda: True)

            def _update_joint_state(self, torques, commands, encoder_infos=None):
                self.updates += 1
                return "original update"

        recorder = NativeControllerRecorder()
        recorder.install(Motor, Robot)
        motor, robot = Motor(), Robot()
        for motor_id in range(1, 8):
            result = motor.set_control(motor_id, "DM4310", 0.2, 0.3, 20.0, 0.5, 0.1)
            self.assertEqual(result, Feedback(0.2, 0.3))
            self.assertEqual(motor.last, (motor_id, "DM4310", 0.2, 0.3, 20.0, 0.5, 0.1))
        self.assertTrue(recorder.health(robot)["healthy"])
        commands = SimpleNamespace(
            pos=np.zeros(7), vel=np.zeros(7), kp=np.ones(7), kd=np.ones(7)
        )
        self.assertEqual(
            robot._update_joint_state(np.zeros(7), commands), "original update"
        )
        self.assertEqual(robot.updates, 1)
        self.assertEqual(len(recorder.controller_rows), 1)
        # Even malformed diagnostic fields must not prevent the native operation.
        self.assertEqual(
            robot._update_joint_state(np.zeros(7), object()), "original update"
        )
        self.assertEqual(robot.updates, 2)
        self.assertEqual(recorder.logging_error_count, 1)
        with tempfile.TemporaryDirectory() as directory:
            recorder.flush(Path(directory))
            self.assertTrue(
                (Path(directory) / "native_can_transactions.jsonl.gz").exists()
            )
        recorder.last_feedback_ns[1] = 0
        self.assertFalse(recorder.health(robot)["healthy"])


if __name__ == "__main__":
    unittest.main()
