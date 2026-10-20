"""Attended hardware session helpers; importing this module cannot open a driver."""

# ruff: noqa: BLE001, S110 -- instrumentation failures must never stop native control
from __future__ import annotations

import gzip
import json
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from .hardware_replay import load_hardware_plan


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")


class NativeControllerRecorder:
    """Observe existing methods without changing their arguments or control math.

    Native set_control times bracket a host CAN request/feedback transaction.
    There is no device application timestamp in that response protocol.
    """

    def __init__(self, capacity=200_000):
        self.capacity = capacity
        self.can_rows = deque(maxlen=capacity)
        self.controller_rows = deque(maxlen=capacity)
        self.bootstrap_events = []
        self.total_can_rows = 0
        self.total_controller_rows = 0
        self.last_feedback_ns = {}
        self.lock = threading.Lock()
        self.originals = []
        self.robots = []
        self.chains = []
        self.logging_errors = deque(maxlen=100)
        self.logging_error_count = 0

    def log_error(self, stage, error):
        # Diagnostics must never prevent the unchanged native control call.
        try:
            self.logging_error_count += 1
            self.logging_errors.append(
                {
                    "stage": stage,
                    "type": type(error).__name__,
                    "message": str(error),
                    "host_monotonic_ns": time.monotonic_ns(),
                }
            )
        except Exception:
            pass

    def install(self, single_motor_class, robot_class, chain_class=None):
        recorder = self
        original_robot_init = robot_class.__init__

        def tracked_robot_init(robot, *args, **kwargs):
            recorder.robots.append(robot)
            return original_robot_init(robot, *args, **kwargs)

        self.originals.append((robot_class, "__init__", original_robot_init))
        robot_class.__init__ = tracked_robot_init
        if chain_class is not None:
            original_chain_init = chain_class.__init__

            def tracked_chain_init(chain, *args, **kwargs):
                recorder.chains.append(chain)
                return original_chain_init(chain, *args, **kwargs)

            self.originals.append((chain_class, "__init__", original_chain_init))
            chain_class.__init__ = tracked_chain_init
        original_send = single_motor_class.set_control

        def logged_send(interface, motor_id, motor_type, pos, vel, kp, kd, torque):
            before = time.monotonic_ns()
            try:
                result = original_send(
                    interface, motor_id, motor_type, pos, vel, kp, kd, torque
                )
            except BaseException as error:
                try:
                    recorder.bootstrap_events.append(
                        {
                            "kind": "CAN transaction failure",
                            "motor_id": motor_id,
                            "monotonic_ns": time.monotonic_ns(),
                            "exception": f"{type(error).__name__}: {error}",
                        }
                    )
                except Exception as log_error:
                    recorder.log_error("CAN error logging", log_error)
                raise
            after = time.monotonic_ns()
            try:
                row = {
                    "motor_id": motor_id,
                    "motor_type": motor_type,
                    "host_send_before_monotonic_ns": before,
                    "host_feedback_after_monotonic_ns": after,
                    "raw_motor_position_rad": pos,
                    "raw_motor_velocity_rad_s": vel,
                    "onboard_kp": kp,
                    "onboard_kd": kd,
                    "feedforward_torque_Nm": torque,
                    "feedback": jsonable(vars(result)),
                }
                with recorder.lock:
                    recorder.can_rows.append(row)
                    recorder.total_can_rows += 1
                    recorder.last_feedback_ns[motor_id] = after
            except Exception as log_error:
                recorder.log_error("CAN feedback logging", log_error)
            return result

        self.originals.append((single_motor_class, "set_control", original_send))
        single_motor_class.set_control = logged_send
        original_update = robot_class._update_joint_state

        def logged_update(robot, motor_torques, joint_commands, encoder_infos=None):
            # Called with the native state lock held, after the existing limiter.
            row = None
            try:
                limiter = getattr(robot, "_gripper_force_limiter", None)
                row = {
                    "host_buffer_before_monotonic_ns": time.monotonic_ns(),
                    "post_limiter_raw_position_rad": joint_commands.pos.copy(),
                    "post_limiter_raw_velocity_rad_s": joint_commands.vel.copy(),
                    "onboard_kp": joint_commands.kp.copy(),
                    "onboard_kd": joint_commands.kd.copy(),
                    "feedforward_torque_Nm": motor_torques.copy(),
                    "observed_joint_position_normalized_gripper": robot._joint_state.pos.copy(),
                    "observed_joint_velocity_driver_units": robot._joint_state.vel.copy(),
                    "observed_joint_effort_motor_Nm": robot._joint_state.eff.copy(),
                    "limiter_clogged": None if limiter is None else limiter._is_clogged,
                    "limiter_adjusted_raw_position_rad": None
                    if limiter is None
                    else limiter._gripper_adjusted_qpos,
                }
            except Exception as log_error:
                recorder.log_error("lower-controller snapshot", log_error)
            result = original_update(
                robot, motor_torques, joint_commands, encoder_infos
            )
            if row is not None:
                try:
                    row["host_buffer_after_monotonic_ns"] = time.monotonic_ns()
                    with recorder.lock:
                        recorder.controller_rows.append(row)
                        recorder.total_controller_rows += 1
                except Exception as log_error:
                    recorder.log_error("lower-controller append", log_error)
            return result

        self.originals.append((robot_class, "_update_joint_state", original_update))
        robot_class._update_joint_state = logged_update
        for name in ("motor_on", "motor_off", "clean_error"):
            original = getattr(single_motor_class, name)

            def logged_bootstrap(
                interface, *args, _original=original, _name=name, **kwargs
            ):
                try:
                    recorder.bootstrap_events.append(
                        {
                            "kind": _name,
                            "args": jsonable(args),
                            "kwargs": jsonable(kwargs),
                            "host_before_monotonic_ns": time.monotonic_ns(),
                        }
                    )
                except Exception as log_error:
                    recorder.log_error("bootstrap event logging", log_error)
                return _original(interface, *args, **kwargs)

            self.originals.append((single_motor_class, name, original))
            setattr(single_motor_class, name, logged_bootstrap)

    def health(self, follower, maximum_feedback_age_ns=100_000_000):
        now = time.monotonic_ns()
        with self.lock:
            ages = {
                str(key): now - value for key, value in self.last_feedback_ns.items()
            }
        running = bool(follower.motor_chain.running)
        thread_alive = bool(follower._server_thread.is_alive())
        fresh = all(
            str(i) in ages and ages[str(i)] <= maximum_feedback_age_ns
            for i in range(1, 8)
        )
        return {
            "motor_chain_running": running,
            "controller_thread_alive": thread_alive,
            "all_seven_motor_feedback_transactions_fresh": fresh,
            "feedback_transaction_age_ns": ages,
            "maximum_feedback_age_ns": maximum_feedback_age_ns,
            "healthy": running and thread_alive and fresh,
        }

    def flush(self, destination):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        with self.lock:
            can_rows, controller_rows = list(self.can_rows), list(self.controller_rows)
            totals = (self.total_can_rows, self.total_controller_rows)
        for name, rows in [
            ("native_can_transactions", can_rows),
            ("lower_controller", controller_rows),
        ]:
            with gzip.open(destination / f"{name}.jsonl.gz", "wt") as stream:
                for row in rows:
                    stream.write(json.dumps(jsonable(row), allow_nan=False) + "\n")
        write_json(
            destination / "instrumentation.json",
            {
                "native_can_transactions": len(can_rows),
                "native_can_total": totals[0],
                "lower_controller_rows": len(controller_rows),
                "lower_controller_total": totals[1],
                "oldest_rows_discarded": [
                    totals[0] - len(can_rows),
                    totals[1] - len(controller_rows),
                ],
                "bootstrap_events": self.bootstrap_events,
                "logging_error_count": self.logging_error_count,
                "logging_errors": list(self.logging_errors),
                "interpretation": "Host timestamps around original CAN request/feedback and post-limiter command-buffer submissions. Device application timestamps unavailable. No control math or gains changed by instrumentation.",
            },
        )

    def stop_threads_after_confirmed_power_off(self):
        """Only thread cleanup, only after the caller confirms physical motor power off."""
        for robot in self.robots:
            event = getattr(robot, "_stop_event", None)
            if event is not None:
                event.set()
        for chain in self.chains:
            chain.running = False
        for robot in self.robots:
            thread = getattr(robot, "_server_thread", None)
            if thread is not None:
                thread.join(timeout=2)


class HealthyFollower:
    """Reject cached observations if native control/feedback has stopped."""

    def __init__(self, robot, recorder):
        self.robot, self.recorder = robot, recorder

    def _check(self):
        report = self.recorder.health(self.robot)
        if not report["healthy"]:
            raise RuntimeError(f"native controller/feedback health failed: {report}")

    def get_robot_info(self):
        self._check()
        return self.robot.get_robot_info()

    def get_observations(self):
        self._check()
        return self.robot.get_observations()

    def command_joint_pos(self, command):
        self._check()
        return self.robot.command_joint_pos(command)


def inspect_prepared_start(plan_path, follower, recorder=None, robot=None):
    """Read-only staged report. The executor independently rechecks before sending."""
    arrays, plan = load_hardware_plan(plan_path)
    info, obs = follower.get_robot_info(), follower.get_observations()
    q, dq = (
        np.r_[obs["joint_pos"], obs["gripper_pos"]],
        np.r_[obs["joint_vel"], obs["gripper_vel"]],
    )
    error, velocity_error = q - arrays["initial_q"], dq - arrays["initial_dq"]
    controller_matches = all(
        np.allclose(info[key], value, rtol=0, atol=1e-9)
        for key, value in plan["expected_controller"].items()
    )
    ready = (
        controller_matches
        and np.isfinite(q).all()
        and np.isfinite(dq).all()
        and np.max(abs(error[:6])) <= 0.01
        and abs(error[6]) <= 0.02
        and np.max(abs(velocity_error)) <= 0.05
    )
    return {
        "host_time_ns": time.time_ns(),
        "host_monotonic_ns": time.monotonic_ns(),
        "initial_q_rad_aperture": q,
        "initial_dq_rad_s_aperture_s": dq,
        "initial_error_rad_aperture": error,
        "initial_velocity_error": velocity_error,
        "controller_matches": controller_matches,
        "robot_info": info,
        "initial_observation": obs,
        "numerical_start_ready": bool(ready),
        "health": None if recorder is None else recorder.health(robot),
        "physical_start": "operator assertion plus separately retained live camera; not inferred from old demonstrations",
        "next_action": "managed observed-pose PD/gravity hold; await root review before RUN",
    }
