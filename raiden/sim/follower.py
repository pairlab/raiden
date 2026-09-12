"""Simulated follower arm with the i2rt ``MotorChainRobot`` surface used by raiden."""

from __future__ import annotations

from typing import Dict

import numpy as np

from raiden.sim.client import DEFAULT_ADDRESS, SimConnection

# The sim gripper is a MuJoCo position actuator; these mirror the real LINEAR_4310 motor
# gains only so RobotController can store/restore "gains" uniformly.
_GRIPPER_KP = 20.0
_GRIPPER_KD = 0.5


class SimFollower:
    """7-DoF (6 joints + normalized gripper, 1 = open) follower backed by the sim server."""

    def __init__(self, address: str = DEFAULT_ADDRESS):
        self._c = SimConnection(address)
        kp, kd = self._c.call("get_kp_kd")
        self._kp = np.concatenate([np.asarray(kp, dtype=np.float64)[:6], [_GRIPPER_KP]])
        self._kd = np.concatenate([np.asarray(kd, dtype=np.float64)[:6], [_GRIPPER_KD]])
        self.gravity_comp_factor = 1.0

    # -- i2rt Robot API ---------------------------------------------------
    def num_dofs(self) -> int:
        return 7

    def get_joint_pos(self) -> np.ndarray:
        return np.asarray(self._c.call("get_joint_pos"), dtype=np.float64)

    def get_observations(self) -> Dict[str, np.ndarray]:
        obs = self._c.call("get_observations")
        return {k: np.asarray(v, dtype=np.float64) for k, v in obs.items()}

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        pos = np.asarray(joint_pos, dtype=np.float64).reshape(-1)
        assert pos.shape == (7,), f"expected 7-d command, got {pos.shape}"
        self._c.call("command_joint_pos", joint_pos=pos)

    def update_kp_kd(self, kp: np.ndarray, kd: np.ndarray) -> None:
        kp = np.asarray(kp, dtype=np.float64).reshape(-1)
        kd = np.asarray(kd, dtype=np.float64).reshape(-1)
        assert kp.shape == self._kp.shape == kd.shape
        self._kp, self._kd = kp.copy(), kd.copy()
        self._c.call("update_kp_kd", kp=kp, kd=kd)

    def close(self) -> None:
        self._c.close()

    # -- extras ------------------------------------------------------------
    def reset_scene(self) -> dict:
        """Re-sample the task objects (arm goes to the model home)."""
        return self._c.call("reset")

    def get_object_poses(self) -> dict:
        return self._c.call("get_object_poses")

    def get_rig(self) -> dict:
        return self._c.call("get_rig")
