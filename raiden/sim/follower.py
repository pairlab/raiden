"""Simulated follower arm with the i2rt ``MotorChainRobot`` surface used by raiden."""

from __future__ import annotations

from typing import Dict

import numpy as np

from raiden.sim.client import DEFAULT_ADDRESS, SimConnection

# These gripper values only complete the MotorChainRobot gain vector.
# Each simulator owns its actual gripper controller and travel limits.
_GRIPPER_KP = 20.0
_GRIPPER_KD = 0.5


class SimFollower:
    """7-DoF (6 joints + normalized gripper, 1 = open) follower backed by the sim server."""

    def __init__(self, address: str = DEFAULT_ADDRESS, arm: str = "left"):
        self._c = SimConnection(address)
        try:
            rig = self._c.call("get_rig")
            available = tuple(rig.get("arms", ("left",)))
            if arm not in available:
                raise ValueError(f"simulator has arms {available}, cannot connect {arm!r}")
            self.arm = arm
            self._arm_kwargs = {"arm": arm} if "arms" in rig else {}
            kp, kd = self._c.call("get_kp_kd", **self._arm_kwargs)
            self._kp = np.concatenate([np.asarray(kp, dtype=np.float64)[:6], [_GRIPPER_KP]])
            self._kd = np.concatenate([np.asarray(kd, dtype=np.float64)[:6], [_GRIPPER_KD]])
        except BaseException:
            self._c.close()
            raise
        self.gravity_comp_factor = 1.0

    # -- i2rt Robot API ---------------------------------------------------
    def num_dofs(self) -> int:
        return 7

    def get_joint_pos(self) -> np.ndarray:
        return np.asarray(self._c.call("get_joint_pos", **self._arm_kwargs), dtype=np.float64)

    def get_observations(self) -> Dict[str, np.ndarray]:
        obs = self._c.call("get_observations", **self._arm_kwargs)
        return {k: np.asarray(v, dtype=np.float64) for k, v in obs.items()}

    def get_timed_observations(self):
        if not self._arm_kwargs:
            return self.get_observations(), {"state_timestamp_ns": -1, "state_timestamp_valid": False,
                                             "state_timestamp_source": "unavailable"}
        observation, timing = self._c.call("get_timed_observations", **self._arm_kwargs)
        return {key: np.asarray(value, dtype=np.float64) for key, value in observation.items()}, timing

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        pos = np.asarray(joint_pos, dtype=np.float64).reshape(-1)
        assert pos.shape == (7,), f"expected 7-d command, got {pos.shape}"
        self._c.call("command_joint_pos", joint_pos=pos, **self._arm_kwargs)

    def get_kp_kd(self):
        return self._kp.copy(), self._kd.copy()

    def update_kp_kd(self, kp: np.ndarray, kd: np.ndarray) -> None:
        kp = np.asarray(kp, dtype=np.float64).reshape(-1)
        kd = np.asarray(kd, dtype=np.float64).reshape(-1)
        assert kp.shape == self._kp.shape == kd.shape
        self._c.call("update_kp_kd", kp=kp, kd=kd, **self._arm_kwargs)
        self._kp, self._kd = kp.copy(), kd.copy()

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
