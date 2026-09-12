"""Drive the raiden teleop / recording stack against the MESA digital twin.

The MESA repo runs ``scripts/raiden_sim_server.py`` (its own Python 3.10 venv); this package
is the client side. :class:`SimFollower` duck-types the i2rt follower used by
:class:`raiden.robot.controller.RobotController`, :class:`SimCamera` implements the
:class:`raiden.cameras.Camera` interface, so Quest teleop, the recorder, the converter and
the LeRobot export run unchanged.
"""

from raiden.sim.calibration import table_pose, write_sim_calibration, write_sim_files
from raiden.sim.camera import SimCamera, load_sim_cameras
from raiden.sim.client import DEFAULT_ADDRESS, SimConnection, parse_address
from raiden.sim.follower import SimFollower
from raiden.sim.task import TaskMonitor

__all__ = [
    "DEFAULT_ADDRESS",
    "SimCamera",
    "SimConnection",
    "SimFollower",
    "TaskMonitor",
    "load_sim_cameras",
    "parse_address",
    "table_pose",
    "write_sim_calibration",
    "write_sim_files",
]
