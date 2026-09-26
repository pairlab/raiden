"""The sim server's per-tick command log, saved beside a recorded episode.

``robot_data.npz`` samples the command on raiden's ~100 Hz robot loop, which runs free of the
sim's physics thread, so it cannot say which command each physics tick ran.
``sim_action_log.npz`` does, so ``scripts/sim_replay.py`` can replay the episode exactly.
A server without the logging RPCs writes no file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from raiden.sim.client import DEFAULT_ADDRESS, SimConnection

FILENAME = "sim_action_log.npz"


class ActionLog:
    """Starts and stops the server's per-tick log around one episode."""

    def __init__(self, address: str = DEFAULT_ADDRESS):
        self._c = SimConnection(address)
        self._warned = False

    def _call(self, method: str):
        try:
            return self._c.call(method)
        except (RuntimeError, EOFError, OSError) as exc:
            if not self._warned:
                print(f"  Note: no per-tick action log ({method}: {exc})")
                self._warned = True
            return None

    def start(self) -> None:
        self._call("start_action_log")

    def save(self, episode_dir: Path) -> None:
        log = self._call("stop_action_log")
        if log and len(log["cmd"]):
            np.savez_compressed(str(Path(episode_dir) / FILENAME), **log)

    def discard(self) -> None:
        self._call("stop_action_log")

    def close(self) -> None:
        self._c.close()
