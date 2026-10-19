"""Native simulator tick targets, command events, and physical/controller state.

The sampled ``robot_data.npz`` is separate from the simulator's applied tick log.
Isaac structures are stored as JSON; MuJoCo keeps its numeric NPZ format.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from raiden.sim.client import DEFAULT_ADDRESS, SimConnection
from raiden.sim.serialization import write_json, write_jsonl

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
        if not log:
            return
        dest = Path(episode_dir)
        log = dict(log)
        initial = log.pop("initial_state", None)
        if initial is not None:
            write_json(dest / "sim_initial_state.json", initial)
        if "command_events" in log:
            write_jsonl(dest / "sim_action_events.jsonl", log.pop("command_events"))
        if "states" in log:
            write_jsonl(
                dest / "sim_action_states.jsonl",
                ({"state": state, "t_ns": state["t_ns"], "sim_time": state["sim_time"]}
                 for state in log.pop("states")),
            )
        if "metadata" in log:
            write_json(dest / "sim_action_metadata.json", log.pop("metadata"))
        arrays = {key: np.asarray(value) for key, value in log.items()}
        structured = [key for key, value in arrays.items() if value.dtype.hasobject]
        if structured:
            raise ValueError(f"unexpected structured action log fields: {structured}")
        if arrays:
            np.savez_compressed(str(dest / FILENAME), **arrays)

    def discard(self) -> None:
        self._call("stop_action_log")

    def close(self) -> None:
        self._c.close()
