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
        self._accepts_outcome = False

    def start(self) -> None:
        result = self._c.call("start_action_log")
        if isinstance(result, dict):
            self._accepts_outcome = result.get("metadata", {}).get("native_attempts", {}).get("accepts_outcome", False)

    def _stop(self, outcome):
        kwargs = {"outcome": outcome} if self._accepts_outcome else {}
        return self._c.call("stop_action_log", **kwargs)

    def save(self, episode_dir: Path) -> None:
        log = self._stop("operator_success")
        if not isinstance(log, dict) or not log:
            raise RuntimeError("simulator returned no native action log")
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
        if "cmd" not in arrays or "sim_time" not in arrays:
            raise ValueError("native action log is missing cmd or sim_time")
        np.savez_compressed(str(dest / FILENAME), **arrays)

    def discard(self) -> None:
        self._stop("operator_failure")

    def close(self) -> None:
        self._c.close()
