"""BDDL task feedback from the sim, mirroring what MESA's ``collect_data.py`` prints.

The twin runs a BDDL problem, so it knows the language goal, the ordered subtask
predicates and whether the goal is satisfied.  :class:`TaskMonitor` polls that over RPC
and turns it into operator feedback: a one-off line per subtask completion for the teleop
terminal, and a short summary for the Rerun panel.
"""

from __future__ import annotations

import time
from typing import List, Optional

from raiden.sim.client import DEFAULT_ADDRESS, SimConnection


class TaskMonitor:
    """Polls ``get_task_status`` on its own connection, at most every ``period`` seconds."""

    def __init__(
        self,
        address: str = DEFAULT_ADDRESS,
        period: float = 0.5,
        conn: Optional[SimConnection] = None,
    ):
        # Own connection by default: polling must never queue behind a blocking render.
        self._c = conn or SimConnection(address)
        self._period = period
        self._next_t = 0.0
        self.status: dict = {}
        self.events: List[str] = []

    def poll(self, force: bool = False) -> bool:
        """Refresh if due (or ``force``); True when the status changed.  Never raises on a dead server."""
        now = time.monotonic()
        if now < self._next_t and not force:
            return False
        self._next_t = now + self._period
        try:
            new = self._c.call("get_task_status")
        except (RuntimeError, EOFError, OSError):
            return False
        old, self.status = self.status, new
        if old:
            for subtask, before, after in zip(
                new["subtasks"], old["done"], new["done"]
            ):
                if after and not before:
                    self.events.append(f"done: {subtask}")
            if new["success"] and not old["success"]:
                self.events.append("task success")
        return new != old

    def drain_events(self) -> List[str]:
        events, self.events = self.events, []
        return events

    @property
    def current(self) -> str:
        """The first unfinished subtask, empty once they are all done."""
        for subtask, done in zip(
            self.status.get("subtasks", []), self.status.get("done", [])
        ):
            if not done:
                return subtask
        return ""

    def summary(self) -> str:
        """Markdown for the Rerun text panel."""
        if not self.status:
            return "waiting for the sim..."
        lines = [f"# {self.status['language']}", ""]
        for subtask, done in zip(self.status["subtasks"], self.status["done"]):
            lines.append(f"- [{'x' if done else ' '}] {subtask}")
        lines.append("")
        lines.append("**SUCCESS**" if self.status["success"] else "_in progress_")
        return "\n".join(lines)

    def close(self) -> None:
        self._c.close()
