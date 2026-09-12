"""Tiny RPC client for ``raiden_sim_server.py`` (multiprocessing.connection, pickle)."""

from __future__ import annotations

import threading
from multiprocessing.connection import Client
from typing import Any, Tuple

# NOT 5555: adb scans localhost 5554-5585 for emulators and its handshake wedges
# multiprocessing.connection's authkey exchange, killing the server's accept loop.
DEFAULT_ADDRESS = "127.0.0.1:5599"
_AUTHKEY = b"raiden-sim"


def parse_address(address: str) -> Tuple[str, int]:
    address = address or DEFAULT_ADDRESS
    if address in ("1", "true", "sim"):
        address = DEFAULT_ADDRESS
    host, _, port = address.rpartition(":")
    return host or "127.0.0.1", int(port or 5599)


class SimConnection:
    """One socket to the sim server; calls are serialised with a lock.

    Give each concurrent user (follower, every camera) its own connection so a
    blocking render never delays the 100 Hz command loop.
    """

    def __init__(self, address: str = DEFAULT_ADDRESS):
        self.address = address
        host, port = parse_address(address)
        try:
            self._conn = Client((host, port), authkey=_AUTHKEY)
        except ConnectionRefusedError as exc:
            raise ConnectionRefusedError(
                f"no raiden sim server at {host}:{port}; start it in the MESA repo with "
                "`DISPLAY=:0 MUJOCO_GL=glfw uv run python scripts/raiden_sim_server.py`"
            ) from exc
        self._lock = threading.Lock()

    def call(self, method: str, **kwargs: Any) -> Any:
        with self._lock:
            self._conn.send((method, kwargs))
            status, value = self._conn.recv()
        if status != "ok":
            raise RuntimeError(f"sim server {method} failed: {value}")
        return value

    def close(self) -> None:
        try:
            self._conn.close()
        except OSError:
            pass
