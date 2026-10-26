"""Own an Isaac server subprocess and connect the existing teleop/record paths."""

from __future__ import annotations

import math
import os
import signal
import socket
import subprocess
import tempfile
import time
from pathlib import Path

from raiden.sim.client import require_arm_mode


class IsaacServer:
    """A local server with explicit scene, mode, environment, and process lifetime."""

    def __init__(
        self,
        *,
        scene,
        arms,
        project,
        python,
        task_config=None,
        port=5599,
        fps=30,
        seed=0,
        startup_timeout=180,
        log=None,
        visualizer="none",
    ):
        self.scene = Path(scene).expanduser().resolve()
        self.project = Path(project).expanduser().resolve()
        # Preserve the venv/bin/python path; resolving its symlink would bypass
        # that interpreter's pyvenv.cfg and discard the pinned environment.
        self.python = Path(python).expanduser().absolute()
        self.visualizer = visualizer
        self.arms, self.port = arms, int(port)
        self.address = f"127.0.0.1:{self.port}"
        self.timeout = float(startup_timeout)
        if arms not in ("single", "bimanual") or not 1 <= self.port <= 65535:
            raise ValueError(
                "explicit single/bimanual mode and a valid port are required"
            )
        if (
            not self.scene.exists()
            or not self.python.is_file()
            or not (self.project / "scripts/raiden_sim_server.py").is_file()
        ):
            raise ValueError("scene, Isaac Python and simulator project must exist")
        if (
            not math.isfinite(fps)
            or not math.isfinite(self.timeout)
            or fps <= 0
            or self.timeout <= 0
        ):
            raise ValueError("camera rate and startup timeout must be positive")
        if log is None:
            fd, name = tempfile.mkstemp(prefix="raiden-isaac-", suffix=".log")
            os.close(fd)
            self.log = Path(name)
        else:
            self.log = Path(log).expanduser().resolve()
            self.log.parent.mkdir(parents=True, exist_ok=True)
        self.command = [
            str(self.python),
            "scripts/raiden_sim_server.py",
            "--scene",
            str(self.scene),
            "--arms",
            arms,
            "--port",
            str(self.port),
            "--fps",
            str(fps),
            "--seed",
            str(seed),
            "--visualizer",
            visualizer,
        ]
        if task_config:
            config = Path(task_config).expanduser().resolve()
            if not config.is_file():
                raise ValueError("task configuration must exist")
            self.command += ["--task-config", str(config)]
        self.process = None
        self._stream = None

    def __enter__(self):
        # Never attach the collection session to an unrelated existing server.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", self.port))
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, [str(self.project / "src"), environment.get("PYTHONPATH")])
        )
        environment["HEADLESS"] = "1" if self.visualizer == "none" else "0"
        self._stream = self.log.open("w")
        try:
            self.process = subprocess.Popen(
                self.command,
                cwd=self.project,
                env=environment,
                stdout=self._stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            print(f"Isaac server log: {self.log}", flush=True)
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"Isaac exited with status {self.process.returncode}; see {self.log}"
                    )
                # The server only starts listening after scene validation and camera warmup.
                try:
                    rig = require_arm_mode(self.address, self.arms)
                except OSError:
                    time.sleep(0.1)
                    continue
                if rig.get("backend") != "isaac":
                    raise RuntimeError("launched server does not identify as Isaac")
                return self
            raise TimeoutError(
                f"Isaac did not become ready within {self.timeout}s; see {self.log}"
            )
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def __exit__(self, *_):
        self.close()
