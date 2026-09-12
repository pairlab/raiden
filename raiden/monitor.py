"""Live view of a recording in progress.

Uses MESA's interface: a single OpenCV window, the same thing robosuite's ``OpenCVRenderer``
puts on screen during ``collect_data.py``.  The difference is what goes in it -- the frames
the recorder actually captured, tiled, with the BDDL task state drawn on top, so the operator
sees the data being collected rather than a separate free camera.

Frames arrive through :meth:`log`, which only parks the newest one; a background thread does
the tiling and drawing.  Recording threads are never blocked, and against the sim the rig
cameras cost the server nothing extra.  ``view=`` adds an operator camera: one extra render
on the server per monitor frame (~0.3 ms on the GPU).

``web=True`` swaps the window for a Rerun stream, for when the display is not local.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional

import cv2
import numpy as np

TILE_HEIGHT = 360
MAX_WIDTH = 1920
WINDOW = "raiden monitor"


class LiveMonitor:
    """Streams frames handed to :meth:`log`, plus sim task state when ``sim`` is set."""

    def __init__(
        self,
        camera_names: List[str],
        *,
        fps: float = 15.0,
        web: bool = False,
        web_port: int = 9090,
        sim: str = "",
        view: str = "",
        app_id: str = "raiden_monitor",
        guides=None,
    ):
        self.fps = fps
        self.guides = guides
        self.view = view
        self.names = ([view] if view else []) + list(camera_names)
        self._sim = sim
        self._web = web
        self._latest: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()

        if web:
            self._init_rerun(app_id, web_port)

        self._thread = threading.Thread(target=self._loop, name="monitor", daemon=True)
        self._thread.start()

    def _init_rerun(self, app_id: str, web_port: int) -> None:
        from urllib.parse import quote

        import rerun as rr
        import rerun.blueprint as rrb

        panels = [rrb.Spatial2DView(origin=f"/{n}", name=n) for n in self.names]
        if self._sim:
            panels.append(rrb.TextDocumentView(origin="/task", name="task"))
        rr.init(
            app_id,
            default_blueprint=rrb.Blueprint(
                rrb.Horizontal(*panels), collapse_panels=True
            ),
        )
        uri = rr.serve_grpc(grpc_port=web_port + 1)
        rr.serve_web_viewer(web_port=web_port, open_browser=False)
        print(f"\n  monitor: http://localhost:{web_port}?url={quote(uri, safe='')}\n")

    def log(self, name: str, image_bgr: np.ndarray) -> None:
        """Hand over the newest frame.  Called from grab threads, so it must stay cheap."""
        with self._lock:
            self._latest[name] = image_bgr

    def log_joints(self, arm: str, q: np.ndarray) -> None:
        """Park the newest arm joints for the grasp guides.  Called from the robot loop."""
        if self.guides is not None:
            self.guides.set_joints(arm, q)

    def _snapshot(self) -> Dict[str, np.ndarray]:
        with self._lock:
            return dict(self._latest)

    # -- rendering ---------------------------------------------------------
    def _tile(self, frames: Dict[str, np.ndarray], task) -> Optional[np.ndarray]:
        panels = []
        for name in self.names:
            img = frames.get(name)
            if img is None:
                continue
            img = self._with_guides(name, img)
            scale = TILE_HEIGHT / img.shape[0]
            panel = cv2.resize(img, (int(img.shape[1] * scale), TILE_HEIGHT))
            cv2.putText(
                panel, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )
            panels.append(panel)
        if not panels:
            return None
        tiled = np.hstack(panels)
        if tiled.shape[1] > MAX_WIDTH:
            scale = MAX_WIDTH / tiled.shape[1]
            tiled = cv2.resize(tiled, (MAX_WIDTH, int(tiled.shape[0] * scale)))
        if task is not None and task.status:
            self._draw_task(tiled, task)
        return tiled

    def _with_guides(self, name: str, img: np.ndarray) -> np.ndarray:
        """The MESA collector's grasp guides, on a copy so the recorder's frame is untouched."""
        if self.guides is None or not self.guides.has(name):
            return img
        return self.guides.draw(name, img.copy())

    @staticmethod
    def _draw_task(img: np.ndarray, task) -> None:
        lines = [task.status["language"]]
        lines += [
            f"[{'x' if done else ' '}] {subtask}"
            for subtask, done in zip(task.status["subtasks"], task.status["done"])
        ]
        if task.status["success"]:
            lines.append("SUCCESS")
        y = img.shape[0] - 14 * len(lines) - 10
        cv2.rectangle(img, (0, y - 16), (420, img.shape[0]), (0, 0, 0), -1)
        for i, line in enumerate(lines):
            done = line.startswith("[x]") or line == "SUCCESS"
            colour = (0, 255, 0) if done else (255, 255, 255)
            cv2.putText(
                img, line, (8, y + 14 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1
            )

    def _loop(self) -> None:
        conn = task = None
        if self._sim:
            from raiden.sim import SimConnection, TaskMonitor

            task = TaskMonitor(self._sim)
            if self.view:
                conn = SimConnection(self._sim)
        try:
            while not self._stop.is_set():
                if conn is not None:
                    try:
                        rgb = conn.call("render", cameras=[self.view], depth=False)
                        self.log(self.view, rgb[self.view]["rgb"][:, :, ::-1])
                    except (RuntimeError, EOFError, OSError):
                        pass
                if task is not None and task.poll():
                    for event in task.drain_events():
                        print(f"  {event}")
                self._show(self._snapshot(), task)
                self._stop.wait(1.0 / self.fps)
        finally:
            if conn is not None:
                conn.close()
            if task is not None:
                task.close()
            if not self._web:
                cv2.destroyWindow(WINDOW)
                cv2.waitKey(1)

    def _show(self, frames: Dict[str, np.ndarray], task) -> None:
        if self._web:
            import rerun as rr

            for name, image in frames.items():
                image = self._with_guides(name, image)
                rr.log(
                    name, rr.Image(image, color_model="BGR").compress(jpeg_quality=85)
                )
            if task is not None and task.status:
                rr.log(
                    "task", rr.TextDocument(task.summary(), media_type="text/markdown")
                )
            return
        tiled = self._tile(frames, task)
        if tiled is not None:
            cv2.imshow(WINDOW, tiled)
            cv2.waitKey(1)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


def make_monitor(
    camera_names: List[str],
    *,
    sim: str = "",
    view: str = "",
    web: bool = False,
    web_port: int = 9090,
    app_id: str = "raiden_monitor",
    guides=None,
) -> Optional[LiveMonitor]:
    """Build a monitor, or return None (with a warning) if it cannot start."""
    try:
        return LiveMonitor(
            camera_names,
            sim=sim,
            view=view,
            web=web,
            web_port=web_port,
            app_id=app_id,
            guides=guides,
        )
    except Exception as exc:  # a viewer must never take the recording down
        print(
            f"  monitor unavailable ({type(exc).__name__}: {exc}); recording without it"
        )
        return None
