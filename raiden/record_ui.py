"""Browser UI for ``rd record --ui``: live views with the task's placement limits, session controls and QC.

Served on http://localhost:<port> from a background thread. The recorder asks it for the task and teacher, tells it
the session phase, feeds it frames and polls it for the operator's commands, next to the leader/Quest buttons, the
pedal and the keyboard, which keep working. Nothing here moves the robot: the page starts, stops and labels recordings,
and deletes or relabels saved episodes when nothing is recording.

Frames: while a recording runs, the recorder's capture threads hand theirs over (``monitor``); in between, a pump
reads the open cameras itself, paused around every pipeline restart (``pause_pump`` / ``resume_pump``).
"""

from __future__ import annotations

import asyncio
import collections
import json
import queue
import shutil
import sys
import threading
import time
import traceback
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from aiohttp import WSMsgType, web

from raiden import qc
from raiden._config import CAMERA_CONFIG, DB_DIR
from raiden.camera_config import CameraConfig
from raiden.db.database import get_db

PAGE = Path(__file__).with_name("record_ui.html")
# Commands the page may send in each phase; anything else is dropped with a notice.
PHASE_COMMANDS = {
    "ready": {"start", "end"},
    "recording": {"stop", "success", "failure"},
    "verdict": {"success", "failure", "skip"},
}
RENDER_HZ, CHECK_HZ = 10.0, 3.0
WHITE, BLUE, ORANGE, GREEN, RED, YELLOW, CYAN = (
    (255, 255, 255), (255, 120, 30), (0, 140, 255), (0, 210, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0))


class _Tee:
    """A stdout/stderr that also hands every write to the page's terminal panel."""

    def __init__(self, stream, sink):
        self._stream, self._sink = stream, sink

    def write(self, text):
        n = self._stream.write(text)
        try:
            self._sink(text)
        except Exception:
            pass
        return n

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


class _Monitor:
    """What the recorder's capture threads call; forwards to the operator window too if --monitor is on."""

    def __init__(self, ui: "RecordUI", inner=None):
        self._ui, self._inner = ui, inner

    def log(self, name: str, image_bgr: np.ndarray) -> None:
        self._ui.put_frame(name, image_bgr)
        if self._inner is not None:
            self._inner.log(name, image_bgr)

    def log_joints(self, arm: str, q: np.ndarray) -> None:
        if self._inner is not None:
            self._inner.log_joints(arm, q)

    def close(self) -> None:
        if self._inner is not None:
            self._inner.close()


class RecordUI:
    def __init__(self, port: int = 8765, host: str = "127.0.0.1", data_dir: str = "data",
                 vla_dir: Path = qc.VLA_DEFAULT, open_browser: bool = True):
        self.url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}"
        self._host, self._port, self._open_browser = host, port, open_browser
        self._data = Path(data_dir)
        self._lock = threading.RLock()
        self._state: Dict = dict(phase="setup", notice="", live={}, episodes=[], coverage=[], session_tally={})
        self._cmd: Optional[str] = None  # the one pending session command
        self._setup_q: "queue.Queue[dict]" = queue.Queue()
        self._frames: Dict[str, Tuple[int, np.ndarray]] = {}  # camera -> (seq, latest frame)
        self._jpeg: Dict[str, Tuple[int, bytes]] = {}  # camera -> (seq, annotated JPEG)
        self._seq = 0
        self._stop = threading.Event()
        self._cameras: List = []
        self._pump_run = threading.Event()  # set: the pump may read the cameras
        self._pump_idle = threading.Event()  # set: the pump is not inside a camera call
        self._pump_idle.set()
        self._qc_q: "queue.Queue[Path]" = queue.Queue()
        self._results: Dict[str, Dict] = {}
        self._task: Optional[str] = None
        self._cfg = CameraConfig(CAMERA_CONFIG)
        self._geo = qc.load_geometry(vla_dir)
        self._settings = qc.QCSettings()
        self._threads: List[threading.Thread] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._log_buf: collections.deque = collections.deque(maxlen=400)
        self._log_partial = ""
        self._log_seq = 0
        self._log_lock = threading.Lock()
        self._streams = None
        self._last_poll = 0.0

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Start the server (the next free port if this one is taken) and the worker threads.

        Raises RuntimeError if no port can be bound, so the caller can fall back to the terminal.
        """
        for port in range(self._port, self._port + 5):
            self._port, self._bound, self._bind_error = port, threading.Event(), None
            threading.Thread(target=self._serve, name="record-ui-serve", daemon=True).start()
            self._bound.wait(timeout=5.0)
            if self._bind_error is None and self._bound.is_set():
                break
        else:
            raise RuntimeError(f"recording UI: no free port from {self._port - 4} ({self._bind_error})")
        self.url = f"http://{'localhost' if self._host in ('127.0.0.1', '0.0.0.0') else self._host}:{self._port}"
        for target in (self._render_loop, self._qc_loop, self._pump_loop):
            t = threading.Thread(target=target, name=f"record-ui{target.__name__}", daemon=True)
            t.start()
            self._threads.append(t)
        self._streams = (sys.stdout, sys.stderr)
        sys.stdout, sys.stderr = _Tee(sys.stdout, self._to_log), _Tee(sys.stderr, self._to_log)
        print(f"\n  Recording UI: {self.url}\n")
        if self._open_browser:
            threading.Timer(1.0, lambda: webbrowser.open(self.url)).start()

    def close(self) -> None:
        self.set_phase("ended")
        time.sleep(0.5)  # let the page see the end
        if self._streams is not None:
            sys.stdout, sys.stderr = self._streams
            self._streams = None
        self._stop.set()
        self._pump_run.clear()
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)

    # ------------------------------------------------------------------ session setup
    def setup_session(self) -> Tuple[str, str, int]:
        """Block until the page picks the task and teacher; returns (task name, instruction, teacher id)."""
        db = get_db()
        with self._lock:
            self._state["setup"] = dict(tasks=[dict(name=t["name"], instruction=t["instruction"]) for t in db.get_tasks()],
                                        teachers=[t["name"] for t in db.get_teachers()])
            self._state["phase"] = "setup"
        while True:
            s = self._setup_q.get()
            try:
                return self._apply_setup(s)
            except ValueError as e:
                self.notice(str(e))

    def _apply_setup(self, s: dict) -> Tuple[str, str, int]:
        from raiden.recorder import validate_task_name

        db = get_db()
        if s.get("new_task", {}).get("name"):
            name, instr = s["new_task"]["name"].strip(), s["new_task"].get("instruction", "").strip()
            err = validate_task_name(name)
            if err or not instr:
                raise ValueError(err or "the new task needs an instruction")
            if db.get_task_by_name(name) is None:
                db.add_task(name, instr)
        else:
            name = s.get("task") or ""
        task = db.get_task_by_name(name)
        if task is None:
            raise ValueError(f"no task {name!r}")
        tname = (s.get("new_teacher") or s.get("teacher") or "").strip()
        if not tname:
            raise ValueError("pick or add a teacher")
        teacher = db.get_teacher_by_name(tname)
        teacher_id = teacher["id"] if teacher else db.add_teacher(tname)
        with self._lock:
            self._task = task["name"]
            self._state.update(task=task["name"], instruction=task["instruction"], teacher=tname, phase="init")
            self._state.pop("setup", None)
        self._load_results()
        return task["name"], task["instruction"], teacher_id

    # ------------------------------------------------------------------ recorder -> UI
    def set_phase(self, phase: str, **info) -> None:
        with self._lock:
            self._state.update(phase=phase, **info)
            if phase != "verdict":
                self._state.pop("verdict_deadline", None)

    def update(self, **info) -> None:
        """Change state fields without changing the phase."""
        with self._lock:
            self._state.update(info)

    def poll_interface(self, interface) -> None:
        """Show the teleop device's state (Quest tracking and calibration); at most twice a second."""
        now = time.monotonic()
        if now - self._last_poll < 0.5:
            return
        self._last_poll = now
        try:
            status = interface.ready_hint or ""
        except Exception:
            status = ""
        self.update(calibrating=bool(getattr(interface, "calibrating", False)), device_status=status,
                    device_help=getattr(interface, "banner", "") or "")

    def set_tally(self, tally: Dict[str, int]) -> None:
        with self._lock:
            self._state["session_tally"] = dict(tally)

    def notice(self, text: str) -> None:
        with self._lock:
            self._state["notice"] = f"{datetime.now():%H:%M:%S} {text}"

    def episode_saved(self, saved_dir: Path) -> None:
        self._qc_q.put(Path(saved_dir))

    def monitor(self, inner=None) -> _Monitor:
        return _Monitor(self, inner)

    # ------------------------------------------------------------------ UI -> recorder
    def take(self, *names: str) -> Optional[str]:
        """The pending command if it is one of ``names`` (consumed), else None."""
        with self._lock:
            if self._cmd in names:
                cmd, self._cmd = self._cmd, None
                return cmd
        return None

    # ------------------------------------------------------------------ frames
    def attach_cameras(self, cameras: List) -> None:
        self._cameras = list(cameras)
        self._pump_run.set()

    def pause_pump(self) -> None:
        """Stop reading the cameras and wait until the pump is out of any camera call."""
        self._pump_run.clear()
        self._pump_idle.wait(timeout=5.0)

    def resume_pump(self) -> None:
        if self._cameras:
            self._pump_run.set()

    def put_frame(self, name: str, image_bgr: np.ndarray) -> None:
        with self._lock:
            self._seq += 1
            self._frames[name] = (self._seq, image_bgr)

    def _to_log(self, text: str) -> None:
        with self._log_lock:
            buf = self._log_partial + text
            parts = buf.split("\n")
            self._log_partial = parts.pop()
            for line in parts:
                line = line.rsplit("\r", 1)[-1]  # progress lines overwrite themselves
                self._log_buf.append(line)
            if parts:
                self._log_seq += 1

    def _pump_loop(self) -> None:
        while not self._stop.is_set():
            if not self._pump_run.wait(timeout=0.2):
                continue
            self._pump_idle.clear()
            try:
                for cam in self._cameras:
                    if not self._pump_run.is_set():
                        break
                    if cam.grab():
                        self.put_frame(cam.name, cam.get_frame().color.copy())
            except Exception as e:  # a camera hiccup must not end the session
                self.notice(f"camera read failed: {e!r}")
                time.sleep(0.5)
            finally:
                self._pump_idle.set()
            time.sleep(1.0 / 15)

    # ------------------------------------------------------------------ overlays and live checks
    def _render_loop(self) -> None:
        last_seq: Dict[str, int] = {}
        last_check = 0.0
        while not self._stop.is_set():
            time.sleep(1.0 / RENDER_HZ)
            with self._lock:
                frames = dict(self._frames)
            check = time.monotonic() - last_check > 1.0 / CHECK_HZ
            live = {}
            for name, (seq, img) in frames.items():
                if last_seq.get(name) == seq:
                    continue
                last_seq[name] = seq
                try:
                    v = img.copy()  # drawn on; the checks read the clean frame
                    if "wrist" in name:
                        live.update(self._wrist_overlay(img, v, check))
                    else:
                        live.update(self._scene_overlay(img, v, check))
                    ok, buf = cv2.imencode(".jpg", v, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        with self._lock:
                            self._jpeg[name] = (seq, buf.tobytes())
                except Exception:
                    traceback.print_exc()
            if check:
                last_check = time.monotonic()
                if live:
                    with self._lock:
                        self._state["live"] = {**self._state.get("live", {}), **live}

    def _scene_overlay(self, raw: np.ndarray, v: np.ndarray, check: bool) -> Dict:
        geo = self._geo
        for box, colour in ((geo.white, WHITE), (geo.blue, BLUE), (geo.orange, ORANGE)):
            uv = qc.box_outline(box, geo.table_z, geo).round().astype(np.int32)
            cv2.polylines(v, [uv], True, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.polylines(v, [uv], True, colour, 2, cv2.LINE_AA)
        live = {}
        c = self._state.get("live", {}).get("croissant")
        if check:
            live["scene_mean"] = round(float(raw.mean()), 1)
            p = qc.croissant_position(raw, geo)
            if p is None:
                c = None
            else:
                in_blue, in_white = qc.croissant_verdict(p[0], p[1], geo)
                cell = qc.grid_cell(p[0], p[1], geo)
                c = dict(x=round(p[0], 3), y=round(p[1], 3), in_blue=in_blue, in_white=in_white,
                         cell=list(cell) if cell else None)
            live["croissant"] = c
            o = qc.oven_position(raw, geo)
            if o is None:
                ov = None
            else:
                x_ok, y_ok, o_white = qc.oven_verdict(o[0], o[1], geo)
                ov = dict(x=round(o[0], 3), y=round(o[1], 3), match=o[2], x_ok=x_ok, y_ok=y_ok, in_white=o_white,
                          cell=qc.oven_cell(o[1], geo))
            live["oven"] = ov
        else:
            ov = self._state.get("live", {}).get("oven")
        if ov:
            f = qc.OVEN_FOOTPRINT
            box = (ov["x"] + f[0], ov["x"] + f[1], ov["y"] + f[2], ov["y"] + f[3])
            colour = GREEN if ov["x_ok"] and ov["y_ok"] and ov["in_white"] else RED
            uv = qc.box_outline(box, geo.table_z, geo).round().astype(np.int32)
            cv2.polylines(v, [uv], True, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.polylines(v, [uv], True, colour, 2, cv2.LINE_AA)
        if c:
            uv = qc.project(np.array([[c["x"], c["y"], geo.table_z + qc.CRO_COM_Z]]), geo.T, geo.K)[0]
            colour = GREEN if c["in_blue"] and c["in_white"] else RED
            cv2.circle(v, (int(uv[0]), int(uv[1])), 9, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.circle(v, (int(uv[0]), int(uv[1])), 9, colour, 2, cv2.LINE_AA)
        return live

    def _wrist_overlay(self, raw: np.ndarray, v: np.ndarray, check: bool) -> Dict:
        h, w = v.shape[:2]
        raw_mean = float(raw.mean())
        for k in (1, 2):  # camera_preview --grid
            cv2.line(v, (w * k // 3, 0), (w * k // 3, h), YELLOW, 1)
            cv2.line(v, (0, h * k // 3), (w, h * k // 3), YELLOW, 1)
        ref = self._settings.ref_tips
        for r_ in ref:
            cv2.line(v, (0, r_), (w, r_), CYAN, 1)
        live = {}
        wl = self._state.get("live", {}).get("wrist")
        if check:
            tips = qc.fingertips(raw)
            ok = [t is not None and abs(t[0] - r_) <= self._settings.max_tip_px for t, r_ in zip(tips, ref)]
            wl = dict(tips=[list(t) if t else None for t in tips], ok=all(ok), mean=round(raw_mean, 1),
                      bright_ok=raw_mean >= self._settings.min_wrist_mean)
            live["wrist"] = wl
        if wl:
            for t in wl["tips"]:
                if t:
                    cv2.circle(v, (t[1], t[0]), 7, GREEN if wl["ok"] else RED, 2, cv2.LINE_AA)
        return live

    # ------------------------------------------------------------------ QC and saved episodes
    def _qc_dir(self) -> Path:
        return self._data / "qc" / (self._task or "none")

    def _load_results(self) -> None:
        f = self._qc_dir() / "qc_results.json"
        with self._lock:
            self._results = json.load(open(f)) if f.exists() else {}
        raw = self._data / "raw" / self._task
        eps = sorted(p for p in raw.iterdir() if p.is_dir() and p.name.isdigit()) if raw.exists() else []
        names = {e.name for e in eps}
        with self._lock:
            for gone in set(self._results) - names:
                del self._results[gone]
        for e in eps:
            mf = e / "metadata.json"
            if mf.exists() and json.load(open(mf)).get("complete", False):
                r = self._results.get(e.name)
                if r is None or r.get("_stamp") != mf.stat().st_mtime or r.get("_qc_version") != qc.QC_VERSION:
                    self._qc_q.put(e)
        self._publish_results()

    def _qc_loop(self) -> None:
        while not self._stop.is_set():
            try:
                ep = self._qc_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                r = qc.check(ep, self._task, self._geo, self._cfg, self._settings, get_db())
            except Exception as e:
                r = dict(episode=ep.name, reasons=[f"QC error: {e!r}"], duration_s=None)
            mf = ep / "metadata.json"
            r["_stamp"] = mf.stat().st_mtime if mf.exists() else None
            with self._lock:
                self._results[ep.name] = r
            self._log(qc.line(r))
            self._publish_results()

    def _publish_results(self) -> None:
        with self._lock:
            qc.apply_duration_outliers(self._results)
            G, outside = qc.coverage(self._results, self._geo)
            eps = [dict(episode=k, duration_s=r.get("duration_s"), label=r.get("label"), reasons=r.get("reasons", []),
                        croissant=r.get("croissant"), oven=r.get("oven"), closes=r.get("closes"))
                   for k, r in sorted(self._results.items(), reverse=True)]
            O, o_out = qc.oven_coverage(self._results, self._geo)
            oy0, oy1 = self._geo.oven_region[2], self._geo.oven_region[3]
            self._state.update(oven_coverage=O.tolist(), oven_outside=o_out,
                               oven_grid=np.linspace(oy1, oy0, qc.OVEN_GRID + 1).round(3).tolist(),
                               oven_sim_x=round((self._geo.oven_region[0] + self._geo.oven_region[1]) / 2, 3))
            bx0, bx1, by0, by1 = self._geo.blue
            self._state.update(episodes=eps, coverage=G.tolist(), coverage_outside=outside,
                               grid=dict(x=np.linspace(bx0, bx1, qc.GRID_X + 1).round(3).tolist(),
                                         y=np.linspace(by1, by0, qc.GRID_Y + 1).round(3).tolist()),
                               qc_tally=dict(total=len(eps), passed=sum(not e["reasons"] for e in eps)))
            results = dict(self._results)
        if self._task:
            d = self._qc_dir()
            d.mkdir(parents=True, exist_ok=True)
            json.dump(results, open(d / "qc_results.json", "w"), indent=1)

    def _log(self, text: str) -> None:
        if self._task:
            d = self._qc_dir()
            d.mkdir(parents=True, exist_ok=True)
            with open(d / "qc_log.txt", "a") as f:
                f.write(text + "\n")

    def _relabel(self, episode: str, status: str) -> None:
        db = get_db()
        demo = db.get_demonstration_by_raw_path(f"{self._data}/raw/{self._task}/{episode}")
        if demo is None:
            raise ValueError(f"episode {episode} has no DB row")
        db.update_demonstration(demo["id"], status=status)
        with self._lock:
            if episode in self._results:
                self._results[episode]["label"] = status
        self._publish_results()
        self.notice(f"{episode} marked {status}")

    def _delete(self, episode: str) -> None:
        if self._state.get("phase") in ("recording", "verdict", "saving"):
            raise ValueError("finish the current recording first")
        ep = self._data / "raw" / self._task / episode
        db = get_db()
        demo = db.get_demonstration_by_raw_path(f"{self._data}/raw/{self._task}/{episode}")
        if demo is None and not ep.exists():
            raise ValueError(f"no episode {episode}")
        backup = Path(DB_DIR) / f"demonstrations.json.bak-{datetime.now():%Y%m%d-%H%M%S}-pre-delete"
        shutil.copy2(Path(DB_DIR) / "demonstrations.json", backup)
        if demo:
            conv = demo.get("converted_data_path")
            db.delete_demonstration(demo["id"])
            if conv and Path(conv).exists():
                shutil.rmtree(conv)
        if ep.exists():
            shutil.rmtree(ep)
        with self._lock:
            self._results.pop(episode, None)
        self._log(f"{episode} deleted from the UI (DB backup {backup.name})")
        self._publish_results()
        self.notice(f"{episode} deleted")

    # ------------------------------------------------------------------ web server
    def _serve(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        app = web.Application()
        app.router.add_get("/", self._page)
        app.router.add_get("/ws", self._ws)
        app.router.add_get("/stream/{camera}", self._stream)
        runner = web.AppRunner(app, access_log=None)
        self._loop.run_until_complete(runner.setup())
        try:
            self._loop.run_until_complete(web.TCPSite(runner, self._host, self._port).start())
        except OSError as e:
            self._bind_error = e
            self._loop.run_until_complete(runner.cleanup())
            self._bound.set()
            return
        self._bound.set()
        try:
            self._loop.run_forever()
        finally:
            self._loop.run_until_complete(runner.cleanup())

    async def _page(self, request):
        return web.FileResponse(PAGE)

    async def _stream(self, request):
        name = request.match_info["camera"]
        resp = web.StreamResponse(headers={"Content-Type": "multipart/x-mixed-replace; boundary=frame",
                                           "Cache-Control": "no-cache"})
        await resp.prepare(request)
        last = -1
        try:
            while not self._stop.is_set():
                with self._lock:
                    seq, jpg = self._jpeg.get(name, (-1, b""))
                if seq != last and jpg:
                    last = seq
                    await resp.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                await asyncio.sleep(1.0 / RENDER_HZ)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        return resp

    async def _ws(self, request):
        ws = web.WebSocketResponse(heartbeat=10)
        await ws.prepare(request)

        async def push():
            sent_seq = -1
            while not ws.closed and not self._stop.is_set():
                with self._lock:
                    st = dict(self._state, now=time.time())
                with self._log_lock:
                    if self._log_seq != sent_seq:
                        sent_seq = self._log_seq
                        st["log"] = list(self._log_buf)[-300:]
                await ws.send_str(json.dumps(st, default=str))
                await asyncio.sleep(0.25)

        pusher = asyncio.ensure_future(push())
        try:
            async for m in ws:
                if m.type == WSMsgType.TEXT:
                    try:
                        self._command(json.loads(m.data))
                    except Exception as e:
                        self.notice(f"{e}")
        finally:
            pusher.cancel()
        return ws

    def _command(self, c: dict) -> None:
        cmd = c.get("cmd")
        if cmd == "session":
            if self._state.get("phase") != "setup":
                raise ValueError("the session has already started")
            self._setup_q.put(c)
        elif cmd == "label":
            self._relabel(c["episode"], c["status"])
        elif cmd == "delete":
            self._delete(c["episode"])
        else:
            with self._lock:
                allowed = PHASE_COMMANDS.get(self._state.get("phase"), set())
                if cmd not in allowed:
                    raise ValueError(f"'{cmd}' does nothing while {self._state.get('phase')}")
                self._cmd = cmd
