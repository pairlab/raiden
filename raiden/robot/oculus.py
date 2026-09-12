"""Meta Quest controller reader (poses + buttons) over ADB.

Compact re-implementation of rail-berkeley/oculus_reader (Apache-2.0).  The
companion Android app (``third_party/oculus_reader/teleop-debug.apk``) prints
both controller poses and button states to logcat at ~70 Hz; this module
installs/starts the app and parses that stream.

One-time setup: enable Developer Mode on the headset, ``sudo apt install
android-tools-adb``, plug in USB and accept the debugging prompt.  The headset
need not be worn: ``keep_awake()`` tells it the proximity sensor is covered so
tracking stays on while it sits on the desk facing the operator.
"""

import os
import re
import shutil
import subprocess
import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np

APK_NAME = "com.rail.oculus.teleop"
APK_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "third_party",
    "oculus_reader",
    "teleop-debug.apk",
)
_LOG_TAG = "wE9ryARX"
_BOOL_KEYS = {
    "R": ("A", "B", "RThU", "RJ", "RG", "RTr"),
    "L": ("X", "Y", "LThU", "LJ", "LG", "LTr"),
}


def parse_buttons(text: str) -> Dict[str, object]:
    """Parse the button field of one logcat line.

    Boolean keys: A B X Y, RJ/LJ (joystick click), RG/LG (grip), RTr/LTr (trigger).
    Tuple keys: rightJS/leftJS (x, y), rightGrip/leftGrip and rightTrig/leftTrig (0..1,).
    """
    parts = text.split(",")
    buttons: Dict[str, object] = {}
    for hand, keys in _BOOL_KEYS.items():
        if hand in parts:
            parts.remove(hand)
            buttons.update({k: False for k in keys})
    for key in list(buttons):
        if key in parts:
            buttons[key] = True
            parts.remove(key)
    for elem in parts:
        fields = elem.split(" ")
        if len(fields) < 2:
            continue
        try:
            buttons[fields[0]] = tuple(float(x) for x in fields[1:])
        except ValueError:
            pass
    return buttons


def parse_line(data: str) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Parse ``"l:<16 floats>|r:<16 floats>&<buttons>"`` into (poses, buttons)."""
    try:
        transforms_str, buttons_str = data.split("&")
    except ValueError:
        return {}, {}
    poses: Dict[str, np.ndarray] = {}
    for pair in transforms_str.split("|"):
        if ":" not in pair:
            continue
        hand, values = pair.split(":", 1)
        nums = [float(v) for v in values.split() if v]
        if len(nums) == 16:
            poses[hand] = np.array(nums, dtype=np.float64).reshape(4, 4)
    return poses, parse_buttons(buttons_str)


def _is_online(device) -> bool:
    """True if adb reports the device as ``device`` (offline ones raise)."""
    try:
        return device.get_state() == "device"
    except Exception:
        return False


class OculusReader:
    """Streams controller poses/buttons from the Quest app via ``adb logcat``."""

    def __init__(self, ip_address: Optional[str] = None, port: int = 5555):
        if shutil.which("adb") is None:
            raise RuntimeError(
                "adb not found. Install it with: sudo apt install android-tools-adb"
            )
        from ppadb.client import Client as AdbClient

        subprocess.run(["adb", "start-server"], check=False, capture_output=True)
        client = AdbClient(host="127.0.0.1", port=5037)
        if ip_address is not None:
            client.remote_connect(ip_address, port)
            self.device = client.device(f"{ip_address}:{port}")
            if self.device is None:
                raise RuntimeError(
                    f"Quest not reachable at {ip_address}:{port}. Plug in USB once and run "
                    f"`adb tcpip {port}`, then `adb shell ip route` to find its IP."
                )
        else:
            usb = [
                d
                for d in client.devices()
                if d.serial.count(".") < 3 and not d.serial.startswith("emulator-")
            ]
            # adb scans localhost ports 5554-5585 for emulators, so a raiden_sim_server on
            # 5555 shows up as an offline "emulator-5554"; only keep devices that answer.
            usb = [d for d in usb if _is_online(d)]
            if not usb:
                raise RuntimeError(
                    "No Quest found over USB. `adb devices` must list the headset as 'device' "
                    "(if 'unauthorized', put the headset on and accept USB debugging)."
                )
            self.device = usb[0]

        self._lock = threading.Lock()
        self._poses: Dict[str, np.ndarray] = {}
        self._buttons: Dict[str, object] = {}
        self._last_update = 0.0
        self._running = False
        self._install()

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def _install(self) -> None:
        if self.device.is_installed(APK_NAME):
            return
        apk = os.path.abspath(APK_PATH)
        if not os.path.exists(apk):
            raise RuntimeError(f"Quest teleop APK missing: {apk}")
        print("  - Installing Quest teleop app (first run only)...")
        self.device.install(apk, test=True)
        if not self.device.is_installed(APK_NAME):
            raise RuntimeError(
                "APK install failed. Is Developer Mode enabled on the headset?"
            )

    def keep_awake(self, enable: bool = True) -> None:
        """Make the Quest behave as if worn so tracking stays on while on the desk."""
        action = "prox_close" if enable else "prox_open"
        self.device.shell(f"am broadcast -a com.oculus.vrpowermanager.{action}")
        if enable:
            self.device.shell("input keyevent KEYCODE_WAKEUP")

    def _launch_app(self) -> None:
        self.device.shell(
            f'am start -n "{APK_NAME}/{APK_NAME}.MainActivity" '
            "-a android.intent.action.MAIN -c android.intent.category.LAUNCHER"
        )

    def _app_in_foreground(self) -> bool:
        out = (
            self.device.shell(
                "dumpsys activity activities | grep -m1 topResumedActivity"
            )
            or ""
        )
        return APK_NAME in out

    def controller_tracking(self) -> Dict[str, str]:
        """OS-level tracking state per hand, e.g. ``{"r": "POSITION", "l": "ORIENTATION"}``.

        POSITION = full 6-DoF; ORIENTATION = the cameras cannot see the controller
        (position frozen); NONE = controller asleep.  Returns {} on failure.
        """
        try:
            out = (
                self.device.shell("dumpsys OVRRemoteService | grep 'Paired device'")
                or ""
            )
        except Exception:
            return {}
        result: Dict[str, str] = {}
        for line in out.splitlines():
            m_type = re.search(r"Type:\s+(Right|Left)", line)
            m_track = re.search(r"TrackingStatus:\s+([A-Z_]+)", line)
            if m_type and m_track:
                result["r" if m_type.group(1) == "Right" else "l"] = m_track.group(1)
        return result

    def start(self) -> None:
        self._running = True
        self._launch_app()
        threading.Thread(target=self._stream, name="oculus-logcat", daemon=True).start()
        # The app only streams while it is the foreground immersive app; the
        # Quest shell steals focus whenever the worn state flips or a menu opens.
        threading.Thread(
            target=self._watchdog, name="oculus-watchdog", daemon=True
        ).start()

    def _stream(self) -> None:
        """Keep a logcat stream open; reopen it when adb drops it (USB hiccup, adb restart)."""
        while self._running:
            try:
                self.device.shell("logcat -T 0", handler=self._read_logcat)
            except Exception:
                pass
            if self._running:
                time.sleep(1.0)

    def _watchdog(self) -> None:
        while self._running:
            time.sleep(2.0)
            try:
                if self._running and not self._app_in_foreground():
                    self._launch_app()
            except Exception:
                pass

    def stop(self) -> None:
        self._running = False
        try:
            self.device.shell(f"am force-stop {APK_NAME}")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get(self) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
        """Latest (poses, buttons); poses are 4x4 controller->headset transforms."""
        with self._lock:
            return dict(self._poses), dict(self._buttons)

    @property
    def age(self) -> float:
        """Seconds since the last packet (inf before the first one)."""
        return (
            time.monotonic() - self._last_update if self._last_update else float("inf")
        )

    def _read_logcat(self, connection) -> None:
        f = connection.socket.makefile()
        try:
            while self._running:
                line = f.readline()
                if not line:
                    break
                if _LOG_TAG not in line:
                    continue
                poses, buttons = parse_line(line.split(_LOG_TAG + ": ", 1)[1].strip())
                if not poses:
                    continue
                with self._lock:
                    self._poses, self._buttons = poses, buttons
                    self._last_update = time.monotonic()
        except (UnicodeDecodeError, OSError):
            pass
        finally:
            f.close()
            connection.close()
