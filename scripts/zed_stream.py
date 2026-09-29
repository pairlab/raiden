#!/usr/bin/env python
"""Stream the rig's GMSL ZED cameras from the ZED Box (Jetson) to the recording PC.

Runs on the Jetson, with pyzed as its only dependency.  Each camera is opened at
SVGA (960x600) @ 30 fps and streamed with the Jetson's hardware encoder on its
own port; ``rd record`` and ``camera_preview.py`` receive them through the
``stream`` entries in ``~/.config/raiden/camera.json`` and record SVO2 on the PC.
ZED X One (mono) cameras are opened with ``sl.CameraOne``, stereo ZED X with
``sl.Camera`` (both views are streamed, so depth can be computed at conversion).

    ssh pair@192.168.50.2 '~/pair-mse/yam_teleop/.conda-env/bin/python -' < scripts/zed_stream.py
    ssh pair@192.168.50.2 '~/pair-mse/yam_teleop/.conda-env/bin/python - --camera 41925345:30004' < scripts/zed_stream.py

Default cameras: left wrist 301058360:30000, right wrist 306353224:30002,
front ZED X 41925345:30004.  A GMSL camera can be opened by one process only.
Ctrl-C (or closing the ssh session) stops the streams.
"""

import argparse
import collections
import signal
import threading
import time

import pyzed.sl as sl

CAMERAS = ("301058360:30000", "306353224:30002", "41925345:30004")


def stream(serial: int, port: int, args, stop: threading.Event, stats: dict) -> None:
    mono = serial in {d.serial_number for d in sl.CameraOne.get_device_list()}
    cam = sl.CameraOne() if mono else sl.Camera()
    init = sl.InitParametersOne() if mono else sl.InitParameters()
    init.set_from_serial_number(serial)
    init.camera_resolution = getattr(sl.RESOLUTION, args.resolution)
    init.camera_fps = args.fps
    if not mono:
        init.depth_mode = sl.DEPTH_MODE.NONE
    status = cam.open(init)
    if status == sl.ERROR_CODE.SUCCESS:
        params = sl.StreamingParameters()
        params.port = port
        params.codec = getattr(sl.STREAMING_CODEC, args.codec)
        params.bitrate = args.bitrate
        status = cam.enable_streaming(params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"{serial}: open/stream on port {port} failed: {status}", flush=True)
        cam.close()
        stop.set()
        return
    info = cam.get_camera_information()
    res = info.camera_configuration.resolution
    print(
        f"{serial} ({info.camera_model}) {res.width}x{res.height}@{args.fps} "
        f"{args.codec} -> port {port}",
        flush=True,
    )
    s = stats[serial] = {"ts": collections.deque(maxlen=301), "errors": 0}
    while not stop.is_set():
        if cam.grab() == sl.ERROR_CODE.SUCCESS:
            s["ts"].append(cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
        else:
            s["errors"] += 1
    cam.disable_streaming()
    cam.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--camera",
        action="append",
        help="SERIAL:PORT, even port (default: the rig's three)",
    )
    ap.add_argument("--resolution", default="SVGA")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--codec", choices=("H264", "H265"), default="H265")
    # SVGA gains little above ~20 Mbit/s; recordings keep this bitrate.
    ap.add_argument(
        "--bitrate", type=int, default=20000, help="kbit/s (SDK: 1000-60000)"
    )
    args = ap.parse_args()

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGHUP, lambda *_: stop.set())
    stats: dict = {}
    threads = []
    for entry in args.camera or CAMERAS:
        serial, port = (int(v) for v in entry.split(":"))
        threads.append(
            threading.Thread(target=stream, args=(serial, port, args, stop, stats))
        )
        threads[-1].start()
    try:
        while not stop.wait(10.0):
            line = []
            for serial, s in list(stats.items()):
                ts = list(s["ts"])
                fps = (len(ts) - 1) / ((ts[-1] - ts[0]) / 1e9) if len(ts) > 1 else 0.0
                line.append(f"{serial} {fps:.1f} fps, {s['errors']} grab errors")
            print(time.strftime("%H:%M:%S"), " | ".join(line), flush=True)
    except KeyboardInterrupt:
        pass
    finally:  # also on a broken pipe after the ssh session is gone
        stop.set()
        for t in threads:
            t.join()


if __name__ == "__main__":
    main()
