#!/usr/bin/env python
"""Measure the white balance, exposure and gain that lock each RealSense camera.

On auto exposure and auto white balance every pipeline start opens dark and
green for about a second, and the colours drift between sessions.  This opens
each RealSense camera in ``~/.config/raiden/camera.json`` on auto, lets it
settle, then reopens it locked, as a recording starts, and finds the manual
values that come closest to the settled auto image in the policy's centre
square crop.  They go to the camera's ``color_controls``; ``rd record``,
``rd serve`` and ``camera_preview.py`` apply them on every pipeline start.

Set up the scene and turn the rig lights on first: the values are only right
for this light.  About 30 s per camera; the robot is not used.

    uv run python scripts/lock_color.py
    uv run python scripts/lock_color.py --cameras scene_camera
    uv run python scripts/lock_color.py --dry-run     # measure and print only

The cameras stay locked when it exits.  Set ``color_controls`` to ``"auto"``
(or delete it) to go back to auto.  The D435 reports no frame metadata, and
its option readback is not live under auto, so the auto values cannot simply
be read and frozen: white balance and gain are bisected against the image.
"""

import argparse
import json
import shutil
import time
from datetime import datetime

import numpy as np
import pyrealsense2 as rs

from raiden._config import CAMERA_CONFIG
from raiden.camera_config import CameraConfig

# 100 µs units: whole periods of 60 Hz lighting's 120 Hz flicker, at most one
# frame at 30 fps.  Tried in this order.
FLICKER_SAFE_EXPOSURES = (250, 333, 167, 83)
WB_RANGE = (2800, 6500)
GAIN_RANGE = (0, 128)


def measure(cam, settle: int = 12, n: int = 3) -> tuple[float, float]:
    """Mean brightness and blue/red ratio in the centre square crop, after *settle* frames."""
    for _ in range(settle):
        cam.grab()
    values = []
    for _ in range(n):
        cam.grab()
        img = cam.get_frame().color
        h, w = img.shape[:2]
        x0 = (w - h) // 2
        c = img[:, x0 : x0 + h].reshape(-1, 3).astype(np.float64).mean(axis=0)
        values.append((c.mean(), c[0] / max(c[2], 1e-3)))
    mean, blue_red = np.mean(values, axis=0)
    return float(mean), float(blue_red)


def bisect(set_value, lo, hi, read, target, increasing: bool, iters: int) -> float:
    for _ in range(iters):
        mid = (lo + hi) / 2
        set_value(mid)
        lo, hi = (mid, hi) if (read() < target) == increasing else (lo, mid)
    return (lo + hi) / 2


def fit(cam, target_mean: float, target_ratio: float) -> dict:
    """White balance, exposure and gain that give *target_mean* and *target_ratio*.

    *cam* must have been opened locked, so that auto white balance never ran in
    this stream: the D435 keeps part of the auto state when it is turned off
    mid-stream, and the same white balance then gives other colours than it
    does locked from the first frame, as every recording starts.
    """
    sensor = cam._profile.get_device().first_color_sensor()

    def set_wb(v):
        sensor.set_option(rs.option.white_balance, float(round(v / 10) * 10))

    def set_gain(v):
        sensor.set_option(rs.option.gain, float(round(v)))

    wb = (
        round(
            bisect(set_wb, *WB_RANGE, lambda: measure(cam)[1], target_ratio, False, 9)
            / 10
        )
        * 10
    )
    set_wb(wb)

    exposure, bracketed = FLICKER_SAFE_EXPOSURES[0], False
    for e in FLICKER_SAFE_EXPOSURES:
        sensor.set_option(rs.option.exposure, float(e))
        set_gain(GAIN_RANGE[0])
        darkest = measure(cam)[0]
        set_gain(GAIN_RANGE[1])
        if darkest <= target_mean <= measure(cam)[0]:
            exposure, bracketed = e, True
            break
    sensor.set_option(rs.option.exposure, float(exposure))
    gain = round(
        bisect(set_gain, *GAIN_RANGE, lambda: measure(cam)[0], target_mean, True, 7)
    )
    set_gain(gain)

    warnings = []
    if wb in WB_RANGE:
        warnings.append(
            f"white balance at the {wb} K limit: the colour cannot match auto"
        )
    if not bracketed:
        warnings.append("no flicker-safe exposure brackets the auto brightness")
    return {
        "color_controls": {"white_balance": wb, "exposure": exposure, "gain": gain},
        "warnings": warnings,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--cameras", nargs="*", default=None, help="default: every RealSense camera"
    )
    ap.add_argument(
        "--warmup-s", type=float, default=5.0, help="auto settling time per camera"
    )
    ap.add_argument("--dry-run", action="store_true", help="do not write camera.json")
    ap.add_argument(
        "--report", default="", help="also write the measurements to this JSON file"
    )
    args = ap.parse_args()

    cfg = CameraConfig(CAMERA_CONFIG)
    names = [
        n
        for n in (args.cameras or cfg.list_camera_names())
        if cfg.get_camera_type(n) == "realsense"
    ]
    if not names:
        raise SystemExit("no RealSense cameras in camera.json")

    results = {}
    for name in names:
        cam = cfg.create_camera(name)
        cam._color_controls = None  # the auto image, whatever camera.json holds
        cam.open()
        print(f"  [{name}] settling on auto for {args.warmup_s:.0f} s ...")
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.warmup_s:
            cam.grab()
        target = measure(cam, settle=30, n=10)
        cam.close()

        print(f"  [{name}] matching white balance, exposure and gain ...")
        cam._color_controls = {
            "white_balance": 4600,
            "exposure": FLICKER_SAFE_EXPOSURES[0],
            "gain": 64,
        }
        cam.open()
        r = fit(cam, *target)
        cam.close()

        cam._color_controls = r["color_controls"]  # reopened as a recording starts
        cam.open()
        locked = measure(cam, settle=30, n=10)
        cam.close()
        r["auto"] = {"mean": round(target[0], 1), "blue_red": round(target[1], 3)}
        r["locked"] = {"mean": round(locked[0], 1), "blue_red": round(locked[1], 3)}
        results[name] = r
        print(
            f"  [{name}] {r['color_controls']}  "
            f"auto mean {r['auto']['mean']} b/r {r['auto']['blue_red']}  ->  "
            f"locked mean {r['locked']['mean']} b/r {r['locked']['blue_red']}"
        )
        for w in r["warnings"]:
            print(f"  [{name}] warning: {w}")

    if args.report:
        with open(args.report, "w") as f:
            json.dump(
                {
                    "measured_at": datetime.now().isoformat(timespec="seconds"),
                    **results,
                },
                f,
                indent=2,
            )
    if args.dry_run:
        return
    backup = f"{CAMERA_CONFIG}.bak-{datetime.now():%Y%m%d-%H%M%S}"
    shutil.copy2(CAMERA_CONFIG, backup)
    for name, r in results.items():
        cfg.cameras[name]["color_controls"] = r["color_controls"]
    cfg._save()
    print(f"  ✓ color_controls written to {CAMERA_CONFIG} (previous: {backup})")


if __name__ == "__main__":
    main()
