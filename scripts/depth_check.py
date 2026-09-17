#!/usr/bin/env python3
"""Record a few seconds from every configured camera and check the depth it saves.

Depth recording is a per-camera switch in ``~/.config/raiden/camera.json``
(``"depth": true``), and nothing in a recording session reports whether it is on:
a color-only bag converts without error and simply leaves ``depth/`` empty.  Run
this before a collection that needs depth.  No robot is involved.

    cd ~/robot/raiden
    uv run python scripts/depth_check.py
    uv run python scripts/depth_check.py --seconds 5 --keep

It records bags exactly the way ``rd record`` does (same config, same crop, same
parallel start), extracts them with the converter's own bag reader, and then checks
what landed on disk:

- one ``depth/<camera>/*.npz`` per RGB frame, no holes,
- ``uint16`` in **millimetres** (the device depth scale is read from the bag and
  reported, so a non-default scale shows up here rather than silently rescaling),
- depth aligned to the color grid and cropped with it, so both are the same shape.

It also prints the bag write rate, which is the reason depth gets turned off.
"""
import argparse
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np

from raiden._config import CAMERA_CONFIG
from raiden.camera_config import CameraConfig
from raiden.converter import _extract_bag
from raiden.recorder import load_cameras_from_config

OUT = Path("/var/tmp/raiden_depth_check")


def record(cameras, cameras_dir: Path, seconds: float) -> float:
    """Record all cameras in parallel for *seconds*; return the elapsed time."""
    cameras_dir.mkdir(parents=True, exist_ok=True)

    starts = [
        threading.Thread(
            target=cam.start_recording,
            args=(cameras_dir / f"{cam.name}.{cam.recording_extension}",),
            daemon=True,
        )
        for cam in cameras
    ]
    for t in starts:
        t.start()
    for t in starts:
        t.join()

    stop = threading.Event()

    def pump(cam) -> None:
        while not stop.is_set():
            cam.grab()

    t0 = time.monotonic()
    pumps = [threading.Thread(target=pump, args=(cam,), daemon=True) for cam in cameras]
    for t in pumps:
        t.start()
    time.sleep(seconds)
    stop.set()
    for t in pumps:
        t.join()
    elapsed = time.monotonic() - t0

    for cam in cameras:
        cam.stop_recording()
    return elapsed


def check(name: str, bag: Path, out: Path, elapsed: float) -> bool:
    """Extract one bag and report on its depth.  True if depth is usable."""
    rgb_dir, depth_dir = out / "rgb" / name, out / "depth" / name
    ts, info = _extract_bag(bag, rgb_dir, depth_dir)

    mb = bag.stat().st_size / 1e6
    print(f"\n  {name}")
    print(f"    bag        : {mb:.0f} MB, {mb / elapsed:.0f} MB/s over {elapsed:.1f}s")

    rgbs = sorted(rgb_dir.glob("*.png"))
    npzs = sorted(depth_dir.glob("*.npz")) if depth_dir.exists() else []
    print(f"    frames     : {len(rgbs)} rgb, {len(npzs)} depth ({len(ts)} timestamps)")

    if not npzs:
        print(f"    DEPTH OFF  : the bag holds no depth stream. Set "
              f'"depth": true for {name} in {CAMERA_CONFIG}')
        return False

    import cv2

    color = cv2.imread(str(rgbs[0]))
    d = np.load(npzs[0])["depth"]
    ok = True

    if len(npzs) != len(rgbs):
        print(f"    HOLES      : {len(rgbs) - len(npzs)} frames have no depth")
        ok = False
    if d.dtype != np.uint16:
        print(f"    NOT U16    : depth dtype is {d.dtype}")
        ok = False
    if d.shape != color.shape[:2]:
        print(f"    UNALIGNED  : depth {d.shape} vs rgb {color.shape[:2]}")
        ok = False

    valid = d[d > 0]
    frac = valid.size / d.size
    print(f"    depth      : {d.dtype} {d.shape}, {frac:.0%} valid")
    if valid.size:
        pct = np.percentile(valid, [1, 50, 99]).astype(int)
        print(f"    metric mm  : p1 {pct[0]}  median {pct[1]}  p99 {pct[2]}")
        if not 50 <= pct[1] <= 10000:
            print("    SUSPECT    : median is outside 0.05-10 m; check the units")
            ok = False
    else:
        print("    NO RETURN  : every pixel is 0 — lens covered, or too close")
        ok = False
    if frac < 0.5:
        print(f"    SPARSE     : only {frac:.0%} of pixels have a return; a wrist "
              "D435 below its ~0.2 m minimum range reads mostly zero")

    # The first frame or two after a pipeline start can come back saturated
    # (every pixel 65535 mm), which survives conversion as 65.5 m of "depth".
    bad = [p.stem for p in npzs
           if (np.load(p)["depth"] == 65535).mean() > 0.5]
    if bad:
        print(f"    SATURATED  : {len(bad)} frame(s) are mostly 65535 mm "
              f"(first {bad[0]}) — drop them or re-record")
        ok = False
    if info:
        print(f"    intrinsics : fx {info['fx']:.1f} cx {info['cx']:.1f} "
              f"{info['width']}x{info['height']}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=3.0, help="record duration")
    ap.add_argument("--keep", action="store_true",
                    help=f"leave the bags and frames in {OUT}")
    args = ap.parse_args()

    cfg = CameraConfig(CAMERA_CONFIG)
    print(f"Camera config: {CAMERA_CONFIG}")
    for name in cfg.list_camera_names():
        entry = cfg.list_cameras()[name]
        want = entry.get("depth", True) if isinstance(entry, dict) else True
        print(f"  {name:<20} {cfg.get_camera_type(name):<10} depth={want}")

    if OUT.exists():
        shutil.rmtree(OUT)
    cameras = load_cameras_from_config(CAMERA_CONFIG)
    try:
        elapsed = record(cameras, OUT / "cameras", args.seconds)
    finally:
        for cam in cameras:
            cam.close()

    ok = True
    for cam in cameras:
        bag = OUT / "cameras" / f"{cam.name}.{cam.recording_extension}"
        if bag.suffix != ".bag":
            print(f"\n  {cam.name}: not a RealSense bag, skipping")
            continue
        ok &= check(cam.name, bag, OUT, elapsed)

    total = sum(p.stat().st_size for p in (OUT / "cameras").glob("*")) / 1e6
    print(f"\n  all cameras: {total / elapsed:.0f} MB/s of raw bag")
    print("  → 10 demos x 40 s ≈ "
          f"{total / elapsed * 40 * 10 / 1000:.0f} GB of bags")

    if args.keep:
        print(f"\nKept {OUT}")
    else:
        shutil.rmtree(OUT)

    print("\n" + ("Depth is being saved as metric uint16." if ok
                  else "Depth is NOT usable — see above."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
