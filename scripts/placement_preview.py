#!/usr/bin/env python
"""Live scene-camera view of the croissant task's placement limits, to place the croissant and the oven before a
policy run. Needs no robot, Quest, CAN or recording session: only the scene camera, opened the way ``rd record``
opens it (camera.json, colour lock included).

It draws the recording page's own overlay (``RecordUI._scene_overlay`` on ``raiden.qc``): the white, blue and orange
boxes, and the croissant (circle) and oven (outline) where they are now, green inside the limits and red outside, with
their x, y in the base frame. Blue-box cells tinted yellow have no successful demo of ``--task`` yet
(data/qc/<task>/qc_results.json, as on the page's coverage grid).

A RealSense streams to one process at a time: quit this (q) before ``rd serve`` or ``rd record`` opens the cameras.

    uv run python scripts/placement_preview.py
    uv run python scripts/placement_preview.py --snapshot scene.png      # one annotated frame, no window
    uv run python scripts/placement_preview.py --bag data/raw/croissant_oven_real_newrig/0000/cameras/scene_camera.bag

q or Esc quits, s saves a snapshot to data/placement_preview/. The limits are only as good as rig.json's scene camera
pose: if the camera got bumped, check it with ~/robot/vla-benchmark/scripts/raiden_lab/spawn_overlay.py --reference.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from raiden import qc
from raiden._config import CAMERA_CONFIG
from raiden.camera_config import CameraConfig
from raiden.cameras.realsense import RealSenseCamera
from raiden.record_ui import GREEN, RED, WHITE, YELLOW, RecordUI

REPO = Path(__file__).resolve().parents[1]
NAME = "scene_camera"


def coverage(task: str, geo: qc.Geometry):
    """(successes per blue-box cell, successes per oven y bin) from the task's QC results; None if there are none."""
    f = REPO / "data/qc" / task / "qc_results.json"
    if not f.exists():
        return None
    results = json.load(open(f))
    return qc.coverage(results, geo)[0], qc.oven_coverage(results, geo)[0]


def text(v: np.ndarray, s: str, row: int, colour, scale: float = 0.5) -> None:
    org = (8, row)
    cv2.putText(v, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(v, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def annotate(ui: RecordUI, raw: np.ndarray, cov):
    """The page's scene overlay checked on this frame, the cells with no success yet and the readout; (image, live)."""
    geo, v = ui._geo, raw.copy()
    if cov is not None:  # under the outlines
        bx0, bx1, by0, by1 = geo.blue
        dx, dy = (bx1 - bx0) / qc.GRID_X, (by1 - by0) / qc.GRID_Y
        layer = v.copy()
        for i, j in zip(*np.nonzero(cov[0] == 0)):  # row 0 nearest the robot, column 0 the robot's left (+y)
            cell = (bx0 + i * dx, bx0 + (i + 1) * dx, by1 - (j + 1) * dy, by1 - j * dy)
            cv2.fillPoly(layer, [qc.box_outline(cell, geo.table_z, geo, n=2).round().astype(np.int32)], YELLOW)
        cv2.addWeighted(layer, 0.35, v, 0.65, 0, dst=v)
    live = ui._scene_overlay(raw, v, True)
    c, o = live["croissant"], live["oven"]
    if c is None:
        text(v, "croissant not seen", 20, YELLOW)
    else:
        why = [s for ok, s in ((c["in_blue"], "centre outside the blue box"), (c["in_white"], "may cross the white box"))
               if not ok]
        text(v, f"croissant x {c['x']:.3f} y {c['y']:+.3f}: {', '.join(why) or 'OK'}", 20, RED if why else GREEN)
    if o is None:
        text(v, "oven knobs not seen", 40, YELLOW)
    else:
        sim_x = (geo.oven_region[0] + geo.oven_region[1]) / 2
        why = [s for ok, s in ((o["x_ok"], f"x off the sim's {sim_x:.3f}"), (o["y_ok"], "y outside the sim's range"),
                               (o["in_white"], "crosses the white box")) if not ok]
        text(v, f"oven x {o['x']:.3f} y {o['y']:+.3f}: {', '.join(why) or 'OK'}", 40, RED if why else GREEN)
    if cov is not None:
        G, O = cov
        here = ([("croissant cell", G[tuple(c["cell"])])] if c and c["cell"] else []) + (
            [("oven y bin", O[o["cell"]])] if o and o["cell"] is not None else [])
        if here:
            text(v, "successful demos from here: " + ", ".join(f"{k} {m}" for k, m in here), 60,
                 YELLOW if min(m for _, m in here) == 0 else WHITE)
        oy = np.linspace(geo.oven_region[3], geo.oven_region[2], qc.OVEN_GRID + 1)
        empty = [f"{(oy[j] + oy[j + 1]) / 2:+.3f}" for j in range(qc.OVEN_GRID) if O[j] == 0]
        if empty:
            text(v, "oven y with no success yet: " + " ".join(empty), 80, YELLOW)
    text(v, "white: whole objects  blue: croissant centre  orange: oven range  yellow: no success yet",
         v.shape[0] - 10, WHITE, 0.4)
    return v, live


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", default="", help="save one annotated frame to this path and exit (no window)")
    ap.add_argument("--bag", default="", help="replay a recorded scene_camera.bag instead of the camera")
    ap.add_argument("--task", default="croissant_oven_real_newrig", help="coverage of this task's successes; '' for none")
    ap.add_argument("--scale", type=float, default=1.5, help="window size relative to 640x480")
    args = ap.parse_args()

    ui = RecordUI(open_browser=False)  # not started: only its geometry and scene overlay
    cfg = CameraConfig(CAMERA_CONFIG)
    if args.bag:
        cam = RealSenseCamera.from_bag(NAME, Path(args.bag), crop=cfg.get_crop(NAME))
    else:
        cam = cfg.create_camera(NAME)
        try:
            cam.open()
        except RuntimeError as e:
            raise SystemExit(f"the scene camera did not open ({e}). Is rd serve or rd record running?")
    win, out = "placement limits (q quits, s saves)", REPO / "data/placement_preview"
    cov, cov_t, n, t0 = None, 0.0, 0, time.monotonic()
    try:
        while True:
            if not cam.grab():
                if args.bag:
                    break  # end of the bag
                continue
            n += 1
            if args.snapshot and not args.bag and time.monotonic() - t0 < 1.5:
                continue  # a live snapshot waits for the stream to settle
            if args.task and time.monotonic() - cov_t > 2.0:  # follows the recording page's rewrites
                cov, cov_t = coverage(args.task, ui._geo), time.monotonic()
            img, live = annotate(ui, cam.get_frame().color, cov)
            if args.snapshot:
                cv2.imwrite(args.snapshot, img)
                rate = "" if args.bag else f", {n / (time.monotonic() - t0):.0f} fps"
                print(f"  saved {args.snapshot} (frame {n}{rate}): croissant {live['croissant']}, oven {live['oven']}")
                return
            cv2.imshow(win, cv2.resize(img, None, fx=args.scale, fy=args.scale))
            key = cv2.waitKey(33 if args.bag else 1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("s"):
                out.mkdir(parents=True, exist_ok=True)
                p = out / f"{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(str(p), img)
                print(f"  saved {p}")
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
