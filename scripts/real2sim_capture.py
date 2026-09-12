#!/usr/bin/env python3
"""Guided multi-view capture of the rig with the (hand-held / re-mounted) scene camera.

Walks through a list of views. For each one it prints where to put the camera and shows a live
preview window; press ENTER in that window to grab ~1 s of frames and stores the colour image plus the per-pixel *median* depth
(aligned to colour) and intrinsics, in the same layout as ``data/real2sim/captures/home``::

    data/real2sim/captures/<view>/scene_camera_color.png
    data/real2sim/captures/<view>/scene_camera_depth_m.npy
    data/real2sim/captures/<view>/meta.json

The arm must stay at the home pose (joints are copied from the home capture's meta.json).

Usage::

    uv run python scripts/real2sim_capture.py                 # 3 default views, ENTER to capture each
    uv run python scripts/real2sim_capture.py --views current  # just grab the current view

Run it with the ``!`` prefix in Claude Code so the prompts show up in your terminal (needs DISPLAY).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from raiden._config import CAMERA_CONFIG
from raiden.camera_config import CameraConfig
from raiden.cameras.realsense import RealSenseCamera, rs

VIEWS = {
    "current": "leave the camera where it is (baseline)",
    "rail_topdown": "TOP-DOWN over the arm base: camera ~50-70 cm straight above the base foot, looking\n"
    "    down, so the base foot, the extrusion rail and ~30 cm of rail on both sides are in frame",
    "scene_side": "SECOND SCENE VIEW: ~90 deg around from the original camera spot (e.g. from the side of\n"
    "    the table, camera ~60 cm high, ~1 m away), whole arm and some table in frame",
    "rail_side": "LOW SIDE VIEW of the rail: camera a few cm above the table top, ~50 cm from the base,\n"
    "    looking along the table so the rail, base foot and table edge are seen edge-on\n"
    "    (put a ruler next to the rail if you have one)",
}
DEFAULT_VIEWS = ["rail_topdown", "scene_side", "rail_side"]
WINDOW = "real2sim capture  (ENTER capture, q skip, ESC quit)"


def capture_median(cam: RealSenseCamera, n_frames: int = 30):
    colors, depths = [], []
    while len(depths) < n_frames:
        if not cam.grab():
            continue
        f = cam.get_frame()
        colors.append(f.color)
        depths.append(f.depth)
    depth = np.stack(depths)
    depth[depth <= 0] = np.nan
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", RuntimeWarning
            )  # all-NaN pixels (no depth) -> 0
            med = np.nanmedian(depth, axis=0)
    med[np.isnan(med)] = 0.0
    return colors[-1], med.astype(np.float32)


def _show(screen, font, color_bgr, banner: str):
    import pygame

    rgb = np.ascontiguousarray(color_bgr[:, :, ::-1])
    surf = pygame.image.frombuffer(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB")
    screen.blit(surf, (0, 0))
    pygame.draw.rect(screen, (0, 0, 0), (0, 0, rgb.shape[1], 30))
    screen.blit(font.render(banner, True, (0, 255, 0)), (8, 6))
    pygame.display.flip()


def wait_for_key(cam: RealSenseCamera, screen, font, banner: str) -> str:
    """Live preview until ENTER ('capture'), q ('skip') or ESC ('quit') is pressed in the window."""
    import pygame

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return "quit"
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                    return "capture"
                if ev.key == pygame.K_q:
                    return "skip"
                if ev.key == pygame.K_ESCAPE:
                    return "quit"
        if not cam.grab():
            continue
        _show(screen, font, cam.get_frame().color, banner)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--views",
        nargs="+",
        default=DEFAULT_VIEWS,
        choices=list(VIEWS),
        help="views to capture, in order",
    )
    ap.add_argument("--camera", default="scene_camera")
    ap.add_argument("--resolution", type=int, nargs=2, default=(848, 480))
    ap.add_argument("--out", default="data/real2sim/captures")
    ap.add_argument(
        "--home",
        default="data/real2sim/captures/home",
        help="home capture (joints are copied from it)",
    )
    args = ap.parse_args()

    home_meta = json.load(open(Path(args.home) / "meta.json"))
    cfg = CameraConfig(CAMERA_CONFIG)
    serial = cfg.cameras[args.camera]["serial"]
    cam = RealSenseCamera(args.camera, serial, resolution=tuple(args.resolution))
    cam.open()
    cam._align = rs.align(
        rs.stream.color
    )  # depth in colour pixels, like the home capture
    K, dist, (w, h) = cam.get_intrinsics()
    # warm up (auto exposure)
    for _ in range(15):
        cam.grab()
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption(WINDOW)
    font = pygame.font.SysFont(None, 24)

    try:
        for i, view in enumerate(args.views, 1):
            print("\n" + "=" * 78)
            print(f"[{i}/{len(args.views)}] {view}: {VIEWS[view]}")
            print(
                "    a preview window is open: position the camera, then press ENTER in it to capture"
            )
            print("    (q = skip this view, ESC = quit)")
            print("=" * 78)
            key = wait_for_key(
                cam,
                screen,
                font,
                f"[{i}/{len(args.views)}] {view}  -  ENTER: capture   q: skip   ESC: quit",
            )
            if key == "quit":
                break
            if key == "skip":
                print("    skipped")
                continue
            print("    capturing (hold still) ...", flush=True)
            color, depth = capture_median(cam)
            out = Path(args.out) / view
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / f"{args.camera}_color.png"), color)
            np.save(out / f"{args.camera}_depth_m.npy", depth)
            meta = {
                args.camera: {
                    "serial": serial,
                    "width": int(w),
                    "height": int(h),
                    "fx": float(K[0, 0]),
                    "fy": float(K[1, 1]),
                    "cx": float(K[0, 2]),
                    "cy": float(K[1, 2]),
                    "coeffs": [float(v) for v in dist],
                    "model": "distortion.inverse_brown_conrady",
                    "depth_scale": float(cam._depth_scale),
                },
                "view": view,
                "view_description": VIEWS[view],
                "robot_joints_measured": home_meta.get("robot_joints_measured"),
                "gripper_measured_normalized": home_meta.get(
                    "gripper_measured_normalized"
                ),
                "note": "arm left at raiden home; joints copied from the home capture; scene camera moved by hand",
            }
            json.dump(meta, open(out / "meta.json", "w"), indent=1)
            valid = (depth > 0).mean()
            print(f"    saved {out}  (valid depth {valid * 100:.0f}%)")
            _show(screen, font, color, f"saved {view}")
            time.sleep(0.8)
        print("\nall views captured")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        cam.close()
        pygame.quit()


if __name__ == "__main__":
    sys.exit(main())
