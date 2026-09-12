#!/usr/bin/env python
"""Live side-by-side preview of the configured cameras (scene | wrist) in Rerun.

Opens every camera in ``~/.config/raiden/camera.json`` exactly as ``rd record``
does (same class, resolution and crop settings) and streams them to a Rerun
viewer, scene camera(s) on the left and wrist camera(s) on the right — the
same layout as the converted rollouts.  Use it to adjust camera placement.

Usage::

    uv run python scripts/camera_preview.py              # spawns the Rerun viewer
    uv run python scripts/camera_preview.py --web        # browser viewer (over SSH)
    uv run python scripts/camera_preview.py --cameras scene_camera
    uv run python scripts/camera_preview.py --grid       # rule-of-thirds overlay
    uv run python scripts/camera_preview.py --sim        # watch the MESA digital twin instead
    uv run python scripts/camera_preview.py --sim --view # + operator view and task panel

Ctrl-C to stop.
"""

import argparse
import time

import cv2

from raiden._config import CAMERA_CONFIG
from raiden.camera_config import CameraConfig
from raiden.monitor import LiveMonitor


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--cameras",
        nargs="*",
        default=None,
        help="camera names from camera.json (default: all)",
    )
    ap.add_argument(
        "--grid", action="store_true", help="draw a rule-of-thirds grid on each image"
    )
    ap.add_argument(
        "--no-guides", action="store_true", help="hide the gripper grasp guides"
    )
    ap.add_argument(
        "--web",
        action="store_true",
        help="serve a browser viewer instead of spawning the app",
    )
    ap.add_argument("--web-port", type=int, default=9090)
    ap.add_argument(
        "--sim",
        nargs="?",
        const="127.0.0.1:5599",
        default="",
        help="preview the MESA digital twin (address of a running raiden_sim_server)",
    )
    ap.add_argument(
        "--view",
        nargs="?",
        const="agentview",
        default="",
        help="sim only: add an operator view (agentview, behindview, frontview, birdview, "
        "sideview) and the BDDL task panel; never recorded",
    )
    args = ap.parse_args()
    if args.view and not args.sim:
        raise SystemExit("--view needs --sim (operator views come from the simulator)")

    cfg = CameraConfig(CAMERA_CONFIG)
    if args.sim:
        from raiden.sim import SimConnection

        conn = SimConnection(args.sim)
        available = list(conn.call("get_rig")["cameras"])
        conn.close()
    else:
        available = list(cfg.cameras.keys())
    names = args.cameras or available
    names = sorted(
        set(names) & set(available), key=lambda c: (0 if "scene" in c else 1, c)
    )
    if not names:
        raise SystemExit(f"no such cameras: {args.cameras}; available: {available}")

    cams = []
    if args.sim:
        from raiden.sim import load_sim_cameras

        by_name = {c.name: c for c in load_sim_cameras(args.sim, CAMERA_CONFIG)}
        cams = [by_name[n] for n in names]
    else:
        for name in names:
            cam = cfg.create_camera(name)
            cam.open()
            cams.append(cam)
            crop = cfg.get_crop(name)
            print(
                f"  ✓ {name} (serial {cam.serial_number})"
                + (f"  crop {crop}" if crop else "")
            )

    if args.view:
        print(f"  ✓ {args.view} (operator view, not recorded)")

    guides = None
    if not args.no_guides:
        from raiden.guides import make_guides

        guides = make_guides(cams, sim=args.sim) or None
        if guides is None:
            print("  grasp guides: no camera pose available, skipping")
        elif not args.sim:
            print("  grasp guides: no arm connection, so nothing to draw (use --sim)")

    monitor = LiveMonitor(
        names,
        sim=args.sim,
        view=args.view,
        web=args.web,
        web_port=args.web_port,
        app_id="camera_preview",
        guides=guides,
    )

    joint_conn = None
    if guides is not None and args.sim:
        from raiden.sim import SimConnection

        joint_conn = SimConnection(args.sim)

    print("Streaming — Ctrl-C to stop")

    t_last, n_frames = time.monotonic(), 0
    try:
        while True:
            for cam in cams:
                if not cam.grab():
                    continue
                img = cam.get_frame().color
                if args.grid:
                    img = img.copy()
                    h, w = img.shape[:2]
                    for k in (1, 2):
                        cv2.line(
                            img, (w * k // 3, 0), (w * k // 3, h), (0, 255, 255), 1
                        )
                        cv2.line(
                            img, (0, h * k // 3), (w, h * k // 3), (0, 255, 255), 1
                        )
                monitor.log(cam.name, img)
            if joint_conn is not None:
                try:
                    monitor.log_joints("follower_l", joint_conn.call("get_joint_pos"))
                except (RuntimeError, EOFError, OSError):
                    pass
            n_frames += 1
            now = time.monotonic()
            if now - t_last >= 5.0:
                print(f"  {n_frames / (now - t_last):.1f} fps", end="\r", flush=True)
                t_last, n_frames = now, 0
    except KeyboardInterrupt:
        pass
    finally:
        for cam in cams:
            cam.close()
        if joint_conn is not None:
            joint_conn.close()
        monitor.close()


if __name__ == "__main__":
    main()
