#!/usr/bin/env python3
"""Render a RAW episode's camera bags to an MP4, without converting it first.

``rd convert`` works from DB status and has no per-episode selector, and
``scripts/rollout_video.py`` reads already-converted frames.  This is for the case
where you want to watch one raw episode right now — e.g. to check a verdict before
fixing it in the DB.

    cd ~/robot/raiden
    uv run python scripts/raw_video.py data/raw/cube_final_real/0157
    uv run python scripts/raw_video.py data/raw/cube_final_real/0157 --stride 2 --scale 0.75
    uv run python scripts/raw_video.py data/raw/cube_final_real/01[5-7]* --out /tmp/review

Writes ``<episode>/preview.mp4`` unless --out names a directory.

This is a preview, not the policy's view: cameras are paired by frame index rather
than by the timestamp grid ``rd convert`` builds, so the two panels can sit a frame
apart. Use rollout_video.py on processed data when exact alignment matters.
"""
import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np

from raiden.cameras.realsense import RealSenseCamera

# Scene cameras read better on the left; wrist views go after.
def _order(bags: list) -> list:
    return sorted(bags, key=lambda p: ("wrist" in p.stem, p.stem))


def _label(img, lines, scale):
    y = int(22 * scale)
    for text in lines:
        cv2.putText(img, text, (int(8 * scale), y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (int(8 * scale), y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * scale, (255, 255, 255), 1, cv2.LINE_AA)
        y += int(24 * scale)


def render(ep: Path, out_path: Path, stride: int, scale: float,
           play: bool = False, save: bool = True) -> bool:
    bags = _order(list((ep / "cameras").glob("*.bag"))) if (ep / "cameras").is_dir() else []
    if not bags:
        print(f"  {ep}: no camera bags")
        return False, False

    meta = json.loads((ep / "metadata.json").read_text()) if (ep / "metadata.json").exists() else {}
    fps = max(1, int(meta.get("camera_fps", 30) / stride))

    # Gripper trace, sampled by elapsed time.  The camera and robot clocks differ
    # (metadata carries realsense_clock_offsets), so this is approximate and the
    # overlay marks it with a tilde.
    grip = None
    npz = ep / "robot_data.npz"
    if npz.exists():
        d = np.load(npz, allow_pickle=True)
        if "follower_l_joint_pos_7d" in d:
            grip = np.asarray(d["follower_l_joint_pos_7d"], dtype=float)[:, 6]

    cams = [RealSenseCamera.from_bag(p.stem, p) for p in bags]
    container = av.open(str(out_path), mode="w") if save else None
    win = f"{ep.parent.name}/{ep.name}   [space] pause   [q] next   [esc] quit"
    if play:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    stream = None
    quit_all = False
    n = written = 0
    try:
        while True:
            if not all(c.grab() for c in cams):
                break
            if n % stride:
                n += 1
                continue
            panels = []
            for cam in cams:
                img = cam.get_frame().color
                if scale != 1.0:
                    img = cv2.resize(img, None, fx=scale, fy=scale)
                lines = [cam._name]
                if len(cams) == 1 or cam is cams[0]:
                    t = n / max(1, meta.get("camera_fps", 30))
                    lines.append(f"frame {n}   t={t:5.2f}s")
                    if grip is not None and len(grip):
                        lines.append(f"~gripper {grip[min(int(t / max(meta.get('duration_s', 1), 1e-9) * len(grip)), len(grip) - 1)]:.3f}")
                _label(img, lines, scale)
                panels.append(img)
            h = max(p.shape[0] for p in panels)
            panels = [cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 0, cv2.BORDER_CONSTANT) for p in panels]
            frame = np.hstack(panels)
            if container is not None:
                if stream is None:
                    stream = container.add_stream("libx264", rate=fps)
                    stream.width, stream.height = frame.shape[1], frame.shape[0]
                    stream.pix_fmt = "yuv420p"
                for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="bgr24")):
                    container.mux(packet)
            if play:
                cv2.imshow(win, frame)
                key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
                if key == ord(" "):                  # pause until any key
                    key = cv2.waitKey(0) & 0xFF
                if key == 27:                        # esc: stop the whole run
                    quit_all = True
                    break
                if key == ord("q"):                  # q: skip to the next episode
                    break
            written += 1
            n += 1
    finally:
        if stream is not None:
            for packet in stream.encode():
                container.mux(packet)
        if container is not None:
            container.close()
        if play:
            cv2.destroyWindow(win)
        for cam in cams:
            cam.close()

    if not written:
        print(f"  {ep}: no frames decoded")
        return False, quit_all
    if save:
        print(f"  {out_path}   {written} frames @ {fps} fps   "
              f"{out_path.stat().st_size / 1e6:.1f} MB")
    else:
        print(f"  {ep}   played {written} frames @ {fps} fps")
    return True, quit_all


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episodes", nargs="+", type=Path)
    ap.add_argument("--play", action="store_true",
                    help="show it on screen instead of writing a file "
                         "(space pauses, q skips to the next episode, esc quits)")
    ap.add_argument("--out", type=Path, help="directory for the MP4s (default: alongside the episode)")
    ap.add_argument("--stride", type=int, default=1, help="keep every N-th frame (default 1)")
    ap.add_argument("--scale", type=float, default=1.0, help="resize factor (default 1.0)")
    args = ap.parse_args()

    # --play on its own just watches; ask for --out as well to also keep the MP4.
    save = bool(args.out) or not args.play
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    ok = True
    for ep in args.episodes:
        if not ep.is_dir():
            print(f"not a directory: {ep}")
            ok = False
            continue
        dest = (args.out / f"{ep.parent.name}_{ep.name}.mp4") if args.out else (ep / "preview.mp4")
        good, quit_all = render(ep, dest, args.stride, args.scale, play=args.play, save=save)
        ok &= good
        if quit_all:
            break
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
