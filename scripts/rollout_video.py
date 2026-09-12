#!/usr/bin/env python
"""Render converted episodes as side-by-side MP4s (what the policy sees).

Every camera in the episode is placed next to the others, scene camera(s)
first and wrist camera(s) after, with the frame index and the
measured / commanded gripper value overlaid.  Frames come straight from the
converted ``rgb/<camera>/*.png`` files, so the video shows exactly the
aligned frames that ``rd shardify`` / ``rd lerobot`` consume.

Usage::

    uv run python scripts/rollout_video.py data/processed/cube_stack/0003
    uv run python scripts/rollout_video.py data/processed/cube_stack   # all episodes
    uv run python scripts/rollout_video.py data/processed/cube_stack --out videos/

Output: ``<episode>/rollout.mp4`` (or ``<out>/<task>_<episode>.mp4`` with --out).
"""

import argparse
import json
import pickle
from pathlib import Path

import av
import cv2
import numpy as np
from tqdm import tqdm


def render_episode(ep_dir: Path, out_path: Path, scale: float = 1.0) -> None:
    meta = json.load(open(ep_dir / "metadata.json"))
    # scene camera(s) first, then wrist camera(s)
    cameras = sorted(meta["cameras"], key=lambda c: (0 if "scene" in c else 1, c))
    n = int(meta["num_frames"])
    fps = int(meta.get("framerate", 30))

    first = None
    container = None
    stream = None
    for i in tqdm(range(n), desc=ep_dir.name, leave=False):
        tiles = []
        for cam in cameras:
            img = cv2.imread(str(ep_dir / "rgb" / cam / f"{i:010d}.png"))
            if img is None:
                raise FileNotFoundError(f"{ep_dir}/rgb/{cam}/{i:010d}.png")
            if scale != 1.0:
                img = cv2.resize(
                    img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                )
            cv2.putText(img, cam, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(
                img, cam, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1
            )
            tiles.append(img)
        h = max(t.shape[0] for t in tiles)
        tiles = [
            cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT)
            for t in tiles
        ]
        frame = np.concatenate(tiles, axis=1)

        with open(ep_dir / "lowdim" / f"{i:010d}.pkl", "rb") as f:
            fd = pickle.load(f)
        grip = float(fd["joints"][6]) if "joints" in fd else float("nan")
        grip_cmd = (
            float(fd["action_joints"][6]) if "action_joints" in fd else float("nan")
        )
        text = f"frame {i:4d}/{n}  t={i / fps:5.2f}s  gripper {grip:.2f} (cmd {grip_cmd:.2f})"
        cv2.putText(
            frame,
            text,
            (8, frame.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            frame,
            text,
            (8, frame.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        # yuv420p needs even dimensions
        frame = frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]
        if container is None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            container = av.open(str(out_path), mode="w")
            stream = container.add_stream("libx264", rate=fps)
            stream.width, stream.height = frame.shape[1], frame.shape[0]
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "20", "preset": "fast"}
            first = frame.shape
        if frame.shape != first:
            raise ValueError(f"frame {i} has shape {frame.shape}, expected {first}")
        for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="bgr24")):
            container.mux(packet)

    if container is not None:
        for packet in stream.encode():
            container.mux(packet)
        container.close()
    print(f"✓ {out_path}  ({n} frames @ {fps} fps, {cameras})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "path", type=Path, help="converted episode dir, or task dir for all episodes"
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: inside each episode)",
    )
    ap.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="resize factor per camera (default: 1.0)",
    )
    args = ap.parse_args()

    if (args.path / "metadata.json").exists() and (args.path / "rgb").exists():
        episodes = [args.path]
    else:
        episodes = sorted(
            d for d in args.path.iterdir() if d.is_dir() and (d / "rgb").exists()
        )
    if not episodes:
        raise SystemExit(f"no converted episodes under {args.path}")

    for ep in episodes:
        if args.out is not None:
            out = args.out / f"{ep.parent.name}_{ep.name}.mp4"
        else:
            out = ep / "rollout.mp4"
        render_episode(ep, out, scale=args.scale)


if __name__ == "__main__":
    main()
