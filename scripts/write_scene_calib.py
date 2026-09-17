#!/usr/bin/env python3
"""Refit the scene camera pose from teaser_capture board shots.

``scripts/teaser_capture.py --states 0 --poses N`` solves one ``T_base_cam``
(camera-to-left-arm-base, OpenCV axes) per board shot and writes them to
``<out>/poses.json``.  A single shot is good to about 1 cm / 1 deg; this
script averages the shots, reports the spread and the change relative to
the calibration file, and with ``--write`` patches
``cameras.scene_camera.extrinsics`` in ``~/.config/raiden/calibration_results.json``
(backup kept next to it).  The hand-eye and intrinsics are left untouched.

    uv run python scripts/write_scene_calib.py data/teaser/recal_20260915
    uv run python scripts/write_scene_calib.py data/teaser/recal_20260915 --write

Remember: the same pose lives in ``data/real2sim/calibration/rig.json`` and
the MESA rig copy.  Update those separately if the pose changes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from raiden._config import CALIBRATION_FILE

# SNAP canonical-view pivot in the robot base frame (paper/0.md, teaser_capture.py).
PIVOT = np.array([0.29839303, -0.06934502, 0.03011114])


def load_shots(session: Path, only: list[str] | None) -> dict[str, np.ndarray]:
    poses = json.loads((session / "poses.json").read_text())
    if "left_arm_base" not in poses.get("frame", ""):
        sys.exit(f"{session}/poses.json frame is {poses.get('frame')!r}, expected left_arm_base")
    shots = {}
    for name, p in poses["poses"].items():
        if only and name not in only:
            continue
        if "T_base_cam" not in p:
            continue
        shots[name] = np.asarray(p["T_base_cam"], dtype=np.float64)
    if not shots:
        sys.exit("no board shots found")
    return shots


def average(shots: dict[str, np.ndarray]) -> np.ndarray:
    Ts = list(shots.values())
    R = Rotation.from_matrix([T[:3, :3] for T in Ts]).mean().as_matrix()
    t = np.mean([T[:3, 3] for T in Ts], axis=0)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def delta(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Rotation (deg) and translation (mm) between two camera poses."""
    dR = Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude()
    return float(np.degrees(dR)), float(np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1e3)


def pivot_pixel(T_base_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    p_cam = np.linalg.inv(T_base_cam) @ np.append(PIVOT, 1.0)
    uv = K @ p_cam[:3]
    return uv[:2] / uv[2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path, help="teaser_capture output dir with poses.json")
    ap.add_argument("--poses", nargs="*", help="subset of pose names to average, e.g. pose0 pose2")
    ap.add_argument("--calib", type=Path, default=CALIBRATION_FILE)
    ap.add_argument("--write", action="store_true", help="patch the calibration file (backup kept)")
    args = ap.parse_args()

    shots = load_shots(args.session, args.poses)
    T_new = average(shots)

    print(f"{len(shots)} board shots from {args.session}")
    for name, T in shots.items():
        r, t = delta(T_new, T)
        print(f"  {name}: {r:5.2f} deg  {t:5.1f} mm from the mean")

    calib = json.loads(args.calib.read_text())
    cam = calib["cameras"]["scene_camera"]
    ext = cam["extrinsics"]
    if ext.get("reference_frame") != "left_arm_base":
        sys.exit(f"calibration file scene pose is in {ext.get('reference_frame')!r}, refusing")
    T_old = np.eye(4)
    T_old[:3, :3] = np.asarray(ext["rotation_matrix"])
    T_old[:3, 3] = np.asarray(ext["translation_vector"]).reshape(3)

    r, t = delta(T_old, T_new)
    print(f"\nstored pose ({calib.get('timestamp')}): change {r:.2f} deg, {t:.1f} mm")
    intr = cam.get("intrinsics", {})
    if "camera_matrix" in intr:
        K = np.asarray(intr["camera_matrix"], dtype=np.float64)
        old_px, new_px = pivot_pixel(T_old, K), pivot_pixel(T_new, K)
        print(
            f"pivot pixel at 640x480: stored ({old_px[0]:.1f}, {old_px[1]:.1f}) "
            f"-> new ({new_px[0]:.1f}, {new_px[1]:.1f}), shift "
            f"({new_px[0] - old_px[0]:+.1f}, {new_px[1] - old_px[1]:+.1f}) px"
        )
    print("new T_base_cam:\n" + np.array2string(T_new, precision=6, suppress_small=True))

    if not args.write:
        print("\ndry run; pass --write to patch the calibration file")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = args.calib.with_name(args.calib.name + f".bak_{stamp}")
    shutil.copy2(args.calib, backup)
    ext["rotation_matrix"] = T_new[:3, :3].tolist()
    ext["translation_vector"] = T_new[:3, 3].reshape(3, 1).tolist()
    ext["success"] = True
    ext["reference_frame"] = "left_arm_base"
    ext["source"] = f"write_scene_calib.py mean of {sorted(shots)} in {args.session}"
    calib["timestamp"] = datetime.now().isoformat()
    calib["source"] = f"{calib.get('source', '')} | scene pose refit {stamp} from {args.session}"
    args.calib.write_text(json.dumps(calib, indent=2))
    print(f"\nwrote {args.calib} (backup {backup.name})")
    print("rig.json copies still hold the old pose: data/real2sim/calibration/rig.json and the MESA rig.")


if __name__ == "__main__":
    main()
