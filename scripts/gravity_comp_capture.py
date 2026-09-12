#!/usr/bin/env python3
"""Hold the arm in gravity compensation and grab wrist/scene shots on demand.

The follower arm is initialised with zero stiffness, so it can be pushed around
by hand while the motors carry its own weight.  The cameras stay open and a
shot is written whenever a trigger file appears, which lets an assistant in
another terminal take the picture while the operator holds the pose::

    echo "oven_front_closed" > <out>/TRIGGER      # label is optional

Each shot writes ``<out>/<NNNN>_<label>/`` with one PNG per camera plus
``meta.json``: measured joints, FK of ``grasp_site`` in the arm base frame, the
wrist camera pose in the arm base frame (FK x hand-eye from
``calibration_results.json``) and the on-device intrinsics.

Usage::

    uv run python scripts/gravity_comp_capture.py --out data/real2sim/oven
    uv run python scripts/gravity_comp_capture.py --cameras left_wrist_camera

The arm goes limp the moment gravity compensation engages: hold it before
starting.  Ctrl-C restores position control at the pose the arm is in.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from raiden._config import CALIBRATION_FILE, CAMERA_CONFIG
from raiden.calibration.runner import compute_forward_kinematics
from raiden.camera_config import CameraConfig
from raiden.robot.controller import RobotController

TRIGGER = "TRIGGER"


def load_hand_eye(camera: str) -> np.ndarray | None:
    """4x4 camera pose in the ``grasp_site`` frame, or None if uncalibrated."""
    path = Path(CALIBRATION_FILE)
    if not path.exists():
        return None
    entry = json.loads(path.read_text()).get("cameras", {}).get(camera, {})
    hand_eye = entry.get("hand_eye_calibration")
    if not hand_eye or not hand_eye.get("success"):
        return None
    T = np.eye(4)
    T[:3, :3] = np.array(hand_eye["rotation_matrix"])
    T[:3, 3] = np.array(hand_eye["translation_vector"])
    return T


def open_cameras(names: list[str]) -> dict:
    cfg = CameraConfig(CAMERA_CONFIG)
    cameras = {}
    for name in names:
        cam = cfg.create_camera(name)
        cam.open()
        cameras[name] = cam
        print(f"  - {name} open")
    return cameras


def grab(cam, n_warmup: int = 5):
    """Return the newest colour frame, after flushing the queue."""
    for _ in range(n_warmup):
        cam.grab()
    cam.grab()
    return cam.get_frame().color


def save_shot(
    out: Path, index: int, label: str, cameras: dict, joints: np.ndarray, hand_eye: dict
) -> Path:
    name = f"{index:04d}" + (f"_{label}" if label else "")
    shot = out / name
    shot.mkdir(parents=True, exist_ok=True)

    T_base_grasp = compute_forward_kinematics(joints[:6], arm="left")
    meta = {
        "timestamp": time.time(),
        "label": label,
        "joints": joints.tolist(),
        "ee_pose_grasp_site": T_base_grasp.tolist(),
        "cameras": {},
    }
    for cam_name, cam in cameras.items():
        cv2.imwrite(str(shot / f"{cam_name}.png"), grab(cam))
        K, dist, size = cam.get_intrinsics()
        entry = {
            "intrinsics": K.tolist(),
            "distortion": np.asarray(dist).tolist(),
            "image_size": list(size),
        }
        if cam_name in hand_eye:
            entry["pose_in_base"] = (T_base_grasp @ hand_eye[cam_name]).tolist()
        meta["cameras"][cam_name] = entry

    (shot / "meta.json").write_text(json.dumps(meta, indent=2))
    return shot


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out",
        default="data/real2sim/captures/gravity_comp",
        help="output directory (also holds the TRIGGER file)",
    )
    ap.add_argument(
        "--cameras",
        nargs="*",
        default=["left_wrist_camera", "scene_camera"],
        help="camera names from camera.json",
    )
    ap.add_argument("--poll-hz", type=float, default=20.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trigger = out / TRIGGER
    trigger.unlink(missing_ok=True)

    print("Opening cameras...")
    cameras = open_cameras(args.cameras)
    hand_eye = {}
    for name in cameras:
        T = load_hand_eye(name)
        if T is not None:
            hand_eye[name] = T
            print(f"  - {name} hand-eye loaded")

    controller = RobotController(
        use_right_leader=False,
        use_left_leader=False,
        use_right_follower=False,
        use_left_follower=True,
    )
    controller.check_can_interfaces()
    print("\n*** Support the arm now: it goes free when gravity comp engages. ***")
    controller.initialize_robots(gravity_comp_mode=False)
    controller.enable_gravity_compensation()
    # i2rt's update_kp_kd only stores the gains; they reach the motors on the next
    # command_joint_pos().  Without this the arm keeps holding with full stiffness.
    controller.follower_l.command_joint_pos(controller.follower_l.get_joint_pos())

    index = len([p for p in out.iterdir() if p.is_dir()])
    print(f"\nReady.  Move the arm by hand.  Trigger a shot with:")
    print(f"    echo <label> > {trigger}")
    print("Ctrl-C to restore position control.\n")

    stop = False

    def on_signal(_sig, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    period = 1.0 / args.poll_hz
    last_print = 0.0
    try:
        while not stop:
            joints = controller.get_joint_positions()["follower_l"]
            if trigger.exists():
                label = trigger.read_text().strip().replace(" ", "_")
                trigger.unlink(missing_ok=True)
                shot = save_shot(out, index, label, cameras, joints, hand_eye)
                print(
                    f"[shot {index:04d}] {shot}  joints="
                    f"{np.array2string(joints, precision=3, suppress_small=True)}",
                    flush=True,
                )
                index += 1
            now = time.time()
            if now - last_print > 5.0:
                print(
                    f"  joints {np.array2string(joints, precision=3, suppress_small=True)}",
                    flush=True,
                )
                last_print = now
            time.sleep(period)
    finally:
        print("\nRestoring position control at the current pose...")
        hold = controller.get_joint_positions()["follower_l"]
        controller.disable_gravity_compensation()
        # The restored gains take effect on this command, which holds the arm
        # where it is rather than at the stale initialisation target.
        for _ in range(100):
            controller.follower_l.command_joint_pos(hold)
            time.sleep(0.01)
        for cam in cameras.values():
            cam.close()
        controller.close()
        print("Done.")


if __name__ == "__main__":
    sys.exit(main())
