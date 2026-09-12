"""Fit an `rd calibrate` capture and write a candidate calibration_results.json (converter layout).

Fits the wrist hand-eye X and the board pose B by reprojection over all wrist views, with the board
scale either fixed by the measured square size or free. The scene camera is B @ inv(T_scene_board).

usage: uv run python scripts/write_calib.py <capture dir> [--square-mm 30.4]
Writes <capture dir>/calibration_results.candidate.json. Install by copying it to
~/.config/raiden/calibration_results.json.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from raiden.calibration.runner import compute_forward_kinematics

WRIST, SCENE = "left_wrist_camera", "scene_camera"

ap = argparse.ArgumentParser()
ap.add_argument("capture")
ap.add_argument("--square-mm", type=float, default=None, help="measured square edge; default: fit the scale")
args = ap.parse_args()
cap = Path(args.capture)
meta = json.loads((cap / "captures.json").read_text())
cc = meta["charuco_config"]
board = cv2.aruco.CharucoBoard(
    (cc["squares_x"], cc["squares_y"]), cc["square_length"], cc["marker_length"],
    cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cc["dictionary"])),
)
det = cv2.aruco.CharucoDetector(board)
OBJ = board.getChessboardCorners()
K = {n: np.array(v["camera_matrix"]) for n, v in meta["intrinsics"].items()}
D = {n: np.array(v["distortion_coeffs"]) for n, v in meta["intrinsics"].items()}


def T_(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.ravel(t)
    return T


def vec(T):
    return np.r_[Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]]


def mat(v):
    return T_(Rotation.from_rotvec(v[:3]).as_matrix(), v[3:6])


def detect(cam, fname):
    c, i, _, _ = det.detectBoard(cv2.cvtColor(cv2.imread(str(cap / fname)), cv2.COLOR_BGR2GRAY))
    return c, i


def pnp(cam, c, i):
    ok, rv, tv = cv2.aruco.estimatePoseCharucoBoard(c, i, board, K[cam], D[cam], None, None)
    return T_(cv2.Rodrigues(rv)[0], tv)


wrist, Q, scene = [], [], []
for p in meta["poses"]:
    c, i = detect(WRIST, p["images"][WRIST])
    if i is not None and len(i) >= 6 and p["joints_l"] is not None:
        wrist.append((c, i))
        Q.append(np.array(p["joints_l"][:6]))
    c, i = detect(SCENE, p["images"][SCENE])
    if i is not None and len(i) >= 6:
        scene.append(pnp(SCENE, c, i))
F = [compute_forward_kinematics(q) for q in Q]

# Initial guess: Tsai at the starting scale.
s0 = args.square_mm / 1000 / cc["square_length"] if args.square_mm else 1.0
W0 = [pnp(WRIST, c, i) for c, i in wrist]
W0 = [T_(w[:3, :3], w[:3, 3] * s0) for w in W0]
R, t = cv2.calibrateHandEye([f[:3, :3] for f in F], [f[:3, 3] for f in F],
                            [w[:3, :3] for w in W0], [w[:3, 3] for w in W0])
X0 = T_(R, t)
B0 = F[0] @ X0 @ W0[0]
fit_scale = args.square_mm is None


def unpack(x):
    return mat(x[:6]), mat(x[6:12]), (np.exp(x[12]) if fit_scale else s0)


def resid(x):
    X, B, s = unpack(x)
    r = []
    for (c, i), f in zip(wrist, F):
        Tcb = np.linalg.inv(f @ X) @ B
        pr, _ = cv2.projectPoints(OBJ[i.ravel()] * s, cv2.Rodrigues(Tcb[:3, :3])[0], Tcb[:3, 3], K[WRIST], D[WRIST])
        r.append((pr.reshape(-1, 2) - c.reshape(-1, 2)).ravel())
    return np.concatenate(r)


x0 = np.r_[vec(X0), vec(B0), [np.log(s0)] if fit_scale else []]
sol = least_squares(resid, x0, loss="huber", f_scale=2.0)
X, B, s = unpack(sol.x)
res = resid(sol.x).reshape(-1, 2)
rms = float(np.sqrt(np.mean(np.sum(res ** 2, 1))))

Ss = [T_(T[:3, :3], T[:3, 3] * s) for T in scene]
Rm = Rotation.concatenate([Rotation.from_matrix(T[:3, :3]) for T in Ss]).mean().as_matrix()
T_scene_board = T_(Rm, np.mean([T[:3, 3] for T in Ss], axis=0))
T_base_scene = B @ np.linalg.inv(T_scene_board)
board_ctr = B @ np.r_[OBJ.mean(0) * s, 1]

print(f"wrist views {len(wrist)}, scene views {len(scene)}; scale {s:.4f} "
      f"(square {1000 * cc['square_length'] * s:.1f} mm, {'fitted' if fit_scale else 'fixed'})")
print(f"wrist reprojection RMS {rms:.2f} px")
print(f"hand-eye t {np.round(X[:3, 3], 4)} (grasp_site frame)")
print(f"scene t {np.round(T_base_scene[:3, 3], 4)}, optical axis {np.degrees(np.arcsin(-T_base_scene[2, 2])):.1f} deg down")
print(f"board centre {np.round(board_ctr[:3], 4)}, {1000 * (board_ctr[2] + 0.0199):.1f} mm above the table")

out = {
    "version": "1.0",
    "timestamp": datetime.now().isoformat(),
    "coordinate_frame": "left_arm_base",
    "source": f"write_calib.py on {cap}",
    "charuco_config": {**cc, "square_length": cc["square_length"] * s, "marker_length": cc["marker_length"] * s},
    "cameras": {
        WRIST: {
            "type": "hand_eye",
            "intrinsics": meta["intrinsics"][WRIST],
            "num_poses_used": len(wrist),
            "hand_eye_calibration": {
                "success": True, "method": "reprojection bundle fit",
                "rotation_matrix": X[:3, :3].tolist(), "translation_vector": X[:3, 3].tolist(),
            },
        },
        SCENE: {
            "type": "scene",
            "intrinsics": meta["intrinsics"][SCENE],
            "num_poses_used": len(scene),
            "extrinsics": {
                "success": True, "reference_frame": "left_arm_base",
                "rotation_matrix": T_base_scene[:3, :3].tolist(),
                "translation_vector": T_base_scene[:3, 3].tolist(),
            },
        },
    },
    "quality_metrics": {"wrist_reprojection_rms_px": rms, "board_scale": float(s)},
}
path = cap / "calibration_results.candidate.json"
path.write_text(json.dumps(out, indent=2))
print(f"wrote {path}")
