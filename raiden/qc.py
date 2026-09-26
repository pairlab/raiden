"""Per-episode quality checks for real demos of the raiden_lab croissant task.

Used by ``scripts/qc_episodes.py`` (a watcher) and by ``rd record --ui``. For one finished episode:

* wrist camera, first frame: fingertip rows against the reference and mean brightness. A slipped mount shows as
  tips that moved, or a dark frame looking at the partition.
* scene camera, first frame: brightness; metadata.json's color_controls match camera.json for both cameras.
* pauses: stretches with the arm and gripper still, outside the first and last 0.5 s.
* croissant start (first scene frame, the orange blob projected to its centre-of-mass height): centre of mass in the
  blue box, the whole croissant (51 mm radius) in the white box.
* gripper: close events of the open/closed command; more than ``max_closes`` suggests a regrasp.
* duration: outside ``min_s``/``max_s``, or far from the median of the other episodes.

The placement limits come from the vla-benchmark twin (rig.json, layout.json and the task JSON), so the checks follow
the sim's spawn regions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from raiden.camera_config import CameraConfig
from raiden.cameras.realsense import RealSenseCamera

IN = 0.0254
COM_MARGIN = 0.004  # the task JSON's croissant region is this much inside the centre-of-mass box
CRO_RADIUS = 0.051  # croissant footprint radius about its centre, any yaw
CRO_COM_Z = 0.0215  # centre-of-mass height above the table
OVEN_FOOTPRINT = (-0.1518, 0.1100, -0.1937, 0.1930)  # x, y extents about the oven origin (sim model, handle incl.)
STILL_RAD_S = 0.02
GRID_X, GRID_Y = 3, 6
VLA_DEFAULT = Path.home() / "robot/vla-benchmark"
TASK_JSON = "mesa/task_suites/bddl_files/raiden_lab/croissant_oven_heating_region/source/000.json"


@dataclass
class QCSettings:
    ref_tips: Tuple[int, int] = (312, 304)  # wrist fingertip rows (blue, black), arm at home, 2026-09-26 19:05
    max_tip_px: int = 15
    min_wrist_mean: float = 100.0
    scene_mean: Tuple[float, float] = (90.0, 140.0)
    max_pause: float = 1.0
    max_closes: int = 2
    min_s: float = 5.0
    max_s: float = 90.0


@dataclass
class Geometry:
    white: Tuple[float, float, float, float]  # x0, x1, y0, y1 in the base frame
    blue: Tuple[float, float, float, float]
    orange: Tuple[float, float, float, float]
    table_z: float
    T: np.ndarray = field(repr=False)  # scene camera, camera-to-base, OpenCV axes
    K: np.ndarray = field(repr=False)


def load_geometry(vla: Path = VLA_DEFAULT) -> Geometry:
    rigd = Path(vla) / "mesa/task_suites/rigs/raiden_lab"
    rig, lay = json.load(open(rigd / "rig.json")), json.load(open(rigd / "layout.json"))
    (tx0, tx1), (ty0, ty1) = lay["table"]["atlas"]["x_range"], lay["table"]["atlas"]["y_range"]
    tc = ((tx0 + tx1) / 2, (ty0 + ty1) / 2)
    regions = json.load(open(Path(vla) / TASK_JSON))["regions"]
    x0, y0, x1, y1 = regions["table_object_0_init_region"]["ranges"][0]
    ox0, oy0, ox1, oy1 = regions["table_articulated_object_init_region"]["ranges"][0]
    cam = rig["cameras"]["scene_camera"]
    return Geometry(
        white=(tx0 + 6 * IN, tx1, ty0 + 13 * IN, ty1 - 13 * IN),
        blue=(x0 + tc[0] - COM_MARGIN, x1 + tc[0] + COM_MARGIN, y0 + tc[1] - COM_MARGIN, y1 + tc[1] + COM_MARGIN),
        orange=(ox0 + tc[0] + OVEN_FOOTPRINT[0], ox1 + tc[0] + OVEN_FOOTPRINT[1],
                oy0 + tc[1] + OVEN_FOOTPRINT[2], oy1 + tc[1] + OVEN_FOOTPRINT[3]),
        table_z=float(rig["table"]["z_in_base_at_origin"]), T=np.array(cam["T_base_cam"]), K=np.array(cam["K"]))


def project(P: np.ndarray, T: np.ndarray, K: np.ndarray) -> np.ndarray:
    pc = (P - T[:3, 3]) @ T[:3, :3]
    return (pc @ K.T)[:, :2] / pc[:, 2:]


def box_outline(box, z: float, geo: Geometry, n: int = 60) -> np.ndarray:
    """Pixel polyline of a base-frame box on the plane at height z."""
    x0, x1, y0, y1 = box
    c = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])
    e = np.linspace(0, 1, n)[:, None]
    pts = np.concatenate([a + e * (b - a) for a, b in zip(c[:-1], c[1:])])
    return project(np.c_[pts, np.full(len(pts), z)], geo.T, geo.K)


def first_frame(bag: Path, name: str, crop) -> Optional[np.ndarray]:
    cam = RealSenseCamera.from_bag(name, bag, crop=crop)
    try:
        for _ in range(5):
            if cam.grab():
                return cam.get_frame().color.copy()
    finally:
        cam.close()
    return None


def fingertips(img: np.ndarray) -> List[Optional[Tuple[int, int]]]:
    """[blue-finger tip, black-finger tip] as (row, col), None where not found."""
    b, g, r = [img[..., i].astype(np.int32) for i in range(3)]
    blue = (b - r > 60) & (b - g > 20) & (b > 110)  # the blue finger enters bottom left
    dark = img.max(2) < 45  # the black finger enters bottom right
    tips: List[Optional[Tuple[int, int]]] = []
    for mask, right in ((blue, False), (dark, True)):
        m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        h, w = m.shape
        cands = [k for k in range(1, n) if st[k, 4] > 800 and st[k, 1] + st[k, 3] >= h - 2
                 and ((st[k, 0] + st[k, 2] > w // 2) if right else st[k, 0] < w // 2)]
        if not cands:
            tips.append(None)
            continue
        k = max(cands, key=lambda k: st[k, 4])
        ys, xs = np.nonzero(lab == k)
        tips.append((int(ys.min()), int(xs[ys.argmin()])))
    return tips


def croissant_position(img: np.ndarray, geo: Geometry) -> Optional[Tuple[float, float, int]]:
    """Croissant centre of mass (x, y) in the base frame, and its pixel count, from the largest orange blob over the
    white box; None if there is none."""
    region = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(region, [box_outline(geo.white, geo.table_z, geo, n=2).round().astype(np.int32)], 1)
    region = cv2.dilate(region, np.ones((31, 31), np.uint8))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # The locked croissant is pale orange (S 36-101); the locked table is grey-beige (S <= 9).
    m = ((hsv[..., 0] >= 5) & (hsv[..., 0] <= 25) & (hsv[..., 1] >= 30) & (hsv[..., 2] >= 100)
         & (region > 0)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    if n < 2 or st[1:, 4].max() < 150:
        return None
    k = 1 + int(np.argmax(st[1:, 4]))
    u, v = cen[k]
    ray = geo.T[:3, :3] @ np.linalg.inv(geo.K) @ np.array([u, v, 1.0])
    s = (geo.table_z + CRO_COM_Z - geo.T[2, 3]) / ray[2]
    p = geo.T[:3, 3] + s * ray
    return float(p[0]), float(p[1]), int(st[k, 4])


def croissant_verdict(x: float, y: float, geo: Geometry) -> Tuple[bool, bool]:
    """(centre of mass inside the blue box, whole croissant inside the white box)."""
    bx0, bx1, by0, by1 = geo.blue
    wx0, wx1, wy0, wy1 = geo.white
    in_blue = bx0 <= x <= bx1 and by0 <= y <= by1
    in_white = wx0 <= x - CRO_RADIUS and x + CRO_RADIUS <= wx1 and wy0 <= y - CRO_RADIUS and y + CRO_RADIUS <= wy1
    return in_blue, in_white


def pauses(t, q, grip_cmd, grip_pos, max_pause: float) -> List[Tuple[float, float]]:
    """[(start s, length s)] of still stretches longer than max_pause, outside the first and last 0.5 s."""
    k = 10  # 0.1 s at 100 Hz
    speed = np.abs(q[k:] - q[:-k]).max(1) / np.maximum(t[k:] - t[:-k], 1e-6)
    gspeed = np.abs(grip_pos[k:] - grip_pos[:-k]) / np.maximum(t[k:] - t[:-k], 1e-6)
    gchange = np.abs(np.diff(grip_cmd))[k - 1:] > 0
    still = (speed < STILL_RAD_S) & (gspeed < 0.05) & ~gchange
    tm = t[k:]
    ok = (tm > 0.5) & (tm < t[-1] - 0.5)
    out, start = [], None
    for i in range(len(still)):
        s = still[i] and ok[i]
        if s and start is None:
            start = i
        if (not s or i == len(still) - 1) and start is not None:
            length = tm[i] - tm[start]
            if length > max_pause:
                out.append((round(float(tm[start]), 1), round(float(length), 1)))
            start = None
    return out


def check(ep: Path, task: str, geo: Geometry, cfg: CameraConfig, st: QCSettings, db) -> Dict:
    """Run every check on one finished episode; ``reasons`` is empty when it passes."""
    meta = json.load(open(ep / "metadata.json"))
    r: Dict = dict(episode=ep.name, duration_s=meta.get("duration_s"), reasons=[])
    cc = meta.get("color_controls") or {}
    for name in ("scene_camera", "left_wrist_camera"):
        want, got = cfg.get_color_controls(name), cc.get(name)
        if not isinstance(got, dict) or want is None or any(abs(got.get(k, -1) - v) > 0.5 for k, v in want.items()):
            r["reasons"].append(f"{name} color_controls {got!r} (camera.json {want})")
    w = first_frame(ep / "cameras/left_wrist_camera.bag", "left_wrist_camera", cfg.get_crop("left_wrist_camera"))
    if w is None:
        r["reasons"].append("wrist bag unreadable")
    else:
        r["wrist_mean"] = round(float(w.mean()), 1)
        tb, tk = fingertips(w)
        r["wrist_tips"] = [tb[0] if tb else None, tk[0] if tk else None]
        for label, tip, ref in (("blue", tb, st.ref_tips[0]), ("black", tk, st.ref_tips[1])):
            if tip is None:
                r["reasons"].append(f"wrist: {label} fingertip not found (camera slipped?)")
            elif abs(tip[0] - ref) > st.max_tip_px:
                r["reasons"].append(f"wrist: {label} tip row {tip[0]} vs {ref} (camera slipped?)")
        if r["wrist_mean"] < st.min_wrist_mean:
            r["reasons"].append(f"wrist brightness {r['wrist_mean']:.0f} < {st.min_wrist_mean:.0f} (camera slipped?)")
    s = first_frame(ep / "cameras/scene_camera.bag", "scene_camera", cfg.get_crop("scene_camera"))
    if s is None:
        r["reasons"].append("scene bag unreadable")
    else:
        r["scene_mean"] = round(float(s.mean()), 1)
        if not (st.scene_mean[0] <= r["scene_mean"] <= st.scene_mean[1]):
            r["reasons"].append(f"scene brightness {r['scene_mean']:.0f} outside {tuple(st.scene_mean)}")
        c = croissant_position(s, geo)
        if c is None:
            r["reasons"].append("croissant not found in the first scene frame")
        else:
            x, y, _ = c
            r["croissant"] = [round(x, 3), round(y, 3)]
            in_blue, in_white = croissant_verdict(x, y, geo)
            if not in_blue:
                r["reasons"].append(f"croissant centre ({x:.3f}, {y:+.3f}) outside the blue box")
            if not in_white:
                r["reasons"].append(f"croissant ({x:.3f}, {y:+.3f}) may cross the white box")
    d = np.load(ep / "robot_data.npz")
    t = (d["timestamps"] - d["timestamps"][0]) / 1e9
    gcmd = d["follower_l_joint_cmd"][:, 6]
    r["closes"] = int(np.sum((gcmd[:-1] >= 0.5) & (gcmd[1:] < 0.5)))
    if r["closes"] > st.max_closes:
        r["reasons"].append(f"{r['closes']} gripper closes (> {st.max_closes}: regrasp?)")
    r["pauses"] = pauses(t, d["follower_l_joint_pos"], gcmd, d["follower_l_gripper_pos"][:, 0], st.max_pause)
    for start, length in r["pauses"]:
        r["reasons"].append(f"pause {length:.1f} s at {start:.1f} s")
    dur = r["duration_s"] or 0.0
    if not (st.min_s <= dur <= st.max_s):
        r["reasons"].append(f"duration {dur:.1f} s outside [{st.min_s}, {st.max_s}]")
    demo = db.get_demonstration_by_raw_path(f"data/raw/{task}/{ep.name}")
    r["label"] = demo["status"] if demo else "no DB row"
    return r


def apply_duration_outliers(results: Dict[str, Dict]) -> None:
    """Flag durations far from the median of the other episodes, once there are enough of them."""
    rs = list(results.values())
    for r in rs:
        r["reasons"] = [x for x in r["reasons"] if not x.startswith("duration outlier")]
        others = [x["duration_s"] for x in rs if x is not r and x.get("duration_s")]
        if len(others) >= 5 and r.get("duration_s"):
            m = float(np.median(others))
            if not (0.5 * m <= r["duration_s"] <= 1.8 * m):
                r["reasons"].append(f"duration outlier ({r['duration_s']:.1f} s vs median {m:.1f} s)")


def coverage(results: Dict[str, Dict], geo: Geometry) -> Tuple[np.ndarray, int]:
    """(GRID_X x GRID_Y counts of croissant starts over the blue box; row 0 nearest the robot, column 0 the robot's
    left), and the number outside it."""
    bx0, bx1, by0, by1 = geo.blue
    G = np.zeros((GRID_X, GRID_Y), int)
    outside = 0
    for r in results.values():
        if not r.get("croissant"):
            continue
        cell = grid_cell(*r["croissant"], geo)
        if cell is None:
            outside += 1
        else:
            G[cell] += 1
    return G, outside


def grid_cell(x: float, y: float, geo: Geometry) -> Optional[Tuple[int, int]]:
    bx0, bx1, by0, by1 = geo.blue
    i = int(np.floor((x - bx0) / (bx1 - bx0) * GRID_X))
    j = int(np.floor((by1 - y) / (by1 - by0) * GRID_Y))
    return (i, j) if 0 <= i < GRID_X and 0 <= j < GRID_Y else None


def line(r: Dict) -> str:
    parts = [f"{r['duration_s']:.1f} s" if r.get("duration_s") else "? s", f"label {r.get('label')}"]
    if r.get("croissant"):
        parts.append(f"croissant ({r['croissant'][0]:.3f}, {r['croissant'][1]:+.3f})")
    parts.append(f"closes {r.get('closes')}")
    if r.get("wrist_tips"):
        parts.append(f"wrist tips {r['wrist_tips'][0]}/{r['wrist_tips'][1]} mean {r.get('wrist_mean', 0):.0f}")
    if r.get("scene_mean") is not None:
        parts.append(f"scene {r['scene_mean']:.0f}")
    head = "PASS" if not r["reasons"] else "FLAG: " + "; ".join(r["reasons"])
    return f"{r['episode']} {head}  | " + ", ".join(parts)


def tally_text(results: Dict[str, Dict], geo: Geometry) -> str:
    rs = [results[k] for k in sorted(results)]
    durs = [r["duration_s"] for r in rs if r.get("duration_s")]
    n, npass = len(rs), sum(not r["reasons"] for r in rs)
    labels: Dict[str, int] = {}
    for r in rs:
        labels[r.get("label")] = labels.get(r.get("label"), 0) + 1
    out = [f"TALLY: {n} episodes, {npass} PASS, {n - npass} flagged | labels {labels} | "
           f"median duration {float(np.median(durs)) if durs else 0.0:.1f} s"]
    G, outside = coverage(results, geo)
    bx0, bx1, by0, by1 = geo.blue
    ys, xs = np.linspace(by1, by0, GRID_Y + 1), np.linspace(bx0, bx1, GRID_X + 1)
    out.append(f"croissant starts over the blue box (top = far from the robot; left = robot's left, +y); {outside} outside:")
    out.append("            " + " ".join(f"{(ys[j] + ys[j + 1]) / 2:+6.2f}" for j in range(GRID_Y)) + "   <- y (m)")
    for i in reversed(range(GRID_X)):
        cells = " ".join(f"{G[i, j]:>6d}" if G[i, j] else "     ." for j in range(GRID_Y))
        out.append(f"  x {xs[i]:.2f}-{xs[i + 1]:.2f} {cells}")
    return "\n".join(out)
