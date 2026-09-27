"""Per-episode quality checks for real demos of the raiden_lab croissant task.

Used by ``scripts/qc_episodes.py`` (a watcher) and by ``rd record --ui``. For one finished episode:

* wrist camera, first frame: fingertip rows against the reference and mean brightness. A slipped mount shows as
  tips that moved, or a dark frame looking at the partition.
* scene camera, first frame: brightness; metadata.json's color_controls match camera.json for both cameras.
* pauses: stretches with the arm and gripper still, outside the first and last 0.5 s.
* croissant start (first scene frame, the orange blob projected to its centre-of-mass height): centre of mass in the
  blue box, the whole croissant (51 mm radius) in the white box.
* oven: where it starts (knob faces, first scene frame) against the sim's position, and how far it moved between where
  the knobs are first and last seen (the sim's oven is welded to the table).
* gripper: close events of the open/closed command. The door is hooked open with the gripper open, so the only close
  is the croissant grasp: anything but ``closes`` is flagged (0: no grasp; more: a regrasp or a handle grasp).
* duration: outside ``min_s``/``max_s``, or far from the median of the other episodes.

The placement limits come from the vla-benchmark twin (rig.json, layout.json and the task JSON), so the checks follow
the sim's spawn regions.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs

from raiden.camera_config import CameraConfig
from raiden.cameras.realsense import RealSenseCamera

IN = 0.0254
COM_MARGIN = 0.004  # the task JSON's croissant region is this much inside the centre-of-mass box
CRO_RADIUS = 0.051  # croissant footprint radius about its centre, any yaw
CRO_COM_Z = 0.0215  # centre-of-mass height above the table
OVEN_FOOTPRINT = (-0.1518, 0.1100, -0.1937, 0.1930)  # x, y extents about the oven origin (sim model, handle incl.)
# Knob-face centres about the oven origin at yaw 0 (base frame, z above the table). Each knob is a cylinder (radius
# 14 mm) whose axis points out of the front panel toward the robot; the face is half its length (9 mm) out.
OVEN_KNOB_FACES = np.array([[-0.129, -0.150, 0.165], [-0.129, -0.150, 0.115], [-0.129, -0.150, 0.065]])
OVEN_X_TOL = 0.015  # real oven x may differ this much from the sim's fixed x
OVEN_GRID = 5  # bins over the sim's oven y range
STILL_RAD_S = 0.02
QC_VERSION = 3  # results from an older version are checked again (2: oven position, 3: oven moved, 1 close)
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
    closes: int = 1  # the croissant grasp; the door is hooked open with the gripper open (2026-09-27)
    min_s: float = 5.0
    max_s: float = 90.0
    max_oven_move: float = 0.015  # m between where the oven's knobs are first and last seen


@dataclass
class Geometry:
    white: Tuple[float, float, float, float]  # x0, x1, y0, y1 in the base frame
    blue: Tuple[float, float, float, float]
    orange: Tuple[float, float, float, float]
    oven_region: Tuple[float, float, float, float]  # the sim's range of the oven origin
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
        oven_region=(ox0 + tc[0], ox1 + tc[0], oy0 + tc[1], oy1 + tc[1]),
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


_KNOB_TEMPLATE = Path(__file__).with_name("oven_knob_face.png")  # 25 x 25 px, the top knob face in the scene camera


def _sample(m: np.ndarray, uv: np.ndarray, half: int) -> np.ndarray:
    """Values of a matchTemplate map at template centres ``uv`` (outside the map: -1)."""
    col, row = uv[:, 0].round().astype(int) - half, uv[:, 1].round().astype(int) - half
    ok = (row >= 0) & (row < m.shape[0]) & (col >= 0) & (col < m.shape[1])
    out = np.full(len(uv), -1.0)
    out[ok] = m[row[ok], col[ok]]
    return out


def oven_position(img: np.ndarray, geo: Geometry, min_score: float = 0.55) -> Optional[Tuple[float, float, float]]:
    """Oven origin (x, y) in the base frame at yaw 0, and the mean knob match, from the three knob faces.

    A real knob-face template is correlated over the image (normalised cross-correlation). Every oven position on a
    3 mm grid over the table is scored by the best match within 4 px of each of its three projected faces (the model
    and the real knobs differ by ~3 px), the weakest face counting double so one bright spot cannot win. The pose is
    then solved from the three matched face pixels by least squares. None if the mean match is below min_score (oven
    not in view or occluded).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tpl = cv2.imread(str(_KNOB_TEMPLATE), cv2.IMREAD_GRAYSCALE)
    half = tpl.shape[0] // 2
    ncc = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
    near = cv2.dilate(ncc, np.ones((9, 9), np.uint8))  # best match within 4 px

    def faces(c):
        return [project(np.c_[c[:, 0] + f[0], c[:, 1] + f[1], np.full(len(c), geo.table_z + f[2])], geo.T, geo.K)
                for f in OVEN_KNOB_FACES]

    X, Y = np.meshgrid(np.arange(0.68, 0.88, 0.003), np.arange(-0.17, 0.17, 0.003), indexing="ij")
    c = np.stack([X.ravel(), Y.ravel()], 1)
    per = np.stack([_sample(near, uv, half) for uv in faces(c)], 1)
    k = int(np.argmax(per.mean(1) + per.min(1)))
    score = float(per[k].mean())
    if score < min_score:
        return None
    # the matched face pixels: the correlation peak within 4 px of each predicted face
    peaks = []
    for uv in faces(c[k:k + 1]):
        u0, v0 = int(round(uv[0, 0])) - half, int(round(uv[0, 1])) - half
        win = ncc[max(v0 - 4, 0):v0 + 5, max(u0 - 4, 0):u0 + 5]
        dv, du = np.unravel_index(int(np.argmax(win)), win.shape)
        peaks.append((max(u0 - 4, 0) + du + half, max(v0 - 4, 0) + dv + half))
    peaks = np.array(peaks, float)
    xy = c[k].copy()
    for _ in range(10):  # Gauss-Newton on the face reprojection error, x and y
        r = np.concatenate([uv[0] for uv in faces(xy[None])]) - peaks.ravel()
        J = np.stack([(np.concatenate([uv[0] for uv in faces((xy + d)[None])]) - peaks.ravel() - r) / 1e-4
                      for d in (np.array([1e-4, 0]), np.array([0, 1e-4]))], 1)
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        xy += step
        if np.abs(step).max() < 1e-5:
            break
    return float(xy[0]), float(xy[1]), round(score, 3)


def _open_bag(bag: Path):
    p, c = rs.pipeline(), rs.config()
    rs.config.enable_device_from_file(c, str(bag), repeat_playback=False)
    c.enable_stream(rs.stream.color)
    playback = p.start(c).get_device().as_playback()
    playback.set_real_time(False)
    return p, playback


def _oven_matches(bag: Path, geo: Geometry, crop, ts0: float, t0: float, t1: float,
                  stop_after: Optional[int] = None, every: int = 3) -> List[Tuple[float, float, float]]:
    """(t, x, y) of the oven at every ``every``-th colour frame of a scene bag between t0 and t1 s after its first
    frame (timestamp ``ts0``, ms), where the knobs are seen; stops early after ``stop_after`` matches. The window is
    judged by each frame's own timestamp: right after a seek the bag still returns a frame from before it."""
    p, playback = _open_bag(bag)
    found, i = [], 0
    try:
        if t0 > 0:
            playback.seek(datetime.timedelta(seconds=t0))
        while stop_after is None or len(found) < stop_after:
            ok, fs = p.try_wait_for_frames(2000)
            if not ok:
                break
            cf = fs.get_color_frame()
            if not cf:
                continue
            t = (cf.get_timestamp() - ts0) / 1000
            if t < t0 - 0.1:
                continue
            if t > t1:
                break
            if i % every == 0:
                img = np.asanyarray(cf.get_data())
                if crop is not None:
                    x, y, w, h = crop
                    img = img[y:y + h, x:x + w]
                o = oven_position(img, geo)
                if o is not None:
                    found.append((t, o[0], o[1]))
            i += 1
    finally:
        p.stop()
    return found


def oven_start_end(bag: Path, geo: Geometry, crop=None, span: float = 12.0, n: int = 5):
    """(start, end, last_seen_s, duration_s) of the oven in a scene bag. start and end are the oven (x, y) where its
    knobs are first and last seen: the median of the first and of the last ``n`` matches (every 3rd frame, ~10 Hz).
    The start is searched in the first ``span`` s. The end is searched in the last ``span`` s, then in earlier windows
    until there are ``n`` matches: the arm often hides the knobs for the last seconds, so the end can be well before
    the end of the episode (``last_seen_s``). None for an end with fewer than 3 matches."""
    p, playback = _open_bag(bag)
    try:
        duration = playback.get_duration().total_seconds()
        ok, fs = p.try_wait_for_frames(2000)
        ts0 = fs.get_color_frame().get_timestamp() if ok and fs.get_color_frame() else None
    finally:
        p.stop()
    if ts0 is None:
        return None, None, None, duration
    start = _oven_matches(bag, geo, crop, ts0, 0.0, span, stop_after=n)
    end, t1 = [], duration + 1.0
    while len(end) < n and t1 > 0:
        t0 = max(t1 - span, 0.0)
        end = _oven_matches(bag, geo, crop, ts0, t0, t1) + end
        t1 = t0
    start, end = start[:n], end[-n:]

    def mid(q):
        return tuple(float(v) for v in np.median(np.array(q)[:, 1:], axis=0)) if len(q) >= 3 else None

    return mid(start), mid(end), (float(end[-1][0]) if len(end) >= 3 else None), duration


def oven_verdict(x: float, y: float, geo: Geometry) -> Tuple[bool, bool, bool]:
    """(x within OVEN_X_TOL of the sim's, y within the sim's range, whole oven inside the white box) at yaw 0."""
    ox0, ox1, oy0, oy1 = geo.oven_region
    wx0, wx1, wy0, wy1 = geo.white
    x_ok = ox0 - OVEN_X_TOL <= x <= ox1 + OVEN_X_TOL
    y_ok = oy0 <= y <= oy1
    in_white = (wx0 <= x + OVEN_FOOTPRINT[0] and x + OVEN_FOOTPRINT[1] <= wx1
                and wy0 <= y + OVEN_FOOTPRINT[2] and y + OVEN_FOOTPRINT[3] <= wy1)
    return x_ok, y_ok, in_white


def oven_cell(y: float, geo: Geometry) -> Optional[int]:
    oy0, oy1 = geo.oven_region[2], geo.oven_region[3]
    j = int(np.floor((oy1 - y) / (oy1 - oy0) * OVEN_GRID))  # 0 = the robot's left (+y)
    return j if 0 <= j < OVEN_GRID else None


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
        o = oven_position(s, geo)
        if o is None:
            r["reasons"].append("oven knobs not found in the first scene frame")
        else:
            ox, oy, _ = o
            r["oven"] = [round(ox, 3), round(oy, 3)]
            x_ok, y_ok, o_white = oven_verdict(ox, oy, geo)
            if not x_ok:
                r["reasons"].append(f"oven x {ox:.3f}, sim {sum(geo.oven_region[:2]) / 2:.3f} (> {OVEN_X_TOL * 1000:.0f} mm off)")
            if not y_ok:
                r["reasons"].append(f"oven y {oy:+.3f} outside the sim's ±{geo.oven_region[3]:.3f}")
            if not o_white:
                r["reasons"].append(f"oven ({ox:.3f}, {oy:+.3f}) crosses the white box")
        a, b, seen, dur = oven_start_end(ep / "cameras/scene_camera.bag", geo, cfg.get_crop("scene_camera"))
        if a and b:
            dx, dy = b[0] - a[0], b[1] - a[1]
            r["oven_moved"] = [round(dx, 3), round(dy, 3)]
            r["oven_last_seen_s"] = round(seen, 1)
            if np.hypot(dx, dy) > st.max_oven_move:
                hidden = f" by {seen:.0f} s, knobs hidden for the last {dur - seen:.0f} s" if dur - seen > 3 else ""
                r["reasons"].append(f"oven moved {np.hypot(dx, dy) * 1000:.0f} mm (x {dx * 1000:+.0f}, "
                                    f"y {dy * 1000:+.0f} mm){hidden}")
    d = np.load(ep / "robot_data.npz")
    t = (d["timestamps"] - d["timestamps"][0]) / 1e9
    gcmd = d["follower_l_joint_cmd"][:, 6]
    r["closes"] = int(np.sum((gcmd[:-1] >= 0.5) & (gcmd[1:] < 0.5)))
    if r["closes"] != st.closes:
        why = "croissant not grasped?" if r["closes"] < st.closes else "regrasp or handle grasp?"
        r["reasons"].append(f"{r['closes']} gripper closes (expected {st.closes}: {why})")
    r["pauses"] = pauses(t, d["follower_l_joint_pos"], gcmd, d["follower_l_gripper_pos"][:, 0], st.max_pause)
    for start, length in r["pauses"]:
        r["reasons"].append(f"pause {length:.1f} s at {start:.1f} s")
    dur = r["duration_s"] or 0.0
    if not (st.min_s <= dur <= st.max_s):
        r["reasons"].append(f"duration {dur:.1f} s outside [{st.min_s}, {st.max_s}]")
    demo = db.get_demonstration_by_raw_path(f"data/raw/{task}/{ep.name}")
    r["label"] = demo["status"] if demo else "no DB row"
    r["_qc_version"] = QC_VERSION
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


def coverage(results: Dict[str, Dict], geo: Geometry, label: Optional[str] = "success") -> Tuple[np.ndarray, int]:
    """(GRID_X x GRID_Y counts of croissant starts over the blue box; row 0 nearest the robot, column 0 the robot's
    left), and the number outside it. Only episodes labelled ``label`` count (None: all)."""
    G = np.zeros((GRID_X, GRID_Y), int)
    outside = 0
    for r in results.values():
        if (label is not None and r.get("label") != label) or not r.get("croissant"):
            continue
        cell = grid_cell(*r["croissant"], geo)
        if cell is None:
            outside += 1
        else:
            G[cell] += 1
    return G, outside


def oven_coverage(results: Dict[str, Dict], geo: Geometry, label: Optional[str] = "success") -> Tuple[np.ndarray, int]:
    """(OVEN_GRID counts of oven positions over the sim's y range, column 0 the robot's left; number outside).
    Only episodes labelled ``label`` count (None: all)."""
    G = np.zeros(OVEN_GRID, int)
    outside = 0
    for r in results.values():
        if (label is not None and r.get("label") != label) or not r.get("oven"):
            continue
        j = oven_cell(r["oven"][1], geo)
        if j is None:
            outside += 1
        else:
            G[j] += 1
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
    if r.get("oven"):
        parts.append(f"oven ({r['oven'][0]:.3f}, {r['oven'][1]:+.3f})")
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
    out.append(f"croissant starts of the {labels.get('success', 0)} successes over the blue box (top = far from the "
               f"robot; left = robot's left, +y); {outside} outside:")
    out.append("            " + " ".join(f"{(ys[j] + ys[j + 1]) / 2:+6.2f}" for j in range(GRID_Y)) + "   <- y (m)")
    for i in reversed(range(GRID_X)):
        cells = " ".join(f"{G[i, j]:>6d}" if G[i, j] else "     ." for j in range(GRID_Y))
        out.append(f"  x {xs[i]:.2f}-{xs[i + 1]:.2f} {cells}")
    O, o_out = oven_coverage(results, geo)
    oy = np.linspace(geo.oven_region[3], geo.oven_region[2], OVEN_GRID + 1)
    out.append(f"oven positions of the successes over the sim's y range (left = robot's left); {o_out} outside:")
    out.append("            " + " ".join(f"{(oy[j] + oy[j + 1]) / 2:+6.3f}" for j in range(OVEN_GRID)) + "   <- y (m)")
    out.append("            " + " ".join(f"{O[j]:>6d}" if O[j] else "     ." for j in range(OVEN_GRID)))
    return "\n".join(out)
