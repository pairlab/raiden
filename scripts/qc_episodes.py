#!/usr/bin/env python
"""Per-episode QC of real demos while they are being recorded.

For every finished episode of a task (metadata.json with complete=true):
  * wrist camera, first frame: fingertip rows against the reference and mean brightness (a slipped mount shows
    as tips that moved, or a dark frame looking at the partition)
  * scene camera, first frame: brightness; metadata.json's color_controls match camera.json for both cameras
  * pauses: stretches longer than --max-pause s with the arm and gripper still, outside the first/last 0.5 s
  * croissant start position (first scene frame, orange blob projected to its centre-of-mass height): centre of
    mass inside the blue box, whole croissant (51 mm radius) inside the white box
  * gripper: close events of the open/closed command; more than --max-closes flags a regrasp
  * duration: outside --min-s/--max-s, or far from the median of the episodes so far
One line per episode (PASS or the reasons), then the running tally and the croissant coverage of the blue box.
Results go to data/qc/<task>/ (qc_log.txt, qc_results.json). Deleted episodes drop out of the tally.

    uv run python scripts/qc_episodes.py croissant_oven_real_newrig --watch
"""

import argparse
import json
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from raiden._config import CAMERA_CONFIG
from raiden.camera_config import CameraConfig
from raiden.cameras.realsense import RealSenseCamera

REPO = Path(__file__).resolve().parents[1]
IN = 0.0254
COM_MARGIN = 0.004  # the task JSON's croissant region is this much inside the centre-of-mass box
CRO_RADIUS = 0.051  # croissant footprint radius about its centre, any yaw
CRO_COM_Z = 0.0215  # centre-of-mass height above the table
STILL_RAD_S = 0.02
GRID_X, GRID_Y = 3, 6


def load_geometry(vla: Path):
    rigd = vla / "mesa/task_suites/rigs/raiden_lab"
    rig, lay = json.load(open(rigd / "rig.json")), json.load(open(rigd / "layout.json"))
    (tx0, tx1), (ty0, ty1) = lay["table"]["atlas"]["x_range"], lay["table"]["atlas"]["y_range"]
    tc = ((tx0 + tx1) / 2, (ty0 + ty1) / 2)
    task = json.load(open(vla / "mesa/task_suites/bddl_files/raiden_lab/croissant_oven_heating_region/source/000.json"))
    x0, y0, x1, y1 = task["regions"]["table_object_0_init_region"]["ranges"][0]
    cam = rig["cameras"]["scene_camera"]
    return dict(
        white=(tx0 + 6 * IN, tx1, ty0 + 13 * IN, ty1 - 13 * IN),
        blue=(x0 + tc[0] - COM_MARGIN, x1 + tc[0] + COM_MARGIN, y0 + tc[1] - COM_MARGIN, y1 + tc[1] + COM_MARGIN),
        table_z=rig["table"]["z_in_base_at_origin"], T=np.array(cam["T_base_cam"]), K=np.array(cam["K"]))


def first_frame(bag: Path, name: str, crop):
    cam = RealSenseCamera.from_bag(name, bag, crop=crop)
    try:
        for _ in range(5):
            if cam.grab():
                return cam.get_frame().color.copy()
    finally:
        cam.close()
    return None


def fingertips(img):
    """(blue-finger tip, black-finger tip) as (row, col), None where not found."""
    b, g, r = [img[..., i].astype(np.int32) for i in range(3)]
    blue = (b - r > 60) & (b - g > 20) & (b > 110)  # the blue finger enters bottom left
    dark = img.max(2) < 45  # the black finger enters bottom right
    tips = []
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


def project(P, T, K):
    pc = (P - T[:3, 3]) @ T[:3, :3]
    return (pc @ K.T)[:, :2] / pc[:, 2:]


def croissant_position(img, geo):
    """Centre of mass (x, y) in the base frame from the largest orange blob over the white box, or None."""
    x0, x1, y0, y1 = geo["white"]
    poly = project(np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]]) + [0, 0, geo["table_z"]], geo["T"], geo["K"])
    region = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(region, [poly.round().astype(np.int32)], 1)
    region = cv2.dilate(region, np.ones((31, 31), np.uint8))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] >= 5) & (hsv[..., 0] <= 25) & (hsv[..., 1] >= 30) & (hsv[..., 2] >= 100) & (region > 0)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    if n < 2 or st[1:, 4].max() < 150:
        return None
    k = 1 + int(np.argmax(st[1:, 4]))
    u, v = cen[k]
    T, K = geo["T"], geo["K"]
    ray = T[:3, :3] @ np.linalg.inv(K) @ np.array([u, v, 1.0])
    s = (geo["table_z"] + CRO_COM_Z - T[2, 3]) / ray[2]
    p = T[:3, 3] + s * ray
    return float(p[0]), float(p[1]), int(st[k, 4])


def pauses(t, q, grip_cmd, grip_pos, max_pause):
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


def check(ep: Path, task: str, geo, cfg: CameraConfig, args, db):
    meta = json.load(open(ep / "metadata.json"))
    r = dict(episode=ep.name, duration_s=meta.get("duration_s"), reasons=[])
    # colour lock recorded for both cameras, as camera.json has it
    cc = meta.get("color_controls") or {}
    for name in ("scene_camera", "left_wrist_camera"):
        want = cfg.get_color_controls(name)
        got = cc.get(name)
        if not isinstance(got, dict) or want is None or any(abs(got.get(k, -1) - v) > 0.5 for k, v in want.items()):
            r["reasons"].append(f"{name} color_controls {got!r} (camera.json {want})")
    # wrist: fingertips and brightness
    w = first_frame(ep / "cameras/left_wrist_camera.bag", "left_wrist_camera", cfg.get_crop("left_wrist_camera"))
    if w is None:
        r["reasons"].append("wrist bag unreadable")
    else:
        r["wrist_mean"] = round(float(w.mean()), 1)
        tb, tk = fingertips(w)
        r["wrist_tips"] = [tb[0] if tb else None, tk[0] if tk else None]
        for label, tip, ref in (("blue", tb, args.ref_tips[0]), ("black", tk, args.ref_tips[1])):
            if tip is None:
                r["reasons"].append(f"wrist: {label} fingertip not found (camera slipped?)")
            elif abs(tip[0] - ref) > args.max_tip_px:
                r["reasons"].append(f"wrist: {label} tip row {tip[0]} vs {ref} (camera slipped?)")
        if r["wrist_mean"] < args.min_wrist_mean:
            r["reasons"].append(f"wrist brightness {r['wrist_mean']:.0f} < {args.min_wrist_mean} (camera slipped?)")
    # scene: brightness and croissant start
    s = first_frame(ep / "cameras/scene_camera.bag", "scene_camera", cfg.get_crop("scene_camera"))
    if s is None:
        r["reasons"].append("scene bag unreadable")
    else:
        r["scene_mean"] = round(float(s.mean()), 1)
        if not (args.scene_mean[0] <= r["scene_mean"] <= args.scene_mean[1]):
            r["reasons"].append(f"scene brightness {r['scene_mean']:.0f} outside {args.scene_mean}")
        c = croissant_position(s, geo)
        if c is None:
            r["reasons"].append("croissant not found in the first scene frame")
        else:
            x, y, _ = c
            r["croissant"] = [round(x, 3), round(y, 3)]
            bx0, bx1, by0, by1 = geo["blue"]
            wx0, wx1, wy0, wy1 = geo["white"]
            if not (bx0 <= x <= bx1 and by0 <= y <= by1):
                r["reasons"].append(f"croissant centre ({x:.3f}, {y:+.3f}) outside the blue box")
            if not (wx0 <= x - CRO_RADIUS and x + CRO_RADIUS <= wx1 and wy0 <= y - CRO_RADIUS and y + CRO_RADIUS <= wy1):
                r["reasons"].append(f"croissant ({x:.3f}, {y:+.3f}) may cross the white box")
    # robot data: pauses, gripper cycles
    d = np.load(ep / "robot_data.npz")
    t = (d["timestamps"] - d["timestamps"][0]) / 1e9
    gcmd = d["follower_l_joint_cmd"][:, 6]
    r["closes"] = int(np.sum((gcmd[:-1] >= 0.5) & (gcmd[1:] < 0.5)))
    if r["closes"] > args.max_closes:
        r["reasons"].append(f"{r['closes']} gripper closes (> {args.max_closes}: regrasp?)")
    r["pauses"] = pauses(t, d["follower_l_joint_pos"], gcmd, d["follower_l_gripper_pos"][:, 0], args.max_pause)
    for start, length in r["pauses"]:
        r["reasons"].append(f"pause {length:.1f} s at {start:.1f} s")
    dur = r["duration_s"] or 0.0
    if not (args.min_s <= dur <= args.max_s):
        r["reasons"].append(f"duration {dur:.1f} s outside [{args.min_s}, {args.max_s}]")
    demo = db.get_demonstration_by_raw_path(f"data/raw/{task}/{ep.name}")
    r["label"] = demo["status"] if demo else "no DB row"
    return r


def line(r):
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


def tally(results, geo):
    rs = [results[k] for k in sorted(results)]
    durs = [r["duration_s"] for r in rs if r.get("duration_s")]
    med = float(np.median(durs)) if durs else 0.0
    for r in rs:  # duration outliers against the others, once there are enough
        r["reasons"] = [x for x in r["reasons"] if not x.startswith("duration outlier")]
        others = [x["duration_s"] for x in rs if x is not r and x.get("duration_s")]
        if len(others) >= 5 and r.get("duration_s"):
            m = float(np.median(others))
            if not (0.5 * m <= r["duration_s"] <= 1.8 * m):
                r["reasons"].append(f"duration outlier ({r['duration_s']:.1f} s vs median {m:.1f} s)")
    n, npass = len(rs), sum(not r["reasons"] for r in rs)
    labels = {}
    for r in rs:
        labels[r.get("label")] = labels.get(r.get("label"), 0) + 1
    out = [f"TALLY: {n} episodes, {npass} PASS, {n - npass} flagged | labels {labels} | median duration {med:.1f} s"]
    bx0, bx1, by0, by1 = geo["blue"]
    G = np.zeros((GRID_X, GRID_Y), int)
    outside = 0
    for r in rs:
        if not r.get("croissant"):
            continue
        x, y = r["croissant"]
        i, j = int(np.floor((x - bx0) / (bx1 - bx0) * GRID_X)), int(np.floor((by1 - y) / (by1 - by0) * GRID_Y))
        if 0 <= i < GRID_X and 0 <= j < GRID_Y:
            G[i, j] += 1
        else:
            outside += 1
    ys = np.linspace(by1, by0, GRID_Y + 1)
    out.append(f"croissant starts over the blue box (top = far from the robot; left = robot's left, +y); {outside} outside:")
    out.append("            " + " ".join(f"{(ys[j] + ys[j + 1]) / 2:+6.2f}" for j in range(GRID_Y)) + "   <- y (m)")
    xs = np.linspace(bx0, bx1, GRID_X + 1)
    for i in reversed(range(GRID_X)):
        cells = " ".join(f"{G[i, j]:>6d}" if G[i, j] else "     ." for j in range(GRID_Y))
        out.append(f"  x {xs[i]:.2f}-{xs[i + 1]:.2f} {cells}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task")
    ap.add_argument("--watch", action="store_true", help="keep checking new episodes every 5 s")
    ap.add_argument("--vla", default=str(Path.home() / "robot/vla-benchmark"), help="vla-benchmark checkout (rig, task JSON)")
    ap.add_argument("--ref-tips", type=int, nargs=2, default=[312, 304],
                    help="wrist fingertip rows (blue, black) with the arm at home; measured 2026-09-26 19:05")
    ap.add_argument("--max-tip-px", type=int, default=15)
    ap.add_argument("--min-wrist-mean", type=float, default=100)
    ap.add_argument("--scene-mean", type=float, nargs=2, default=[90, 140])
    ap.add_argument("--max-pause", type=float, default=1.0)
    ap.add_argument("--max-closes", type=int, default=2)
    ap.add_argument("--min-s", type=float, default=5.0)
    ap.add_argument("--max-s", type=float, default=90.0)
    ap.add_argument("--out", default="", help="default data/qc/<task>")
    args = ap.parse_args()
    from raiden.db.database import get_db

    raw = REPO / "data/raw" / args.task
    out = Path(args.out) if args.out else REPO / "data/qc" / args.task
    out.mkdir(parents=True, exist_ok=True)
    state_file, log_file = out / "qc_results.json", out / "qc_log.txt"
    results = json.load(open(state_file)) if state_file.exists() else {}
    geo, cfg = load_geometry(Path(args.vla)), CameraConfig(CAMERA_CONFIG)

    def say(msg):
        print(msg, flush=True)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    say(f"QC {args.task}: {raw}  (blue box x {geo['blue'][0]:.3f}-{geo['blue'][1]:.3f}, y {geo['blue'][2]:+.3f}..{geo['blue'][3]:+.3f};"
        f" white x {geo['white'][0]:.3f}-{geo['white'][1]:.3f}, y ±{geo['white'][3]:.3f}; wrist tips {args.ref_tips})")
    while True:
        changed = False
        eps = sorted(d for d in raw.iterdir() if d.is_dir() and d.name.isdigit()) if raw.exists() else []
        for gone in set(results) - {e.name for e in eps}:
            say(f"{gone} deleted: dropped from the tally")
            del results[gone]
            changed = True
        for ep in eps:
            mf = ep / "metadata.json"
            if not mf.exists() or not json.load(open(mf)).get("complete", False):
                continue
            stamp = mf.stat().st_mtime
            if ep.name in results and results[ep.name].get("_stamp") == stamp:
                continue
            try:
                r = check(ep, args.task, geo, cfg, args, get_db())
            except Exception as e:  # one bad episode must not stop the watcher
                r = dict(episode=ep.name, reasons=[f"QC error: {e!r}"], duration_s=None)
                traceback.print_exc()
            r["_stamp"] = stamp
            results[ep.name] = r
            changed = True
            say(line(r))
        if changed:
            say(tally(results, geo))
            json.dump(results, open(state_file, "w"), indent=1)
        if not args.watch:
            break
        time.sleep(5)


if __name__ == "__main__":
    main()
