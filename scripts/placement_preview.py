#!/usr/bin/env python
"""Live scene-camera view of the croissant task's placement limits, to place the croissant and the oven before a
policy run. Needs no robot, Quest, CAN or recording session: only the scene camera, opened the way ``rd record``
opens it (camera.json, colour lock included).

It draws the recording page's own overlay (``RecordUI._scene_overlay`` on ``raiden.qc``): the white, blue and orange
boxes, and the croissant (circle) and oven (outline) where they are now, green inside the limits and red outside, with
their x, y in the base frame. Blue-box cells tinted yellow have no successful demo of ``--task`` yet
(data/qc/<task>/qc_results.json, as on the page's coverage grid).

With ``--state ID`` it also draws a target start from ``--states`` (data/parity/initial_states.json) in cyan: the sim
croissant's silhouette and long axis at the state's pose, and the oven's footprint. The live croissant (GrabCut mask,
vla-benchmark's croissant_pose.grabcut) is fitted with the same sim model and its offset from the target printed; it
turns green within CRO_TOL. m appends the measured start to data/parity/measured_states.jsonl with the frame.

The scene camera pose comes from SCENE_CALIBRATION, the calibration accepted on 2026-10-18 and not yet installed;
``--calibration ''`` uses rig.json's instead (the pose before the camera moved; its x, y are 9-14 cm off now).

A RealSense streams to one process at a time: quit this (q) before ``rd serve`` or ``rd record`` opens the cameras.

    uv run python scripts/placement_preview.py
    uv run python scripts/placement_preview.py --state parity_s1
    uv run python scripts/placement_preview.py --snapshot scene.png      # one annotated frame, no window
    uv run python scripts/placement_preview.py --bag data/raw/croissant_oven_real_newrig/0000/cameras/scene_camera.bag

q or Esc quits, s saves a snapshot to data/placement_preview/. The overlay is only as good as the scene camera pose: if
the camera got bumped, recalibrate it.
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import ConvexHull

from raiden import qc
from raiden._config import CAMERA_CONFIG
from raiden.camera_config import CameraConfig
from raiden.cameras.realsense import RealSenseCamera
from raiden.record_ui import GREEN, RED, WHITE, YELLOW, RecordUI

REPO = Path(__file__).resolve().parents[1]
NAME = "scene_camera"
SCENE_CALIBRATION = Path.home() / ".config/raiden/calibration_captures/20261018_175242/calibration_results.scene_board_silhouette.json"
STATES = REPO / "data/parity/initial_states.json"
# The sim croissant (the Isaac twin imports MESA's): origin at its bounding-box centre, long side along body y.
CROISSANT_OBJ = qc.VLA_DEFAULT / "mesa/sim/envs/objects/assets/raiden_croissant/visual/model_0.obj"
CYAN = (255, 255, 0)
CRO_TOL = (0.01, 5.0)  # croissant position (m) and long-axis yaw (deg) counted as on target
OVEN_TOL = 0.005
MIN_FIT_IOU = 0.7  # below this the GrabCut mask is not the whole croissant (e.g. it took in the arm or its shadow)


def croissant_model() -> np.ndarray:
    """The sim croissant's convex hull vertices (N, 3), body frame, bottom at z = 0."""
    v = np.array([ln.split()[1:4] for ln in open(CROISSANT_OBJ) if ln.startswith("v ")], float)
    v = v[ConvexHull(v).vertices]
    return v - (0, 0, v[:, 2].min())


def silhouette(model: np.ndarray, x: float, y: float, theta: float, geo: qc.Geometry) -> np.ndarray:
    """Pixel outline of the model resting on the table at (x, y), long axis at theta deg from +x toward +y (body yaw
    theta + 90 deg)."""
    t = np.radians(theta + 90.0)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    p = np.c_[model[:, :2] @ R.T + (x, y), model[:, 2] + geo.table_z]
    return cv2.convexHull(qc.project(p, geo.T, geo.K).astype(np.float32))


def axis_px(x: float, y: float, theta: float, z: float, geo: qc.Geometry, half: float = 0.06) -> np.ndarray:
    d = half * np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta))])
    return qc.project(np.array([[x - d[0], y - d[1], z], [x + d[0], y + d[1], z]]), geo.T, geo.K).round().astype(int)


def fit_croissant(cp, mask: np.ndarray, model: np.ndarray, geo: qc.Geometry):
    """(x, y, theta in [0, 180), IoU) of the model silhouette that best matches the mask: yaw every 5 deg at the mask
    centre, then finer grids."""
    ys, xs = np.nonzero(mask)
    pad = 30
    u0, v0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
    win, off = mask[v0:ys.max() + pad, u0:xs.max() + pad], np.array([u0, v0], np.float32)
    x0, y0 = cp.centre_guess(mask, geo)
    best = (-1.0, x0, y0, 0.0)
    for step, n, yaws in ((0.006, 1, np.arange(0, 180, 5.0)), (0.002, 2, np.arange(-4, 4.1, 2.0)),
                          (0.001, 1, np.arange(-1, 1.1, 0.5))):
        _, bx, by, bt = best
        d = np.arange(-n, n + 1) * step
        for th in (yaws if step > 0.005 else bt + yaws):
            for dx in d:
                for dy in d:
                    s = cp._iou(win, off, silhouette(model, bx + dx, by + dy, th, geo))
                    if s > best[0]:
                        best = (s, bx + dx, by + dy, float(th))
    s, x, y, th = best
    return x, y, th % 180.0, s


def yaw_err(a: float, b: float) -> float:
    """a - b for long-axis yaws (deg, mod 180), in [-90, 90)."""
    return (a - b + 90.0) % 180.0 - 90.0


class Target:
    """One start from the states file: draws it and measures the live croissant and oven against it."""

    def __init__(self, sid: str, states: Path, geo: qc.Geometry):
        sys.path.insert(0, str(qc.VLA_DEFAULT / "scripts/real_eval"))
        import croissant_pose  # vla-benchmark: GrabCut croissant mask

        self.cp, self.sid, self.geo, self.model = croissant_pose, sid, geo, croissant_model()
        st = json.load(open(states))["states"][sid]
        self.c, self.o = st["croissant"], st["oven"]
        self.sil = silhouette(self.model, self.c["x"], self.c["y"], self.c["theta_deg"], geo)
        self.meas = None

    def measure(self, raw: np.ndarray, live: dict) -> None:
        m = self.cp.grabcut(raw, self.geo)
        cro = None
        if m is not None:
            x, y, th, s = fit_croissant(self.cp, m, self.model, self.geo)
            want = np.zeros(m.shape, np.uint8)
            cv2.fillPoly(want, [self.sil.round().astype(np.int32)], 1)
            cro = dict(x=x, y=y, theta_deg=th, fit_iou=s, target_iou=self.cp.mask_iou(m, want),
                       dx=x - self.c["x"], dy=y - self.c["y"], dtheta=yaw_err(th, self.c["theta_deg"]), mask=m)
        ov = live.get("oven")
        oven = None if ov is None else dict(x=ov["x"], y=ov["y"], match=ov["match"], dx=ov["x"] - self.o["x"],
                                            dy=ov["y"] - self.o["y"])
        self.meas = dict(croissant=cro, oven=oven)

    def draw(self, v: np.ndarray) -> None:
        g, c, z = self.geo, self.c, self.geo.table_z + qc.CRO_COM_Z
        f = qc.OVEN_FOOTPRINT
        box = (self.o["x"] + f[0], self.o["x"] + f[1], self.o["y"] + f[2], self.o["y"] + f[3])
        cv2.polylines(v, [qc.box_outline(box, g.table_z, g).round().astype(np.int32)], True, CYAN, 1, cv2.LINE_AA)
        cv2.polylines(v, [self.sil.round().astype(np.int32)], True, CYAN, 2, cv2.LINE_AA)
        a = axis_px(c["x"], c["y"], c["theta_deg"], z, g)
        cv2.line(v, tuple(a[0]), tuple(a[1]), CYAN, 1, cv2.LINE_AA)
        h = v.shape[0]
        cm, om = (self.meas or {}).get("croissant"), (self.meas or {}).get("oven")
        if cm is None or cm["fit_iou"] < MIN_FIT_IOU:
            why = "not segmented" if cm is None else f"segmentation poor (model IoU {cm['fit_iou']:.2f}): check by eye"
            text(v, f"{self.sid} croissant {why}", h - 70, YELLOW)
        else:
            ok = np.hypot(cm["dx"], cm["dy"]) <= CRO_TOL[0] and abs(cm["dtheta"]) <= CRO_TOL[1]
            cs, _ = cv2.findContours(cm["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(v, cs, -1, GREEN if ok else RED, 1, cv2.LINE_AA)
            a = axis_px(cm["x"], cm["y"], cm["theta_deg"], z, g)
            cv2.line(v, tuple(a[0]), tuple(a[1]), GREEN if ok else RED, 1, cv2.LINE_AA)
            text(v, f"{self.sid} croissant dx {cm['dx'] * 100:+.1f} dy {cm['dy'] * 100:+.1f} cm yaw {cm['dtheta']:+.0f} deg"
                    f" IoU {cm['target_iou']:.2f}", h - 70, GREEN if ok else RED)
        if om is None:
            text(v, f"{self.sid}: oven knobs not seen", h - 50, YELLOW)
        else:
            ok = max(abs(om["dx"]), abs(om["dy"])) <= OVEN_TOL
            text(v, f"{self.sid} oven dx {om['dx'] * 100:+.1f} dy {om['dy'] * 100:+.1f} cm", h - 50, GREEN if ok else RED)
        text(v, "cyan: target start  (m saves the measured start)", h - 30, CYAN, 0.4)

    def save(self, raw: np.ndarray, img: np.ndarray, calibration: str) -> Path:
        d = REPO / "data/parity"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        (d / "frames").mkdir(parents=True, exist_ok=True)
        for kind, im in (("raw", raw), ("annotated", img)):
            cv2.imwrite(str(d / "frames" / f"{self.sid}_{stamp}_{kind}.png"), im)
        cro = None if self.meas["croissant"] is None else {
            k: round(float(val), 4) for k, val in self.meas["croissant"].items() if k != "mask"}
        rec = dict(initial_state_id=self.sid, time=stamp, calibration=calibration, target=dict(croissant=self.c, oven=self.o),
                   croissant=cro, oven=self.meas["oven"], frame=f"frames/{self.sid}_{stamp}_raw.png")
        with open(d / "measured_states.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return d / "measured_states.jsonl"


def coverage(task: str, geo: qc.Geometry):
    """(successes per blue-box cell, successes per oven y bin) from the task's QC results; None if there are none."""
    f = REPO / "data/qc" / task / "qc_results.json"
    if not f.exists():
        return None
    results = json.load(open(f))
    return qc.coverage(results, geo)[0], qc.oven_coverage(results, geo)[0]


def text(v: np.ndarray, s: str, row: int, colour, scale: float = 0.5) -> None:
    org = (8, row)
    cv2.putText(v, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(v, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def annotate(ui: RecordUI, raw: np.ndarray, cov):
    """The page's scene overlay checked on this frame, the cells with no success yet and the readout; (image, live)."""
    geo, v = ui._geo, raw.copy()
    if cov is not None:  # under the outlines
        bx0, bx1, by0, by1 = geo.blue
        dx, dy = (bx1 - bx0) / qc.GRID_X, (by1 - by0) / qc.GRID_Y
        layer = v.copy()
        for i, j in zip(*np.nonzero(cov[0] == 0)):  # row 0 nearest the robot, column 0 the robot's left (+y)
            cell = (bx0 + i * dx, bx0 + (i + 1) * dx, by1 - (j + 1) * dy, by1 - j * dy)
            cv2.fillPoly(layer, [qc.box_outline(cell, geo.table_z, geo, n=2).round().astype(np.int32)], YELLOW)
        cv2.addWeighted(layer, 0.35, v, 0.65, 0, dst=v)
    live = ui._scene_overlay(raw, v, True)
    c, o = live["croissant"], live["oven"]
    if c is None:
        text(v, "croissant not seen", 20, YELLOW)
    else:
        why = [s for ok, s in ((c["in_blue"], "centre outside the blue box"), (c["in_white"], "may cross the white box"))
               if not ok]
        text(v, f"croissant x {c['x']:.3f} y {c['y']:+.3f}: {', '.join(why) or 'OK'}", 20, RED if why else GREEN)
    if o is None:
        text(v, "oven knobs not seen", 40, YELLOW)
    else:
        sim_x = (geo.oven_region[0] + geo.oven_region[1]) / 2
        why = [s for ok, s in ((o["x_ok"], f"x off the sim's {sim_x:.3f}"), (o["y_ok"], "y outside the sim's range"),
                               (o["in_white"], "crosses the white box")) if not ok]
        text(v, f"oven x {o['x']:.3f} y {o['y']:+.3f}: {', '.join(why) or 'OK'}", 40, RED if why else GREEN)
    if cov is not None:
        croissant_grid, oven_bins = cov
        here = ([("croissant cell", croissant_grid[tuple(c["cell"])])] if c and c["cell"] else []) + (
            [("oven y bin", oven_bins[o["cell"]])] if o and o["cell"] is not None else [])
        if here:
            text(v, "successful demos from here: " + ", ".join(f"{k} {m}" for k, m in here), 60,
                 YELLOW if min(m for _, m in here) == 0 else WHITE)
        oy = np.linspace(geo.oven_region[3], geo.oven_region[2], qc.OVEN_GRID + 1)
        empty = [f"{(oy[j] + oy[j + 1]) / 2:+.3f}" for j in range(qc.OVEN_GRID) if oven_bins[j] == 0]
        if empty:
            text(v, "oven y with no success yet: " + " ".join(empty), 80, YELLOW)
    text(v, "white: whole objects  blue: croissant centre  orange: oven range  yellow: no success yet",
         v.shape[0] - 10, WHITE, 0.4)
    return v, live


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", default="", help="save one annotated frame to this path and exit (no window)")
    ap.add_argument("--bag", default="", help="replay a recorded scene_camera.bag instead of the camera")
    ap.add_argument("--task", default="", help="optional task whose successful placement coverage is shown")
    ap.add_argument("--scale", type=float, default=1.5, help="window size relative to 640x480")
    ap.add_argument("--calibration", default=str(SCENE_CALIBRATION),
                    help="scene camera pose and K from this calibration_results-format file; '' for rig.json's")
    ap.add_argument("--state", default="", help="draw this start from --states and measure the placement against it")
    ap.add_argument("--states", default=str(STATES), help="initial-state file")
    args = ap.parse_args()

    ui = RecordUI(open_browser=False)  # not started: only its geometry and scene overlay
    if args.calibration:
        sc = json.load(open(Path(args.calibration).expanduser()))["cameras"]["scene_camera"]
        T = np.eye(4)
        T[:3, :3] = sc["extrinsics"]["rotation_matrix"]
        T[:3, 3] = np.ravel(sc["extrinsics"]["translation_vector"])
        ui._geo = dataclasses.replace(ui._geo, T=T, K=np.array(sc["intrinsics"]["camera_matrix"]))
    target = Target(args.state, Path(args.states), ui._geo) if args.state else None
    cfg = CameraConfig(CAMERA_CONFIG)
    if args.bag:
        cam = RealSenseCamera.from_bag(NAME, Path(args.bag), crop=cfg.get_crop(NAME))
    else:
        cam = cfg.create_camera(NAME)
        try:
            cam.open()
        except RuntimeError as e:
            raise SystemExit(f"the scene camera did not open ({e}). Is rd serve or rd record running?")
    win, out = "placement limits (q quits, s saves)", REPO / "data/placement_preview"
    cov, cov_t, n, t0, meas_t = None, 0.0, 0, time.monotonic(), 0.0
    try:
        while True:
            if not cam.grab():
                if args.bag:
                    break  # end of the bag
                continue
            n += 1
            if args.snapshot and not args.bag and time.monotonic() - t0 < 1.5:
                continue  # a live snapshot waits for the stream to settle
            if args.task and time.monotonic() - cov_t > 2.0:  # follows the recording page's rewrites
                cov, cov_t = coverage(args.task, ui._geo), time.monotonic()
            raw = cam.get_frame().color
            img, live = annotate(ui, raw, cov)
            if target is not None:
                if args.snapshot or time.monotonic() - meas_t > 0.3:
                    target.measure(raw, live)
                    meas_t = time.monotonic()
                target.draw(img)
            if args.snapshot:
                cv2.imwrite(args.snapshot, img)
                rate = "" if args.bag else f", {n / (time.monotonic() - t0):.0f} fps"
                print(f"  saved {args.snapshot} (frame {n}{rate}): croissant {live['croissant']}, oven {live['oven']}")
                if target is not None:
                    cm = target.meas["croissant"]
                    print(f"  {args.state}: croissant {None if cm is None else {k: round(float(v), 4) for k, v in cm.items() if k != 'mask'}}, oven {target.meas['oven']}")
                return
            cv2.imshow(win, cv2.resize(img, None, fx=args.scale, fy=args.scale))
            key = cv2.waitKey(33 if args.bag else 1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("s"):
                out.mkdir(parents=True, exist_ok=True)
                p = out / f"{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(str(p), img)
                print(f"  saved {p}")
            if key == ord("m") and target is not None and target.meas is not None:
                print(f"  measured start appended to {target.save(raw, img, args.calibration)}")
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
