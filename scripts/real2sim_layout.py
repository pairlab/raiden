#!/usr/bin/env python3
"""Derive the static scene layout for the digital clone from a calibrated capture.

Reads ``data/real2sim/calibration/rig.json`` and the home capture, and produces
``data/real2sim/calibration/layout.json`` plus a baked top-down table texture
(``table_atlas.png``) in the robot base frame:

* table: plane height and the extent seen by the scene camera (base frame)
* extrusion rail the arm is bolted to (from rig.json)
* up to two large vertical planes near the table (partition wall / screen): normal, offset,
  extent and mean colour; with ``--taped-walls`` the end partitions and the long far wall are
  rebuilt from the tape (inner face on the table edge, top and bottom height, thickness)
* table atlas: real table pixels re-projected to a metric top-down image (GPT6-real2sim
  "table_texture" technique) so the sim table shows the real wood grain

Everything is expressed in the robot base frame (z up, table at z = table_z).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from raiden.real2sim.geometry import (
    depth_to_points,
    fit_plane_ransac,
    inv_T,
    project,
    transform_points,
)


WALL_THICKNESS = 0.02  # a taped wall's box extends this far outward from its inner face


def height_out_of_view(wall: dict, cam: dict, table_z: float, margin=0.01):
    """Tallest wall (m above the table) whose top stays out of the camera's recorded frame; None if
    the camera sees no part of it at any height. The wall's plane is its inner face; its box extends
    ``thickness`` outward from it and starts ``bottom_above_table`` above the table."""
    n, u, d = (
        np.array(wall["normal_in_base"]),
        np.array(wall["u_axis_in_base"]),
        float(wall["offset"]),
    )
    K, (w, h) = np.array(cam["K"]), cam["resolution"]
    T_cam_base = inv_T(np.array(cam["T_base_cam"]))
    t, z = np.meshgrid(
        np.linspace(*wall["extent_along_u"], 100),
        np.arange(wall.get("bottom_above_table", 0.0), 1.5, 0.005),
    )
    lowest = np.inf
    for s in (
        0.0,
        -wall.get("thickness", WALL_THICKNESS),
    ):  # inner and outer face of the box
        P = (s - d) * n + t.reshape(-1, 1) * u
        P[:, 2] = table_z + z.ravel()
        uv, depth = project(K, transform_points(T_cam_base, P))
        seen = (
            (depth > 0)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < w)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < h)
        )
        if seen.any():
            lowest = min(lowest, float(z.ravel()[seen].min()))
    return None if np.isinf(lowest) else float(np.floor((lowest - margin) * 100) / 100)


def taped_walls(
    walls: list[dict],
    table_x: list[float],
    table_y: list[float],
    top: float,
    gap: float,
    cam: dict,
    table_z: float,
) -> list[dict]:
    """Raiden lab, from the tape: identical partitions stand at both table ends and a long wall runs
    along the far side. All three reach ``top`` above the table and leave an open ``gap`` between the
    table top and their bottom edge; their inner faces sit on the table edges, square to the table,
    and they join at the corners.

    Each wall's plane is its inner face and its box extends WALL_THICKNESS outward. The long wall
    spans the table length plus both partitions; the partitions run from the near end of the one the
    scene camera sees to the long wall's outer face. Colours come from the detected planes.
    """
    walls = [dict(w) for w in walls]
    ends = [
        i
        for i, w in enumerate(walls)
        if abs(w["normal_in_base"][1]) > 0.9 and abs(w["center_in_base"][1]) > 0.45
    ]
    sides = [
        i
        for i, w in enumerate(walls)
        if abs(w["normal_in_base"][0]) > 0.9 and w["center_in_base"][0] > 0.45
    ]
    if len(ends) != 1 or len(sides) != 1:
        raise SystemExit(
            f"--taped-walls: expected one end partition and one long wall in view, "
            f"found {len(ends)} and {len(sides)}"
        )

    def box(like: dict, normal_xy, a_xy, b_xy, source: str) -> dict:
        """Wall whose inner face runs from a to b (base xy), with its normal into the table."""
        n = np.array([*normal_xy, 0.0])
        u = np.cross(n, [0.0, 0.0, 1.0])
        a, b = np.array([*a_xy, 0.0]), np.array([*b_xy, 0.0])
        return dict(
            like,
            normal_in_base=n.tolist(),
            offset=float(-n @ a),
            u_axis_in_base=u.tolist(),
            extent_along_u=sorted([float(u @ a), float(u @ b)]),
            center_in_base=[*((a + b)[:2] / 2).tolist(), table_z + (gap + top) / 2],
            z_extent=[table_z + gap, table_z + top],
            top_above_table=top,
            bottom_above_table=gap,
            thickness=WALL_THICKNESS,
            source=source,
        )

    seen = walls[ends[0]]
    n, u, d = (
        np.array(seen["normal_in_base"]),
        np.array(seen["u_axis_in_base"]),
        float(seen["offset"]),
    )
    x0 = min(float((-d * n + t * u)[0]) for t in seen["extent_along_u"])  # its near end
    x1 = table_x[1] + WALL_THICKNESS  # the long wall's outer face
    near = min(table_y, key=lambda y: abs(y - seen["center_in_base"][1]))
    for y_end in table_y:
        s = (
            1.0 if y_end < np.mean(table_y) else -1.0
        )  # normal into the table, towards the base
        if y_end == near:
            walls[ends[0]] = box(
                seen,
                (0.0, s),
                (x0, y_end),
                (x1, y_end),
                "taped end partition; colour and near end from the scene camera",
            )
        else:
            close = box(
                dict(seen, n_points=0),
                (0.0, s),
                (x0, y_end),
                (x1, y_end),
                "taped end partition, same as the seen one; the scene camera capture has none of it",
            )
            cap = height_out_of_view(close, cam, table_z)
            print(
                f"partition at the y={y_end:+.3f} table end: the calibrated scene camera "
                + (
                    f"sees its top {top - cap:.2f} m (a wall up to {cap:.2f} m above the table would stay out "
                    f"of view)"
                    if cap is not None and cap < top
                    else "does not see it"
                )
            )
            walls.append(close)
    walls[sides[0]] = box(
        walls[sides[0]],
        (-1.0, 0.0),
        (table_x[1], table_y[0] - WALL_THICKNESS),
        (table_x[1], table_y[1] + WALL_THICKNESS),
        "taped long wall; colour from the scene camera",
    )
    return walls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default="data/real2sim/captures/home")
    ap.add_argument("--calib", default="data/real2sim/calibration")
    ap.add_argument(
        "--atlas-res",
        type=int,
        default=1000,
        help="pixels per metre of the table atlas",
    )
    ap.add_argument(
        "--table-x-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("X0", "X1"),
        help="physical table extent in base x (m); default: seen extent + under the camera",
    )
    ap.add_argument(
        "--table-y-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("Y0", "Y1"),
        help="physical table extent in base y (m)",
    )
    ap.add_argument(
        "--rail-y-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("Y0", "Y1"),
        help="rail ends in base y (m); default: the whole table length",
    )
    ap.add_argument(
        "--taped-walls",
        type=float,
        nargs=2,
        default=None,
        metavar=("TOP", "GAP"),
        help="identical partitions at both table ends and a long wall along the far side, with their "
        "top TOP and bottom edge GAP above the table (m) and inner faces on the table edges",
    )
    args = ap.parse_args()
    if args.taped_walls and not (args.table_x_range and args.table_y_range):
        ap.error(
            "--taped-walls puts the walls on the taped --table-x-range and --table-y-range edges"
        )
    if args.taped_walls and not 0.0 <= args.taped_walls[1] < args.taped_walls[0]:
        ap.error("--taped-walls needs 0 <= GAP < TOP")
    cap, cal = Path(args.capture), Path(args.calib)
    rig = json.load(open(cal / "rig.json"))
    sc = rig["cameras"]["scene_camera"]
    # rig.json's K is for the recorded frame; the capture image has its own (wider) intrinsics.
    meta = json.load(open(cap / "meta.json"))["scene_camera"]
    K = np.array([[meta["fx"], 0, meta["cx"]], [0, meta["fy"], meta["cy"]], [0, 0, 1]])
    T_base_cam = np.array(sc["T_base_cam"])
    T_cam_base = inv_T(T_base_cam)
    color = cv2.imread(str(cap / "scene_camera_color.png"))
    depth = np.load(cap / "scene_camera_depth_m.npy")
    h, w = depth.shape
    P = depth_to_points(depth, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    valid = (depth > 0.15) & (depth < 3.0)
    table_z = rig["table"]["z_in_base_at_origin"]

    # ---- table plane inliers in the base frame -------------------------------------------
    Pb = transform_points(T_base_cam, P.reshape(-1, 3)).reshape(h, w, 3)
    on_table = valid & (np.abs(Pb[:, :, 2] - table_z) < 0.008)
    tp = Pb[on_table]
    ext = {
        "x": [
            float(np.percentile(tp[:, 0], 0.5)),
            float(np.percentile(tp[:, 0], 99.5)),
        ],
        "y": [
            float(np.percentile(tp[:, 1], 0.5)),
            float(np.percentile(tp[:, 1], 99.5)),
        ],
    }
    # The scene camera is clamped to the table edge, so the table extends at least under it.
    cam_xy = T_base_cam[:2, 3]
    ext["x"] = [
        min(ext["x"][0], float(cam_xy[0]) - 0.15),
        max(ext["x"][1], float(cam_xy[0]) + 0.15),
    ]
    ext["y"] = [
        min(ext["y"][0], float(cam_xy[1]) - 0.15),
        max(ext["y"][1], float(cam_xy[1]) + 0.15),
    ]
    if args.table_x_range:
        ext["x"] = [float(v) for v in args.table_x_range]
    if args.table_y_range:
        ext["y"] = [float(v) for v in args.table_y_range]
    print(
        f"table z={table_z:+.3f}, extent x {ext['x']}, y {ext['y']}"
        f"{' (given)' if args.table_x_range or args.table_y_range else ' (seen + under camera)'}"
    )

    # ---- vertical planes (walls / screens) ------------------------------------------------
    walls = []
    cand = valid & (Pb[:, :, 2] > table_z + 0.05)
    idx = np.flatnonzero(cand)
    pts = Pb.reshape(-1, 3)[idx]
    cols = color.reshape(-1, 3)[idx]
    for i in range(3):
        if len(pts) < 5000:
            break
        n2, d2, inl = fit_plane_ransac(pts, thresh=0.015, iters=1500)
        vertical = abs(n2[2]) < 0.25
        frac = inl.mean()
        if not vertical or inl.sum() < 4000:
            # skip this plane but keep looking
            pts, cols = pts[~inl], cols[~inl]
            continue
        # orient the normal to point towards the robot base (origin side)
        if d2 < 0:
            n2, d2 = -n2, -d2
        Q = pts[inl]
        # in-plane axes
        u = np.cross(n2, [0, 0, 1.0])
        u /= np.linalg.norm(u)
        walls.append(
            {
                "normal_in_base": n2.tolist(),
                "offset": float(d2),  # n . p + d = 0
                "center_in_base": Q.mean(0).tolist(),
                "extent_along_u": [
                    float(np.percentile(Q @ u, 1)),
                    float(np.percentile(Q @ u, 99)),
                ],
                "u_axis_in_base": u.tolist(),
                "z_extent": [
                    float(np.percentile(Q[:, 2], 1)),
                    float(np.percentile(Q[:, 2], 99)),
                ],
                "mean_rgb": (cols[inl].mean(0)[::-1] / 255.0).tolist(),
                "n_points": int(inl.sum()),
            }
        )
        print(
            f"wall {len(walls)}: n={np.round(n2, 3)} d={d2:.3f} pts={inl.sum()} rgb={np.round(cols[inl].mean(0)[::-1])}"
        )
        pts, cols = pts[~inl], cols[~inl]
    if args.taped_walls:
        walls = taped_walls(walls, ext["x"], ext["y"], *args.taped_walls, sc, table_z)
        for w in walls:
            if "source" in w:
                print(
                    f"wall at {np.round(w['center_in_base'][:2], 3)}, {w['bottom_above_table']:.3f}-"
                    f"{w['top_above_table']:.3f} m above the table: {w['source']}"
                )

    # ---- table atlas -----------------------------------------------------------------------
    res = args.atlas_res
    x0, x1 = ext["x"]
    y0, y1 = ext["y"]
    xs = np.arange(x0, x1, 1.0 / res)
    ys = np.arange(y1, y0, -1.0 / res)  # image rows go from +y (top) to -y (bottom)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, table_z)], 1)
    uv, z = project(K, transform_points(T_cam_base, grid))
    uv = uv.reshape(gy.shape + (2,)).astype(np.float32)
    atlas = cv2.remap(
        color, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    # valid where the pixel is on the table plane (not the robot / rail / objects)
    valid_tab = (
        cv2.remap(on_table.astype(np.uint8), uv[..., 0], uv[..., 1], cv2.INTER_NEAREST)
        > 0
    )
    valid_tab = cv2.erode(valid_tab.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    holes = (~valid_tab).astype(np.uint8) * 255
    # fill holes with the median wood colour then inpaint for continuity
    med = np.median(atlas[valid_tab], axis=0)
    atlas[~valid_tab] = med
    atlas = cv2.inpaint(atlas, holes, 7, cv2.INPAINT_TELEA)
    cv2.imwrite(str(cal / "table_atlas.png"), atlas)
    print(
        f"atlas {atlas.shape[1]}x{atlas.shape[0]} px, unseen fraction {1 - valid_tab.mean():.2f}, median wood BGR {med}"
    )

    layout = {
        "frame": "robot base (z up)",
        "table": {
            "z_in_base": table_z,
            "seen_extent_in_base": ext,
            "atlas": {
                "file": "table_atlas.png",
                "x_range": [x0, x1],
                "y_range": [y0, y1],
                "pixels_per_m": res,
                "row0_is_ymax": True,
            },
            "median_rgb": (med[::-1] / 255.0).tolist(),
        },
        "rail": dict(
            rig["rail"],
            **(
                {"y_range": [float(v) for v in args.rail_y_range]}
                if args.rail_y_range
                else {}
            ),
        ),
        "walls": walls,
        "robot_base_height_above_table": -table_z,
    }
    json.dump(layout, open(cal / "layout.json", "w"), indent=1)
    print(f"wrote {cal / 'layout.json'}")


if __name__ == "__main__":
    main()
