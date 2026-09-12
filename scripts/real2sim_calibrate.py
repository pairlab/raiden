#!/usr/bin/env python3
"""Calibrate the real rig for the MESA digital clone using RealSense depth only (no board).

Inputs: a capture directory produced by ``scripts/real2sim_capture.py`` (color + metric depth +
intrinsics for ``scene_camera`` and ``left_wrist_camera``) taken with the arm resting at the
raiden home pose (all joints 0) and nothing on the table.

Steps
-----
1. Scene camera: RANSAC the table plane from depth. Then align the i2rt YAM mesh (at the
   recorded joints) to the depth points above the table with trimmed ICP. This yields
   ``T_base_scenecam`` (camera pose in the robot base frame) and the table height in the base
   frame. Yaw ambiguity is resolved by trying several initialisations and keeping the best fit.
2. Wrist camera: RANSAC the table plane in the wrist depth. Given the FK of the tool frame at
   the recorded joints and the table plane in the base frame, solve the hand-eye transform's
   pitch/roll and height from the plane; the remaining lateral/forward offsets and yaw come from
   a CAD-style prior (camera centred above the gripper, looking between the fingertips).
3. Rail: measure the aluminium extrusion the arm is bolted to (height above table, width) from
   the depth band just above the table around the base.

Outputs ``<capture>/../../calibration/rig.json`` plus overlay images for visual checking.
All transforms are cam2world with world = robot base frame (raiden's extrinsics convention).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from raiden._xml_paths import get_yam_4310_linear_xml_path
from raiden.real2sim.geometry import (
    CV_TO_MJ,
    depth_to_points,
    fit_plane_ransac,
    icp,
    inv_T,
    make_T,
    project,
    rotation_aligning,
    transform_points,
)
from raiden.real2sim.silhouette import (
    SilhouetteRenderer,
    fit_silhouette,
    fit_wrist_offsets,
)


def load_capture(cap: Path, name: str):
    meta = json.load(open(cap / "meta.json"))[name]
    color = cv2.imread(str(cap / f"{name}_color.png"))
    depth = np.load(cap / f"{name}_depth_m.npy")
    K = np.array([[meta["fx"], 0, meta["cx"]], [0, meta["fy"], meta["cy"]], [0, 0, 1]])
    return color, depth, K, meta


def draw_points(img, uv, color, r=1):
    out = img.copy()
    h, w = img.shape[:2]
    uv = np.round(uv).astype(int)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    for u, v in uv[ok]:
        cv2.circle(out, (int(u), int(v)), r, color, -1)
    return out


def measure_rail(P, height, valid, depth, T_base_cam, n, d, verbose=True):
    """Extrusion rail from the scene depth: top height above the table, width, direction.

    Points 1-4.5 cm above the table plane, near the base x axis and on the camera side of the
    base (base y < -0.13, which excludes the base foot and mounting plate) are fitted with a
    line (PCA, two rounds of outlier rejection). Returns (info dict in base frame, unit
    direction of the rail in the *camera* frame) or ({}, None) when too few points are found.
    """
    band = valid & (height > 0.010) & (height < 0.045) & (depth < 1.8)
    Pc = P[band]
    pb = transform_points(T_base_cam, Pc)
    sel = (np.abs(pb[:, 0]) < 0.25) & (pb[:, 1] < -0.13) & (pb[:, 1] > -0.9)
    if sel.sum() < 200:
        return {}, None
    idx = np.flatnonzero(sel)
    for _ in range(2):
        q = pb[idx, :2]
        c = q.mean(0)
        _, _, vt = np.linalg.svd(q - c, full_matrices=False)
        u = vt[0]
        dist = np.abs((q - c) @ np.array([-u[1], u[0]]))
        idx = idx[dist < 0.06]
        if len(idx) < 200:
            return {}, None
    q = pb[idx]
    c = q[:, :2].mean(0)
    _, _, vt = np.linalg.svd(q[:, :2] - c, full_matrices=False)
    u = vt[0]
    if u[1] > 0:
        u = -u
    across = (q[:, :2] - c) @ np.array([-u[1], u[0]])
    # direction in the camera frame (3D PCA of the same points)
    pc = Pc[idx]
    _, _, vtc = np.linalg.svd(pc - pc.mean(0), full_matrices=False)
    dir_cam = vtc[0] / np.linalg.norm(vtc[0])
    h = height[band][idx]
    info = {
        "height_above_table": float(np.percentile(h, 90)),
        "width": float(np.percentile(across, 97) - np.percentile(across, 3)),
        "x_at_base_y0": float(c[0] - u[0] * c[1] / u[1])
        if abs(u[1]) > 1e-6
        else float(c[0]),
        "yaw_from_base_y_deg": float(np.degrees(np.arctan2(u[0], -u[1]))),
        "y_seen_extent_in_base": [
            float(np.percentile(q[:, 1], 2)),
            float(np.percentile(q[:, 1], 98)),
        ],
        "n_points": int(len(idx)),
    }
    if verbose:
        print(
            f"[scene] rail: top {info['height_above_table']:.3f} m above table, width {info['width']:.3f} m, "
            f"x at base {info['x_at_base_y0']:+.3f}, yaw from base y {info['yaw_from_base_y_deg']:+.1f} deg "
            f"({info['n_points']} pts)"
        )
    return info, dir_cam


# --------------------------------------------------------------------------------------
# Scene camera
# --------------------------------------------------------------------------------------
def calibrate_scene(
    cap: Path,
    joints: np.ndarray,
    out_dir: Path,
    verbose=True,
    dark_v_max=80,
    roi_xyxy=(0, 115, 365, 340),
    max_arm_depth=1.15,
    landmarks=None,
    rail_constraint=False,
    z_fixed=None,
    rail_pixels=None,
    rail_height=0.02,
):
    color, depth, K, meta = load_capture(cap, "scene_camera")
    P = depth_to_points(depth, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    valid = (depth > 0.15) & (depth < 2.5)
    n, d, inl_flat = fit_plane_ransac(P[valid], thresh=0.006)
    inl = np.zeros(depth.shape, bool)
    inl[valid] = inl_flat
    height = P @ n + d  # signed height above table (camera on + side)
    if verbose:
        print(
            f"[scene] table plane: {inl.sum()} inliers, camera height above table {d:.3f} m, "
            f"tilt below horizontal {np.degrees(np.arcsin(-n[2])):.1f} deg"
        )

    # Robot mask from colour: the YAM is matte black. Restrict to a region of interest that
    # excludes the dark partition wall (same depth as the arm, so depth cannot separate them).
    # The YAM's links 2/3 are bare aluminium; the motors, wrist and gripper are matte black.
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    near = (depth < max_arm_depth) & valid
    # black parts often return no depth; accept depth-less dark pixels only next to near geometry
    near_dil = cv2.dilate(near.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(
        bool
    )
    dark = (hsv[:, :, 2] < dark_v_max) & (near | (~valid & near_dil))
    silver = (hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 120) & near & (height > 0.07)
    roi = np.zeros(depth.shape, bool)
    x0, y0, x1, y1 = roi_xyxy
    roi[y0:y1, x0:x1] = True
    # Bright, unsaturated screens/partitions near the arm: drop the dominant vertical plane(s)
    # from the silver candidates (the links are not planar at this scale).
    for _ in range(2):
        idx = np.flatnonzero(silver & roi)
        if len(idx) < 2000:
            break
        n2, _, inl2 = fit_plane_ransac(P.reshape(-1, 3)[idx], thresh=0.012, iters=1000)
        if inl2.mean() < 0.3 or np.degrees(np.arccos(abs(n2 @ n))) < 60:
            break
        silver.reshape(-1)[idx[inl2]] = False
    target = (dark | silver) & roi
    target = cv2.morphologyEx(
        target.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    ).astype(bool)
    cv2.imwrite(str(out_dir / "scene_arm_mask.png"), target.astype(np.uint8) * 255)

    # Initial in-plane position: back-project the mask centroid onto a plane 12 cm above the table.
    ys, xs = np.nonzero(target)
    ray = np.linalg.inv(K) @ np.array([xs.mean(), ys.mean(), 1.0])
    s_ = (0.12 - d) / (n @ ray)
    c0 = ray * s_
    ex0 = np.cross(n, [0.0, 0.0, 1.0])
    ex0 /= np.linalg.norm(ex0)
    ey0 = np.cross(n, ex0)
    foot = -d * n
    init_xy = ((c0 - foot) @ ex0, (c0 - foot) @ ey0)

    renderer = SilhouetteRenderer(
        get_yam_4310_linear_xml_path(), K, depth.shape[1], depth.shape[0], joints
    )
    T_cam_base, iou_val, params = fit_silhouette(
        renderer,
        target,
        roi,
        n,
        d,
        init_xy,
        z_init=0.08,
        verbose=verbose,
        depth=depth,
        landmarks=landmarks,
        z_fixed=z_fixed,
    )
    # The arm silhouette at home is nearly symmetric, so yaw is weakly determined. The extrusion
    # rail the base is bolted to is long and well measured by depth: its direction must be the
    # base y axis. Measure it with the current fit, refit with that constraint, repeat once.
    # NOTE: on this rig the measured rail runs ~25 deg off the model's base y axis while the
    # arm silhouette fits well, i.e. the base is mounted at an angle on the rail (or joint 1
    # has a zero offset). The constraint is therefore opt-in; the measured rail yaw is stored
    # in rig.json and the sim places the rail at that angle.
    rail_info, rail_dir_cam = measure_rail(
        P, height, valid, depth, inv_T(T_cam_base), n, d, verbose=verbose
    )
    if rail_pixels is not None:
        # Depth on the shiny extrusion is unreliable; take the rail direction from two pixels on
        # one of its top edges, back-projected onto the plane rail_height above the table.
        pts = []
        for u, v in np.asarray(rail_pixels, float).reshape(2, 2):
            ray = np.linalg.inv(K) @ np.array([u, v, 1.0])
            s_ = (rail_height - d) / (n @ ray)  # n . (s ray) + d = rail_height
            pts.append(ray * s_)
        rail_dir_cam = pts[1] - pts[0]
        rail_dir_cam /= np.linalg.norm(rail_dir_cam)
        y_cam = T_cam_base[:3, :3] @ np.array([0.0, 1.0, 0.0])
        ang = np.degrees(np.arccos(np.clip(abs(y_cam @ rail_dir_cam), 0, 1)))
        rail_info = dict(
            rail_info or {},
            yaw_from_base_y_deg_image=float(ang),
            height_above_table=rail_height,
        )
        if verbose:
            print(
                f"[scene] rail direction from image edge pixels; angle to fitted base y = {ang:.1f} deg"
            )
    if rail_constraint and rail_dir_cam is not None:
        for it in range(2):
            if verbose:
                print(
                    f"[scene] rail pass {it + 1}: refit with rail direction constraint"
                )
            T_cam_base, iou_val, params = fit_silhouette(
                renderer,
                target,
                roi,
                n,
                d,
                init_xy,
                verbose=verbose,
                depth=depth,
                landmarks=landmarks,
                rail_dir_cam=rail_dir_cam,
                init_params=params,
                z_fixed=z_fixed,
            )
            if rail_pixels is None:
                rail_info, rail_dir_cam = measure_rail(
                    P, height, valid, depth, inv_T(T_cam_base), n, d, verbose=verbose
                )
            else:
                y_cam = T_cam_base[:3, :3] @ np.array([0.0, 1.0, 0.0])
                rail_info["yaw_from_base_y_deg_image"] = float(
                    np.degrees(np.arccos(np.clip(abs(y_cam @ rail_dir_cam), 0, 1)))
                )
                if verbose:
                    print(
                        f"[scene] rail (image) angle to base y now {rail_info['yaw_from_base_y_deg_image']:.1f} deg"
                    )
    base_h = T_cam_base[:3, 3] @ n + d
    T_base_cam = inv_T(T_cam_base)
    if rail_pixels is not None:
        # rail line (one top edge) in the base frame -> yaw and centreline x offset for the sim
        pb_edge = transform_points(T_base_cam, np.array(pts))
        u = pb_edge[1, :2] - pb_edge[0, :2]
        u /= np.linalg.norm(u)
        if u[1] < 0:
            u = -u
        yaw = float(np.arctan2(u[0], u[1]))
        c = pb_edge[0, :2]
        x_edge_at_y0 = (
            float(c[0] - u[0] * c[1] / u[1]) if abs(u[1]) > 1e-6 else float(c[0])
        )
        half_w = 0.04  # 80 mm profile; the centreline is half a width from the edge, towards the base
        x_centre = x_edge_at_y0 - np.sign(x_edge_at_y0) * half_w / np.cos(yaw)
        rail_info.update(
            {
                "source": "image edge pixels + table plane",
                "height_above_table": rail_height,
                "width": 2 * half_w,
                "x_at_base_y0": float(x_centre),
                "yaw_from_base_y_deg": float(np.degrees(yaw)),
            }
        )
        if verbose:
            print(
                f"[scene] rail (image): yaw from base y {np.degrees(yaw):+.1f} deg, edge x at base {x_edge_at_y0:+.3f}, "
                f"centre x {x_centre:+.3f}"
            )
    rms, nk = float("nan"), 0
    if verbose:
        print(
            f"[scene] silhouette IoU {iou_val:.3f}; base origin {base_h:+.3f} m above table"
        )

    # Table plane in base frame
    n_base = T_base_cam[:3, :3] @ n
    d_base = d - n_base @ T_base_cam[:3, 3]  # n_b . p_b + d_b = 0
    table_z_in_base = -d_base / n_base[2]  # height of table at base x=y=0
    if verbose:
        print(
            f"[scene] table normal in base frame {np.round(n_base, 4)}, table z at base origin {table_z_in_base:+.4f}"
        )

    if rail_info:
        rail_info["top_z_in_base"] = float(
            table_z_in_base + rail_info["height_above_table"]
        )

    # Table extent seen by the camera (in base frame), useful for sizing the sim table.
    tp = transform_points(T_base_cam, P[inl])
    table_extent = {
        "x": [float(np.percentile(tp[:, 0], 1)), float(np.percentile(tp[:, 0], 99))],
        "y": [float(np.percentile(tp[:, 1], 1)), float(np.percentile(tp[:, 1], 99))],
    }

    # Overlay for visual check: projected model points (green) and table inliers (blue tint).
    sil = renderer.render_mask(T_cam_base)
    ov = color.copy()
    ov[sil] = (0.5 * ov[sil] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    ov[inl] = (0.6 * ov[inl] + 0.4 * np.array([255, 120, 0])).astype(np.uint8)
    # base axes
    axes = np.array([[0, 0, 0], [0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]])
    uva, _ = project(K, transform_points(T_cam_base, axes))
    for i, c in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
        cv2.line(
            ov,
            tuple(np.round(uva[0]).astype(int)),
            tuple(np.round(uva[i + 1]).astype(int)),
            c,
            2,
        )
    cv2.imwrite(str(out_dir / "scene_overlay.png"), ov)

    R_mj = T_base_cam[:3, :3] @ CV_TO_MJ
    return (
        {
            "serial": meta["serial"],
            "resolution": [meta["width"], meta["height"]],
            "K": K.tolist(),
            "T_base_cam": T_base_cam.tolist(),
            "mujoco": {
                "pos": T_base_cam[:3, 3].tolist(),
                "quat_wxyz": np.roll(Rotation.from_matrix(R_mj).as_quat(), 1).tolist(),
                "fovy_deg": float(
                    2 * np.degrees(np.arctan2(meta["height"] / 2, K[1, 1]))
                ),
            },
            "fit": {
                "silhouette_iou": float(iou_val),
                "plane_inliers": int(inl.sum()),
                "base_height_above_table_m": float(base_h),
            },
            "table": {
                "normal_in_base": n_base.tolist(),
                "z_in_base_at_origin": float(table_z_in_base),
                "seen_extent_in_base": table_extent,
            },
            "rail": rail_info,
        },
        T_base_cam,
        (n_base, d_base),
    )


# --------------------------------------------------------------------------------------
# Wrist camera
# --------------------------------------------------------------------------------------
def calibrate_wrist(
    cap: Path,
    joints: np.ndarray,
    plane_base,
    out_dir: Path,
    prior_forward=0.03,
    prior_lateral=0.0,
    verbose=True,
    gripper=0.0475,
    dark_v_max=80,
):
    color, depth, K, meta = load_capture(cap, "left_wrist_camera")
    P = depth_to_points(depth, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    valid = (depth > 0.1) & (depth < 1.0)
    n_c, d_c, inl_flat = fit_plane_ransac(P[valid], thresh=0.004)
    inl = np.zeros(depth.shape, bool)
    inl[valid] = inl_flat
    if verbose:
        print(
            f"[wrist] table plane: {inl.sum()} inliers, camera height above table {d_c:.3f} m, "
            f"optical axis {np.degrees(np.arcsin(-n_c[2])):.1f} deg below horizontal"
        )

    # Tool frame at the recorded joints: i2rt tcp_site (at joint-6 origin, +z along the gripper).
    m = mujoco.MjModel.from_xml_path(get_yam_4310_linear_xml_path())
    dat = mujoco.MjData(m)
    dat.qpos[:6] = joints
    mujoco.mj_forward(m, dat)
    sid = m.site("tcp_site").id
    T_base_tool = make_T(dat.site_xmat[sid].reshape(3, 3), dat.site_xpos[sid].copy())
    T_tool_base = inv_T(T_base_tool)

    n_b, d_b = plane_base
    # plane in tool frame
    n_t = T_tool_base[:3, :3] @ n_b
    d_t = d_b - n_t @ T_tool_base[:3, 3]
    up_t = n_t  # world up expressed in tool frame
    fwd_t = np.array([0.0, 0.0, 1.0])
    lat_t = np.cross(fwd_t, up_t)  # right-handed with (x=lat, y=-up, z=fwd)
    lat_t /= np.linalg.norm(lat_t)

    # Prior orientation: camera x = lateral, camera z = forward pitched down; we only need a
    # starting point, pitch/roll get replaced by the plane constraint below. Cam y comes from the
    # cross product: the tool axis is not parallel to the table (4.5 deg off at home), so -up_t is
    # not perpendicular to fwd_t, and stacking it made R_prior and every rotation derived from it
    # skewed (cam y and z 85.5 deg apart in the 2026-09-09 rig.json).
    R_prior = np.stack(
        [lat_t, np.cross(fwd_t, lat_t), fwd_t], 1
    )  # cam x=lateral, y=down, z=forward
    # Plane constraint: R_tool_cam @ n_c must equal n_t.
    R_tool_cam = rotation_aligning(R_prior @ n_c, n_t) @ R_prior
    # Position: forward/lateral offsets in the table plane and yaw about the table normal are
    # fitted to the silhouette of the black fingers; height comes from the plane distance.
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    target = hsv[:, :, 2] < dark_v_max
    target = cv2.morphologyEx(
        target.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)
    ).astype(bool)
    renderer = SilhouetteRenderer(
        get_yam_4310_linear_xml_path(),
        K,
        depth.shape[1],
        depth.shape[0],
        joints,
        gripper=gripper,
    )
    H, fit_iou, (fwd_m, lat_m, yaw_rad, dz) = fit_wrist_offsets(
        renderer,
        target,
        T_base_tool,
        R_tool_cam,
        up_t,
        fwd_t,
        lat_t,
        d_t,
        d_c,
        init=(prior_forward, prior_lateral, 0.0),
        verbose=verbose,
    )
    # dz > 0: the camera had to move away from the table (up) to match the finger scale, i.e.
    # the table is closer to the tool than the scene fit says: the base sits *lower* above
    # the table by dz (verified: the fitted tool->camera transform is independent of dz).
    base_height_correction = float(-dz)
    R_tool_cam = H[:3, :3]
    # Renderers silently use the nearest rotation, the converter and export use the matrix as is.
    assert np.allclose(R_tool_cam.T @ R_tool_cam, np.eye(3), atol=1e-6), (
        "wrist rotation is not orthonormal"
    )
    T_base_cam = T_base_tool @ H

    # Overlay: table inliers coloured, rendered fingers in green.
    ov = color.copy()
    ov[inl] = (0.6 * ov[inl] + 0.4 * np.array([255, 120, 0])).astype(np.uint8)
    sil = renderer.render_mask(inv_T(T_base_cam))
    ov[sil] = (0.5 * ov[sil] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    cv2.imwrite(str(out_dir / "wrist_overlay.png"), ov)

    R_mj = R_tool_cam @ CV_TO_MJ
    if verbose:
        print(
            f"[wrist] hand-eye (tool->cam) t = {np.round(H[:3, 3], 4)} "
            f"pitch below tool z = {np.degrees(np.arccos(np.clip(R_tool_cam[:, 2] @ fwd_t, -1, 1))):.1f} deg"
        )
    return {
        "serial": meta["serial"],
        "resolution": [meta["width"], meta["height"]],
        "K": K.tolist(),
        "tool_frame": "i2rt tcp_site (joint-6 origin, +z along gripper)",
        "T_tool_cam": H.tolist(),
        "T_base_cam_at_capture": T_base_cam.tolist(),
        "mujoco_in_tool_frame": {
            "pos": H[:3, 3].tolist(),
            "quat_wxyz": np.roll(Rotation.from_matrix(R_mj).as_quat(), 1).tolist(),
            "fovy_deg": float(2 * np.degrees(np.arctan2(meta["height"] / 2, K[1, 1]))),
        },
        "fit": {
            "plane_inliers": int(inl.sum()),
            "camera_height_above_table_m": float(d_c),
            "finger_silhouette_iou": float(fit_iou),
            "base_height_correction_m": base_height_correction,
            "offsets": {
                "forward_m": float(fwd_m),
                "lateral_m": float(lat_m),
                "yaw_deg": float(np.degrees(yaw_rad)),
                "note": "fitted to the finger silhouette; pitch/roll/height from the table plane",
            },
        },
    }


def to_recorded_frame(cam: dict, name: str, cfg) -> dict:
    """Re-express a camera fitted on the capture in the frame the recorder stores.

    Captures stream at ``real2sim_capture.py --resolution`` (848x480, the full D435 width);
    recordings stream at camera.json's ``resolution`` (640x480 unless set) and then apply its
    ``crop``. A D435 colour stream is a horizontal centre crop of a wider one of the same height
    (640x480 = 848x480 minus 104 px each side: fx, fy, cy equal), so the recorded frame is a
    fixed window of the capture and only the principal point moves; the pose is unchanged.
    ``capture.window`` keeps that window (capture pixels) for tools that reuse the capture.
    """
    stream = cfg.get_resolution(name)
    if "capture" in cam or stream is None:
        if stream is None:
            print(
                f"[{name}] no RealSense entry in {cfg.config_file}: rig.json keeps the capture frame"
            )
        return cam
    (cw, ch), (sw, sh) = cam["resolution"], stream
    if sh != ch or (cw - sw) % 2:
        raise SystemExit(
            f"[{name}] the {sw}x{sh} recording stream is not a centre crop of the {cw}x{ch} "
            f"capture; re-capture with real2sim_capture.py --resolution {sw} {sh}"
        )
    x, y, w, h = cfg.get_crop(name) or (0, 0, sw, sh)
    x += (cw - sw) // 2
    K = np.array(cam["K"], dtype=float)
    K[0, 2] -= x
    K[1, 2] -= y
    cam.update(
        resolution=[w, h],
        K=K.tolist(),
        capture={"resolution": [cw, ch], "window": [x, y, w, h]},
    )
    mj = cam["mujoco"] if "mujoco" in cam else cam["mujoco_in_tool_frame"]
    mj["fovy_deg"] = float(2 * np.degrees(np.arctan2(h / 2, K[1, 1])))
    return cam


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--capture", default="data/real2sim/captures/home")
    ap.add_argument("--out", default="data/real2sim/calibration")
    ap.add_argument(
        "--joints",
        type=float,
        nargs=6,
        default=None,
        help="arm joint angles (rad) at capture time (default: meta.json robot_joints_measured, else zeros)",
    )
    ap.add_argument(
        "--wrist-forward", type=float, default=0.03, help="initial guess only (fitted)"
    )
    ap.add_argument("--wrist-lateral", type=float, default=0.0)
    ap.add_argument(
        "--rail-pixels",
        type=float,
        nargs=4,
        default=None,
        metavar=("U1", "V1", "U2", "V2"),
        help="two pixels on one top edge of the rail: rail direction from the image instead of depth",
    )
    ap.add_argument(
        "--rail-height",
        type=float,
        default=0.02,
        help="rail top above the table (m), 2080 profile",
    )
    ap.add_argument(
        "--scene-only",
        action="store_true",
        help="only the scene camera (no wrist capture needed)",
    )
    ap.add_argument(
        "--rail-constraint",
        action="store_true",
        help="force the base y axis parallel to the measured rail (only if the base is square on it)",
    )
    ap.add_argument(
        "--gripper",
        type=float,
        default=0.0475,
        help="gripper finger joint value (m) at capture; 0.0475 = fully open (i2rt linear_4310)",
    )
    ap.add_argument(
        "--dark-v-max",
        type=int,
        default=80,
        help="HSV V threshold for 'black robot' pixels",
    )
    ap.add_argument(
        "--roi",
        type=int,
        nargs=4,
        default=[0, 115, 365, 340],
        metavar=("X0", "Y0", "X1", "Y1"),
        help="image region containing the arm and nothing else dark",
    )
    ap.add_argument(
        "--max-arm-depth", type=float, default=1.15, help="max depth (m) of arm pixels"
    )
    ap.add_argument(
        "--landmark",
        action="append",
        default=None,
        metavar="NAME:U,V",
        help="pixel of a robot site/body in the scene image (breaks the silhouette's mirror "
        "ambiguity). Default for this rig: grasp_site:335,215 link2:205,175 base:192,338",
    )
    args = ap.parse_args()

    cap = Path(args.capture)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.joints is None:
        meta_all = json.load(open(cap / "meta.json"))
        joints = np.array(
            meta_all.get(
                "robot_joints_measured", meta_all.get("robot_joints_assumed", [0] * 6)
            ),
            float,
        )
    else:
        joints = np.array(args.joints)
    print(f"joints at capture: {np.round(joints, 4)}")
    lm_args = args.landmark or ["grasp_site:335,215", "link2:205,175", "base:192,338"]
    landmarks = []
    for item in lm_args:
        name, uv = item.split(":")
        landmarks.append((name, tuple(float(v) for v in uv.split(","))))

    scene, T_base_cam, plane_base = calibrate_scene(
        cap,
        joints,
        out,
        dark_v_max=args.dark_v_max,
        roi_xyxy=tuple(args.roi),
        max_arm_depth=args.max_arm_depth,
        landmarks=landmarks,
        rail_constraint=args.rail_constraint,
        rail_pixels=args.rail_pixels,
        rail_height=args.rail_height,
    )
    # rig.json holds every camera in the frame the recorder stores (see to_recorded_frame).
    from raiden.camera_config import CameraConfig

    cfg = CameraConfig()
    if args.scene_only:
        to_recorded_frame(scene, "scene_camera", cfg)
        json.dump(
            {
                "joints_at_capture": joints.tolist(),
                "table": scene["table"],
                "rail": scene["rail"],
                "cameras": {"scene_camera": scene},
            },
            open(out / "rig.json", "w"),
            indent=1,
        )
        print(f"wrote {out / 'rig.json'} (scene only)")
        return
    wrist = calibrate_wrist(
        cap,
        joints,
        plane_base,
        out,
        args.wrist_forward,
        args.wrist_lateral,
        gripper=args.gripper,
        dark_v_max=args.dark_v_max,
    )
    # The wrist camera's table plane is a precise ruler for the tool (hence base) height above
    # the table; the scene silhouette constrains that height weakly. If they disagree, redo the
    # scene fit with the base height pinned by the wrist, then refit the wrist offsets.
    dz = wrist["fit"]["base_height_correction_m"]
    if abs(dz) > 0.004:
        z_fixed = scene["fit"]["base_height_above_table_m"] + dz
        print(
            f"[scene] base height {scene['fit']['base_height_above_table_m']:.3f} m disagrees with the wrist "
            f"camera by {dz:+.3f} m; refitting scene with base height fixed at {z_fixed:.3f} m"
        )
        scene, T_base_cam, plane_base = calibrate_scene(
            cap,
            joints,
            out,
            dark_v_max=args.dark_v_max,
            roi_xyxy=tuple(args.roi),
            max_arm_depth=args.max_arm_depth,
            landmarks=landmarks,
            rail_constraint=args.rail_constraint,
            z_fixed=z_fixed,
            rail_pixels=args.rail_pixels,
            rail_height=args.rail_height,
        )
        wrist = calibrate_wrist(
            cap,
            joints,
            plane_base,
            out,
            args.wrist_forward,
            args.wrist_lateral,
            gripper=args.gripper,
            dark_v_max=args.dark_v_max,
        )

    to_recorded_frame(scene, "scene_camera", cfg)
    to_recorded_frame(wrist, "left_wrist_camera", cfg)

    rig = {
        "frame": "robot base (i2rt YAM base frame, z up); transforms are cam2world",
        "capture": str(cap),
        "joints_at_capture": joints.tolist(),
        "robot": {"model": "i2rt YAM + linear_4310 gripper", "home_joints": [0] * 6},
        "table": scene["table"],
        "rail": scene["rail"],
        "cameras": {"scene_camera": scene, "left_wrist_camera": wrist},
    }
    with open(out / "rig.json", "w") as f:
        json.dump(rig, f, indent=1)
    print(f"wrote {out / 'rig.json'}; overlays in {out}")


if __name__ == "__main__":
    main()
