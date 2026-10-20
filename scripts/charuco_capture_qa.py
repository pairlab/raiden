#!/usr/bin/env python3
"""Offline geometry and neutral-output QA for native ChArUco camera captures.

Only black/white print regions are measured. Their reflectance, illumination,
camera transfer response and spectral response are not independently known.
No full color matrix, gamma, linear radiance or deployed calibration is inferred.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def board_definition(config):
    return cv2.aruco.CharucoBoard(
        (config["squares_x"], config["squares_y"]),
        config["square_length"],
        config["marker_length"],
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config["dictionary"])),
    )


def neutral_regions(rgb, board, homography, erosion=2, minimum_pixels=25):
    """Project exact printed black/white regions, erode native pixels, reject tiny components."""
    nx, ny = board.getChessboardSize()
    template = board.generateImage((nx * 200, ny * 200), marginSize=0, borderBits=1)
    meters_per_pixel = board.getSquareLength() / 200
    transform = homography @ np.diag([meters_per_pixel, meters_per_pixel, 1.0])
    projected = cv2.warpPerspective(
        template,
        transform,
        (rgb.shape[1], rgb.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderValue=127,
    )
    result, masks = {}, {}
    for name, value in [("black", 0), ("white", 255)]:
        mask = (projected == value).astype(np.uint8)
        mask = cv2.erode(mask, np.ones((2 * erosion + 1, 2 * erosion + 1), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        accepted = [
            i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= minimum_pixels
        ]
        mask = np.isin(labels, accepted)
        masks[name] = mask
        pixels = rgb[mask]
        row = {
            "pixels": int(mask.sum()),
            "accepted_components": len(accepted),
            "component_pixel_counts": [
                int(stats[i, cv2.CC_STAT_AREA]) for i in accepted
            ],
            "erosion_native_pixels": erosion,
            "minimum_component_pixels": minimum_pixels,
        }
        if len(pixels):
            median = np.median(pixels, axis=0)
            row.update(
                median_rgb=median.tolist(),
                p10_rgb=np.percentile(pixels, 10, axis=0).tolist(),
                p90_rgb=np.percentile(pixels, 90, axis=0).tolist(),
                clipped_high_any_channel_fraction=float(
                    np.any(pixels >= 254, axis=1).mean()
                ),
                clipped_low_any_channel_fraction=float(
                    np.any(pixels <= 1, axis=1).mean()
                ),
                r_over_g=float(median[0] / median[1]) if median[1] else None,
                b_over_g=float(median[2] / median[1]) if median[1] else None,
            )
        result[name] = row
    return result, masks


def measure_frame(image, board, intrinsics):
    corners, ids, marker_corners, marker_ids = cv2.aruco.CharucoDetector(
        board
    ).detectBoard(image)
    result = {
        "marker_ids": [] if marker_ids is None else marker_ids.ravel().tolist(),
        "charuco_ids": [] if ids is None else ids.ravel().tolist(),
        "detected": ids is not None,
    }
    if ids is None or len(ids) < 6:
        result["geometry_status"] = "insufficient_corners"
        return result, None, None
    obj = board.getChessboardCorners()[ids.ravel()]
    K = np.array(
        [
            [intrinsics["fx"], 0, intrinsics["cx"]],
            [0, intrinsics["fy"], intrinsics["cy"]],
            [0, 0, 1.0],
        ]
    )
    distortion = np.asarray(intrinsics["distortion_coefficients"])
    ok, rvec, tvec = cv2.solvePnP(obj, corners, K, distortion)
    if not ok:
        result["geometry_status"] = "pnp_failed"
        return result, None, None
    predicted = cv2.projectPoints(obj, rvec, tvec, K, distortion)[0].reshape(-1, 2)
    error = np.linalg.norm(predicted - corners.reshape(-1, 2), axis=1)
    nx, ny = board.getChessboardSize()
    square = board.getSquareLength()
    exterior = np.array(
        [
            [0, 0, 0],
            [nx * square, 0, 0],
            [nx * square, ny * square, 0],
            [0, ny * square, 0],
        ],
        np.float32,
    )
    quad = cv2.projectPoints(exterior, rvec, tvec, K, distortion)[0].reshape(-1, 2)
    homography, _ = cv2.findHomography(obj[:, :2], corners.reshape(-1, 2))
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    result.update(
        geometry_status="measured",
        native_K=K.tolist(),
        distortion=distortion.tolist(),
        charuco_corners_xy=corners.reshape(-1, 2).tolist(),
        T_camera_board=T.tolist(),
        grid_outer_quad_xy=quad.tolist(),
        reprojection_RMS_px=float(np.sqrt(np.mean(error**2))),
        reprojection_max_px=float(error.max()),
        all_inner_corners=len(ids) == len(board.getChessboardCorners()),
        grid_fully_inside_native_frame=bool(
            np.all(quad[:, 0] > 3)
            and np.all(quad[:, 0] < image.shape[1] - 3)
            and np.all(quad[:, 1] > 3)
            and np.all(quad[:, 1] < image.shape[0] - 3)
        ),
    )
    result["neutral_by_erosion"] = {}
    masks = None
    # The planar pixel homography is appropriate here only for the retained zero-distortion native inputs.
    if not np.allclose(distortion, 0, atol=1e-12, rtol=0):
        result["neutral_status"] = "requires_native_distortion_aware_projection"
    else:
        for erosion in (1, 2, 3):
            measurements, selected = neutral_regions(
                image[..., ::-1], board, homography, erosion
            )
            result["neutral_by_erosion"][str(erosion)] = measurements
            if erosion == 2:
                masks = selected
        result["neutral_status"] = "encoded_output_diagnostic_only"
    overlay = image.copy()
    cv2.aruco.drawDetectedCornersCharuco(overlay, corners, ids)
    cv2.polylines(overlay, [np.rint(quad).astype(np.int32)], True, (0, 255, 255), 1)
    return result, overlay, masks


def paired_frame_indices(scene_ns, wrist_ns, tolerance_ns=16_666_667):
    """Pair nearest host receptions within half a nominal 30 Hz period; never pair by index."""
    if np.any(np.diff(scene_ns) <= 0) or np.any(np.diff(wrist_ns) <= 0):
        raise ValueError(
            "capture host reception timestamps must be strictly increasing"
        )
    pairs = []
    for i, timestamp in enumerate(scene_ns):
        j = int(np.abs(np.asarray(wrist_ns, dtype=np.int64) - int(timestamp)).argmin())
        delta = int(wrist_ns[j]) - int(timestamp)
        if abs(delta) <= tolerance_ns:
            pairs.append(
                {
                    "scene_frame_index": i,
                    "wrist_frame_index": j,
                    "wrist_minus_scene_host_reception_ns": delta,
                }
            )
    return pairs


def analyze(capture_path, calibration_path, output, paper_border_path=None):
    capture_path, calibration_path, output = (
        Path(capture_path),
        Path(calibration_path),
        Path(output),
    )
    capture = json.loads(capture_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    if capture["status"] != "complete":
        raise ValueError("capture must be complete")
    board = board_definition(calibration["charuco_config"])
    paper_border = (
        json.loads(Path(paper_border_path).read_text()) if paper_border_path else None
    )
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "mesa.charuco_native_capture_qa/1",
        "capture": str(capture_path.resolve()),
        "capture_sha256": sha256(capture_path),
        "calibration": str(calibration_path.resolve()),
        "calibration_sha256": sha256(calibration_path),
        "board": calibration["charuco_config"],
        "script_sha256": sha256(__file__),
        "physical_print_scale_independently_measured": False,
        "scope": "geometry and relative encoded black/white brightness/cast only; no full color matrix or sensor linearity assumed",
        "clipping_interpretation": "clipping statistics are not a calibrated glare classifier",
        "cameras": {},
        "robot_telemetry": "not captured by camera-only tool; bind separately measured native controller telemetry",
    }
    if paper_border_path:
        report["fixed_scene_paper_border_source"] = str(
            Path(paper_border_path).resolve()
        )
        report["fixed_scene_paper_border_sha256"] = sha256(paper_border_path)
        report["fixed_scene_paper_border_limit"] = (
            "Geometry-defined native ROIs on the stationary board; no color threshold selection. Paper reflectance and optical brighteners unmeasured; narrow-border pixel sensitivity retained separately."
        )
    timestamps = {}
    for camera in ("scene_camera", "left_wrist_camera"):
        receipt_path = capture_path.parent / camera / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        if receipt["status"] != "complete" or receipt["settings_written"]:
            raise ValueError("camera receipt must be complete with no setting writes")
        before = {
            k: v.get("value") for k, v in receipt["before"]["color_options"].items()
        }
        after = {
            k: v.get("value") for k, v in receipt["after"]["color_options"].items()
        }
        if before != after:
            raise ValueError("camera controls changed during capture")
        measurements = []
        for index, frame in enumerate(receipt["frames"]):
            path = capture_path.parent / frame["image"]
            if sha256(path) != frame["sha256"]:
                raise ValueError("native source frame hash changed")
            image = cv2.imread(str(path))
            intrinsics = receipt["native_intrinsics"]
            if image is None or image.shape[:2] != (
                intrinsics["height"],
                intrinsics["width"],
            ):
                raise ValueError("native image size differs from source intrinsics")
            measurement, overlay, masks = measure_frame(image, board, intrinsics)
            if camera == "scene_camera" and paper_border is not None:
                regions = []
                for region in paper_border["patches"]:
                    mask = np.zeros(image.shape[:2], np.uint8)
                    cv2.fillConvexPoly(
                        mask, np.rint(region["polygon_native_xy"]).astype(np.int32), 1
                    )
                    values = image[..., ::-1][mask > 0]
                    median = np.median(values, axis=0)
                    regions.append(
                        {
                            "id": region["id"],
                            "pixels": len(values),
                            "median_rgb": median.tolist(),
                            "p10_rgb": np.percentile(values, 10, axis=0).tolist(),
                            "p90_rgb": np.percentile(values, 90, axis=0).tolist(),
                            "clipped_high_any_channel_fraction": float(
                                np.any(values >= 254, axis=1).mean()
                            ),
                            "r_over_g": float(median[0] / median[1])
                            if median[1]
                            else None,
                            "b_over_g": float(median[2] / median[1])
                            if median[1]
                            else None,
                        }
                    )
                measurement["fixed_paper_border_regions"] = regions
            measurements.append(
                {
                    **frame,
                    "frame_index": index,
                    "image": str(path.resolve()),
                    **measurement,
                }
            )
            if index == len(receipt["frames"]) // 2 and overlay is not None:
                cv2.imwrite(str(output / f"{camera}_corners.png"), overlay)
                if masks is not None:
                    for name, mask in masks.items():
                        cv2.imwrite(
                            str(output / f"{camera}_{name}_mask.png"),
                            mask.astype(np.uint8) * 255,
                        )
        timestamps[camera] = [
            f["host_wait_after_monotonic_ns"] for f in receipt["frames"]
        ]
        good = [m for m in measurements if m.get("geometry_status") == "measured"]
        report["cameras"][camera] = {
            "receipt_sha256": sha256(receipt_path),
            "native_intrinsics": receipt["native_intrinsics"],
            "policy_crop": receipt.get("policy_crop"),
            "accepted_policy_intrinsics": calibration["cameras"][camera]["intrinsics"],
            "controls_unchanged": True,
            "frames": measurements,
            "summary": {
                "frames": len(measurements),
                "detected_frames": len(good),
                "all_inner_corners_frames": sum(
                    m.get("all_inner_corners", False) for m in good
                ),
                "full_grid_inside_frames": sum(
                    m.get("grid_fully_inside_native_frame", False) for m in good
                ),
                "reprojection_RMS_px_range": [
                    min(m["reprojection_RMS_px"] for m in good),
                    max(m["reprojection_RMS_px"] for m in good),
                ]
                if good
                else None,
            },
        }
    report["host_reception_pairs"] = paired_frame_indices(
        timestamps["scene_camera"], timestamps["left_wrist_camera"]
    )
    report["pairing_limit"] = (
        "Independent rolling-shutter cameras; nearest host reception within 16.667ms is not hardware synchronization or an exposure-time guarantee"
    )
    (output / "qa.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--paper-border",
        type=Path,
        help="Previously defined native scene-paper border polygons; requires stationary board/scene camera",
    )
    args = parser.parse_args()
    result = analyze(args.capture, args.calibration, args.out, args.paper_border)
    print(
        json.dumps(
            {camera: data["summary"] for camera, data in result["cameras"].items()},
            indent=2,
        )
    )
    print(args.out / "qa.json")


if __name__ == "__main__":
    main()
