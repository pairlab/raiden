#!/usr/bin/env python3
"""Measure a planar neutral/color chart and test a per-camera output response.

Offline only. Manifest quads cover the OUTSIDE of the patch grid (TL,TR,BR,BL),
excluding a chart's outer frame. Raw images stay unchanged. Renderer PNGs are
optional; when provided, each has its own quad and must show the same target
patches under the declared scene illumination. Fits are diagnostic candidates
conditioned on that illumination/target, not physical sensor spectral calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares


def measure_grid(image, quad, rows, cols, fraction=0.45):
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Need an RGB image with three channels")
    quad = np.asarray(quad, dtype=np.float32)
    if (
        quad.shape != (4, 2)
        or not np.isfinite(quad).all()
        or not cv2.isContourConvex(quad)
    ):
        raise ValueError("quad must be four finite convex corners TL,TR,BR,BL")
    height, width = image.shape[:2]
    if (quad < 0).any() or (quad[:, 0] >= width).any() or (quad[:, 1] >= height).any():
        raise ValueError("Chart quad lies outside image")
    transform = cv2.getPerspectiveTransform(
        np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32), quad
    )
    result = []
    for row in range(rows):
        for col in range(cols):
            cx, cy = (col + 0.5) / cols, (row + 0.5) / rows
            dx, dy = fraction / (2 * cols), fraction / (2 * rows)
            square = np.array(
                [
                    [
                        [cx - dx, cy - dy],
                        [cx + dx, cy - dy],
                        [cx + dx, cy + dy],
                        [cx - dx, cy + dy],
                    ]
                ],
                np.float32,
            )
            polygon = cv2.perspectiveTransform(square, transform)[0]
            mask = np.zeros((height, width), np.uint8)
            cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 1)
            values = image[mask.astype(bool)]
            if len(values) < 25:
                raise ValueError(
                    f"Patch {row},{col} has fewer than25 interior pixels; capture larger chart"
                )
            result.append(
                {
                    "row": row,
                    "col": col,
                    "pixels": len(values),
                    "sample_polygon_xy": polygon.tolist(),
                    "median_rgb": np.median(values, axis=0).tolist(),
                    "p10_rgb": np.percentile(values, 10, axis=0).tolist(),
                    "p90_rgb": np.percentile(values, 90, axis=0).tolist(),
                    "clipped_fraction": float(
                        np.any((values <= 1) | (values >= 254), axis=1).mean()
                    ),
                }
            )
    return result


def analyze(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    target = manifest["target"]
    rows, cols = int(target["rows"]), int(target["cols"])
    if min(rows, cols) < 1:
        raise ValueError("rows and cols must be positive")
    root = manifest_path.parent
    observations = []
    neutral = {tuple(pair) for pair in target.get("neutral_patches_row_col", [])}
    for observation in manifest["observations"]:
        item = dict(observation)
        for domain in ("real", "render"):
            if domain not in item:
                continue
            spec = item[domain]
            path = root / spec["image"]
            pixels = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if pixels is None:
                raise ValueError(f"Cannot read {path}")
            item[domain] = {
                **spec,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "patches": measure_grid(
                    pixels[..., ::-1], spec["quad_xy_px"], rows, cols
                ),
            }
        # Bind each real frame to a recording or target-capture receipt.
        receipt = root / item["camera_receipt"]
        item["camera_receipt_sha256"] = hashlib.sha256(receipt.read_bytes()).hexdigest()
        record = json.loads(receipt.read_text())
        if record.get("camera", item["camera"]) != item["camera"]:
            raise ValueError("Camera role differs from the source receipt")
        if "frames" in record:
            if record.get("status") != "complete" or not record["frames"]:
                raise ValueError(
                    "Camera capture receipt must be complete with recorded frames"
                )
            if item["real"]["sha256"] not in {f["sha256"] for f in record["frames"]}:
                raise ValueError(
                    "Real image hash is absent from its camera capture receipt"
                )
        item["neutral_patch_diagnostics"] = []
        for patch in item["real"]["patches"]:
            if (patch["row"], patch["col"]) in neutral:
                red, green, blue = patch["median_rgb"]
                item["neutral_patch_diagnostics"].append(
                    {
                        "patch": [patch["row"], patch["col"]],
                        "rgb": [red, green, blue],
                        "r_over_g": red / green if green else None,
                        "b_over_g": blue / green if green else None,
                        "clipped_fraction": patch["clipped_fraction"],
                    }
                )
        observations.append(item)
    report = {
        "schema": "raiden.radiometric_target_diagnostic/1",
        "target": target,
        "observations": observations,
        "camera_candidates": [],
        "status": "diagnostic_only_no_default_changes",
        "encoding": "Both image domains are encoded output RGB; no inverse gamma assumed",
        "limitation": "Neutral patches constrain relative cast/brightness. A known color target broadens color excitation; absolute sensor/lighting separation still requires independent radiometry.",
    }
    for camera in sorted(set(o["camera"] for o in observations)):
        pairs = []
        for observation in observations:
            if observation["camera"] != camera or "render" not in observation:
                continue
            for real, render in zip(
                observation["real"]["patches"], observation["render"]["patches"]
            ):
                pairs.append(
                    {
                        "observation": observation["id"],
                        "split": observation["split"],
                        "patch": [real["row"], real["col"]],
                        "fit_eligible": max(
                            real["clipped_fraction"], render["clipped_fraction"]
                        )
                        < 0.01,
                        "real": real["median_rgb"],
                        "render": render["median_rgb"],
                    }
                )
        camera_observations = [o for o in observations if o["camera"] == camera]
        groups = {
            split: {
                o.get("pose_group", o["id"])
                for o in camera_observations
                if o["split"] == split
            }
            for split in ("fit", "holdout")
        }
        if groups["fit"] & groups["holdout"]:
            raise ValueError("A physical pose group cannot occur in fit and holdout")
        for domain in ("real", "render"):
            hashes = {
                split: {
                    o[domain]["sha256"]
                    for o in camera_observations
                    if o["split"] == split and domain in o
                }
                for split in ("fit", "holdout")
            }
            if hashes["fit"] & hashes["holdout"]:
                raise ValueError(
                    f"The same {domain} image cannot occur in fit and holdout"
                )
        train = [p for p in pairs if p["split"] == "fit" and p["fit_eligible"]]
        held = [p for p in pairs if p["split"] == "holdout"]
        if not train or not held:
            continue
        if (
            len({p["observation"] for p in train}) < 2
            or len({p["observation"] for p in held}) < 1
        ):
            raise ValueError(
                "Paired calibration needs at least two fit observations and one whole-observation holdout per camera"
            )
        x, y = (
            np.array([p["render"] for p in train]) / 255,
            np.array([p["real"] for p in train]) / 255,
        )
        singular = np.linalg.svd(x, compute_uv=False)
        if len(singular) < 3 or singular[-1] < 0.02 * singular[0]:
            report["camera_candidates"].append(
                {
                    "camera": camera,
                    "status": "insufficient_color_excitation",
                    "singular_values": singular.tolist(),
                    "reason": "Neutral-only/near-collinear target cannot identify a full color matrix",
                }
            )
            continue
        result = least_squares(
            lambda p: (x @ p.reshape(3, 3) - y).ravel(),
            np.eye(3).ravel(),
            bounds=(-2.0, 3.0),
            loss="soft_l1",
            f_scale=0.02,
        )
        matrix = result.x.reshape(3, 3)
        scores = []
        for pair in pairs:
            real = np.asarray(pair["real"])
            render = np.asarray(pair["render"])
            predicted = np.clip(render @ matrix, 0, 255)
            scores.append(
                {
                    **pair,
                    "candidate": predicted.tolist(),
                    "mae_before": float(np.abs(render - real).mean()),
                    "mae_after": float(np.abs(predicted - real).mean()),
                }
            )
        report["camera_candidates"].append(
            {
                "camera": camera,
                "status": "requires_scene_surface_validation",
                "matrix_row_vector_output_rgb": matrix.tolist(),
                "singular_values": singular.tolist(),
                "pairs": scores,
                "holdout_mae_before": float(
                    np.mean(
                        [s["mae_before"] for s in scores if s["split"] == "holdout"]
                    )
                ),
                "holdout_mae_after": float(
                    np.mean([s["mae_after"] for s in scores if s["split"] == "holdout"])
                ),
            }
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.manifest)
    # Never replace source images, previous reports, or deployed settings.
    with args.out.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(args.out)


if __name__ == "__main__":
    main()
