"""Calibration transform conventions shared by capture, conversion and serving."""

import numpy as np


def validate_transform(matrix, name: str = "transform") -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 transform")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    ):
        raise ValueError(f"{name} must be a rigid transform with a proper rotation")
    return matrix


def left_from_right(calibration: dict) -> np.ndarray | None:
    """Map right-base points into left-base coordinates, including old files."""
    bimanual = calibration.get("bimanual_transform", {})
    canonical = bimanual.get("T_left_from_right")
    if canonical is not None:
        return validate_transform(canonical, "T_left_from_right")
    legacy = bimanual.get("right_base_to_left_base")
    if legacy is not None:
        # Historical files store LEFT -> RIGHT under this misleading key.
        return np.linalg.inv(validate_transform(legacy, "right_base_to_left_base"))
    return None
