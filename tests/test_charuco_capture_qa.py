"""Planar board QA preserves native coordinates and measures only safe neutral interiors."""

import importlib.util
import unittest
from pathlib import Path

import cv2
import numpy as np

spec = importlib.util.spec_from_file_location(
    "charuco_capture_qa", Path(__file__).parents[1] / "scripts/charuco_capture_qa.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CharucoQATest(unittest.TestCase):
    def board(self):
        return module.board_definition(
            {
                "squares_x": 4,
                "squares_y": 5,
                "square_length": 0.05,
                "marker_length": 0.037,
                "dictionary": "DICT_6X6_250",
            }
        )

    def test_host_receptions_are_paired_by_time_with_unmatched_start_tail_removed(self):
        scene = [100_000_000, 133_000_000, 166_000_000, 199_000_000]
        wrist = [30_000_000, 63_000_000, 96_000_000, 129_000_000, 162_000_000]
        pairs = module.paired_frame_indices(scene, wrist)
        self.assertEqual(
            [(p["scene_frame_index"], p["wrist_frame_index"]) for p in pairs],
            [(0, 2), (1, 3), (2, 4)],
        )
        self.assertEqual(
            [p["wrist_minus_scene_host_reception_ns"] for p in pairs], [-4_000_000] * 3
        )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            module.paired_frame_indices([1, 1], [1, 2])

    def test_board_detection_recovers_native_geometry_and_printed_channel_cast(self):
        board = self.board()
        gray = board.generateImage((400, 500), marginSize=0, borderBits=1)
        source = np.array([[0, 0], [400, 0], [400, 500], [0, 500]], np.float32)
        target = np.array([[100, 60], [500, 60], [500, 560], [100, 560]], np.float32)
        H = cv2.getPerspectiveTransform(source, target)
        rgb = np.where(
            gray[..., None] == 255, np.array([180, 200, 220]), np.array([30, 40, 50])
        ).astype(np.uint8)
        native = cv2.warpPerspective(rgb, H, (640, 640), borderValue=(100, 100, 100))
        intr = {
            "fx": 800.0,
            "fy": 800.0,
            "cx": 320.0,
            "cy": 320.0,
            "distortion_coefficients": [0.0] * 5,
        }
        measurement, _, _ = module.measure_frame(native[..., ::-1].copy(), board, intr)
        self.assertTrue(measurement["all_inner_corners"])
        self.assertTrue(measurement["grid_fully_inside_native_frame"])
        self.assertLess(measurement["reprojection_RMS_px"], 0.2)
        neutral = measurement["neutral_by_erosion"]["2"]
        np.testing.assert_equal(neutral["white"]["median_rgb"], [180, 200, 220])
        np.testing.assert_equal(neutral["black"]["median_rgb"], [30, 40, 50])
        self.assertEqual(neutral["white"]["r_over_g"], 0.9)
        self.assertTrue(
            all(n >= 25 for n in neutral["white"]["component_pixel_counts"])
        )

    def test_tiny_components_are_not_promoted_to_neutral_patches(self):
        board = self.board()
        gray = cv2.resize(
            board.generateImage((400, 500), marginSize=0, borderBits=1),
            (40, 50),
            interpolation=cv2.INTER_NEAREST,
        )
        rgb = np.repeat(gray[..., None], 3, axis=2)
        measurement, _ = module.neutral_regions(
            rgb, board, np.diag([200.0, 200.0, 1.0]), erosion=2
        )
        self.assertEqual(measurement["white"]["pixels"], 0)


if __name__ == "__main__":
    unittest.main()
