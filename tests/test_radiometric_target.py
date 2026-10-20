"""Offline chart verification: known mapping, held-out poses, and neutral rank."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

spec = importlib.util.spec_from_file_location(
    "radiometric_target", Path(__file__).parents[1] / "scripts/radiometric_target.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TargetTest(unittest.TestCase):
    def image(self, neutral=False):
        values = np.random.default_rng(21).integers(40, 180, (4, 6, 3), dtype=np.uint8)
        if neutral:
            values[:] = np.arange(24).reshape(4, 6, 1) * 5 + 40
        return np.repeat(np.repeat(values, 40, axis=0), 40, axis=1)

    def manifest(self, root, neutral=False):
        quad = [[0, 0], [239, 0], [239, 159], [0, 159]]
        observations, frames = [], []
        for i in range(3):
            render = self.image(neutral) + i * 3
            real = np.rint(render.astype(float) * np.array([0.8, 0.9, 1.1])).astype(
                np.uint8
            )
            cv2.imwrite(str(root / f"render{i}.png"), render[..., ::-1])
            cv2.imwrite(str(root / f"real{i}.png"), real[..., ::-1])
            frames.append(
                {
                    "sha256": hashlib.sha256(
                        (root / f"real{i}.png").read_bytes()
                    ).hexdigest()
                }
            )
            observations.append(
                {
                    "id": f"pose{i}",
                    "pose_group": f"pose{i}",
                    "camera": "scene_camera",
                    "camera_receipt": "receipt.json",
                    "split": "fit" if i < 2 else "holdout",
                    "real": {"image": f"real{i}.png", "quad_xy_px": quad},
                    "render": {"image": f"render{i}.png", "quad_xy_px": quad},
                }
            )
        (root / "receipt.json").write_text(
            json.dumps(
                {"camera": "scene_camera", "status": "complete", "frames": frames}
            )
        )
        path = root / "manifest.json"
        path.write_text(
            json.dumps({"target": {"rows": 4, "cols": 6}, "observations": observations})
        )
        return path

    def test_patch_sampling_ignores_gaps_and_recovers_grid(self):
        image = self.image()
        patches = module.measure_grid(
            image, [[0, 0], [239, 0], [239, 159], [0, 159]], 4, 6
        )
        np.testing.assert_equal(patches[7]["median_rgb"], image[60, 60])
        with self.assertRaisesRegex(ValueError, "outside"):
            module.measure_grid(image, [[-2, 0], [239, 0], [239, 159], [0, 159]], 4, 6)

    def test_known_global_color_mapping_recovers_whole_pose_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            report = module.analyze(self.manifest(Path(directory)))
        candidate = report["camera_candidates"][0]
        self.assertLess(candidate["holdout_mae_after"], 0.4)
        self.assertLess(
            candidate["holdout_mae_after"], candidate["holdout_mae_before"] / 10
        )
        self.assertEqual(candidate["status"], "requires_scene_surface_validation")

    def test_neutral_target_cannot_fit_full_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            report = module.analyze(self.manifest(Path(directory), neutral=True))
        self.assertEqual(
            report["camera_candidates"][0]["status"], "insufficient_color_excitation"
        )

    def test_pose_group_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.manifest(Path(directory))
            manifest = json.loads(path.read_text())
            manifest["observations"][2]["pose_group"] = "pose0"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "physical pose group"):
                module.analyze(path)

    def test_failed_or_unbound_capture_cannot_supply_chart_evidence(self):
        for mutation in ("failed", "empty", "hash"):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                manifest = self.manifest(root)
                receipt = json.loads((root / "receipt.json").read_text())
                if mutation == "failed":
                    receipt["status"] = "failed"
                elif mutation == "empty":
                    receipt["frames"] = []
                else:
                    receipt["frames"][0]["sha256"] = "wrong"
                (root / "receipt.json").write_text(json.dumps(receipt))
                with self.assertRaisesRegex(ValueError, "capture receipt"):
                    module.analyze(manifest)

    def test_duplicate_images_cannot_bypass_whole_pose_holdout(self):
        for domain in ("real", "render"):
            with (
                self.subTest(domain=domain),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                path = self.manifest(root)
                manifest = json.loads(path.read_text())
                manifest["observations"][2][domain] = manifest["observations"][0][
                    domain
                ]
                path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, f"same {domain} image"):
                    module.analyze(path)


if __name__ == "__main__":
    unittest.main()
