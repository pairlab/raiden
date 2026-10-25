"""Orchestrates the full camera calibration workflow"""

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Import kinematics from i2rt
from i2rt.robots.kinematics import Kinematics

from raiden._config import CALIBRATION_FILE, CALIBRATION_POSES_FILE, CAMERA_CONFIG
from raiden._xml_paths import get_yam_4310_linear_xml_path as _get_combined_xml_path
from raiden.calibration.core import (
    CameraCalibrator,
    ChArUcoBoardConfig,
    load_calibration_poses,
    save_calibration_results,
)
from raiden.camera_config import CameraConfig
from raiden.calibration_utils import validate_transform
from raiden.cameras.base import Camera
from raiden.robot.controller import RobotController, smooth_move_joints

# Global kinematics instance (cached for efficiency)
_kinematics_cache: Optional[Kinematics] = None


def compute_bimanual_base_transform_from_calibration(
    left_arm_poses: List[np.ndarray],
    right_arm_poses: List[np.ndarray],
    left_board_poses: List[Tuple[np.ndarray, np.ndarray]],
    right_board_poses: List[Tuple[np.ndarray, np.ndarray]],
    left_hand_eye_result: Dict,
    right_hand_eye_result: Dict,
) -> np.ndarray:
    """Compute the legacy left-base to right-base transform after hand–eye calibration.

    The board must remain fixed, but each arm may observe it at different times.
    Each robot pose must correspond to its own board observation. Average the
    board pose in each base independently, then relate the two base frames.

    Args:
        left_arm_poses: List of T_left_base_to_left_ee transforms
        right_arm_poses: List of T_right_base_to_right_ee transforms (in right base frame)
        left_board_poses: List of (rvec, tvec) board observations from left camera
        right_board_poses: List of (rvec, tvec) board observations from right camera
        left_hand_eye_result: Hand-eye calibration result for left wrist (T_left_cam_to_left_ee)
        right_hand_eye_result: Hand-eye calibration result for right wrist (T_right_cam_to_right_ee)

    Returns:
        Historical serialized transform, mapping LEFT-base points into RIGHT-base
        coordinates, despite the legacy ``right_base_to_left_base`` name.
    """
    if len(left_arm_poses) < 1 or len(right_arm_poses) < 1:
        raise ValueError("Need at least 1 pose from both arms")

    if len(left_board_poses) < 1 or len(right_board_poses) < 1:
        raise ValueError("Need at least 1 board observation from both cameras")

    from scipy.spatial.transform import Rotation

    if len(left_arm_poses) != len(left_board_poses) or len(right_arm_poses) != len(right_board_poses):
        raise ValueError("Each arm's robot poses and board observations must correspond")
    print(
        "    Computing bimanual base transform from "
        f"{len(left_arm_poses)} left / {len(right_arm_poses)} right observations..."
    )

    # Extract hand-eye transforms (camera to end-effector)
    T_left_cam_to_left_ee = np.eye(4)
    T_left_cam_to_left_ee[:3, :3] = np.array(left_hand_eye_result["rotation_matrix"])
    T_left_cam_to_left_ee[:3, 3] = np.array(left_hand_eye_result["translation_vector"])

    T_right_cam_to_right_ee = np.eye(4)
    T_right_cam_to_right_ee[:3, :3] = np.array(right_hand_eye_result["rotation_matrix"])
    T_right_cam_to_right_ee[:3, 3] = np.array(
        right_hand_eye_result["translation_vector"]
    )

    # Verify rotation matrices are valid
    det_left = np.linalg.det(T_left_cam_to_left_ee[:3, :3])
    det_right = np.linalg.det(T_right_cam_to_right_ee[:3, :3])

    if abs(det_left - 1.0) > 0.01 or abs(det_right - 1.0) > 0.01:
        print(
            f"    Warning: Rotation matrices are not orthogonal (det_left={det_left:.4f}, det_right={det_right:.4f})"
        )

    def mean_board_pose(robot_poses, board_poses, hand_eye):
        transforms = []
        for robot_pose, (rvec, tvec) in zip(robot_poses, board_poses):
            board_to_camera = np.eye(4)
            board_to_camera[:3, :3] = cv2.Rodrigues(rvec)[0]
            board_to_camera[:3, 3] = np.asarray(tvec).reshape(3)
            transforms.append(robot_pose @ hand_eye @ board_to_camera)
        mean = np.eye(4)
        mean[:3, :3] = Rotation.from_matrix([T[:3, :3] for T in transforms]).mean().as_matrix()
        mean[:3, 3] = np.mean([T[:3, 3] for T in transforms], axis=0)
        return mean

    T_left_board = mean_board_pose(left_arm_poses, left_board_poses, T_left_cam_to_left_ee)
    T_right_board = mean_board_pose(right_arm_poses, right_board_poses, T_right_cam_to_right_ee)
    # Preserve the historical numeric contract for existing readers.
    T_right_base_to_left_base = T_right_board @ np.linalg.inv(T_left_board)

    print(
        f"    Bimanual transform (T_right_base_to_left_base) translation: {T_right_base_to_left_base[:3, 3]}"
    )

    return T_right_base_to_left_base


# Bimanual base transform (computed during calibration)
T_RIGHT_BASE_TO_LEFT_BASE: Optional[np.ndarray] = None


def compute_forward_kinematics(
    joint_positions: np.ndarray, arm: str = "left"
) -> np.ndarray:
    """Compute forward kinematics for YAM robot arm using i2rt library

    Args:
        joint_positions: 6 joint positions (without gripper)
        arm: Which arm ("left" or "right")

    Returns:
        4x4 transformation matrix from left arm base to end-effector (grasp_site)
        All transforms are expressed relative to the left arm base frame.
    """
    global _kinematics_cache

    # Initialize kinematics on first call (cached for subsequent calls)
    if _kinematics_cache is None:
        _kinematics_cache = Kinematics(_get_combined_xml_path(), site_name="grasp_site")

    # Ensure we have 6 DoF (without gripper)
    assert len(joint_positions) == 6, (
        f"Expected 6 joint positions, got {len(joint_positions)}"
    )

    # Pad to 8 DOF (combined arm+gripper XML); gripper joints don't affect grasp_site
    q = np.zeros(8)
    q[:6] = joint_positions

    # Compute forward kinematics (returns T_arm_base_to_ee)
    T_arm_base_to_ee = _kinematics_cache.fk(q)

    # Transform to left arm base frame
    if arm == "right":
        if T_RIGHT_BASE_TO_LEFT_BASE is None:
            raise ValueError(
                "Bimanual transform not available. "
                "Calibrate both left and right wrist cameras together to compute it."
            )

        # For right arm: T_left_base_to_ee = T_left_base_to_right_base @ T_right_base_to_ee
        T_left_base_to_right_base = np.linalg.inv(T_RIGHT_BASE_TO_LEFT_BASE)
        return T_left_base_to_right_base @ T_arm_base_to_ee

    # For left arm, already in the correct frame
    return T_arm_base_to_ee


class CalibrationRunner:
    """Runs the complete calibration workflow"""

    def __init__(
        self,
        camera_config_file: str = CAMERA_CONFIG,
        poses_file: str = CALIBRATION_POSES_FILE,
        output_file: str = CALIBRATION_FILE,
        charuco_config: Optional[ChArUcoBoardConfig] = None,
    ):
        self.camera_config_file = camera_config_file
        self.poses_file = poses_file
        self.output_file = output_file
        self._charuco_config_override = charuco_config

        self.camera_config: Optional[CameraConfig] = None
        self.poses_data: Optional[Dict] = None
        self.board_config: Optional[ChArUcoBoardConfig] = None
        self.calibrator: Optional[CameraCalibrator] = None

        self.robot_controller: Optional[RobotController] = None
        self.cameras: Dict[str, Camera] = {}
        self._wrist_arms: Dict[str, str] = {}
        self._move_stop_event = threading.Event()
        self._captures: List[Dict] = []

    def load_configuration(self):
        """Load camera config and calibration poses"""
        print("\nLoading configuration...")

        # Load camera config
        self.camera_config = CameraConfig(self.camera_config_file)
        num_cameras = len(self.camera_config.list_cameras())
        print(f"✓ Camera config loaded: {num_cameras} camera(s) configured")

        # Load poses
        self.poses_data = load_calibration_poses(self.poses_file)
        num_poses = len(self.poses_data["poses"])
        print(f"✓ Calibration poses loaded: {num_poses} pose(s)")

        # Load ChArUco config (CLI override takes precedence over poses file)
        if self._charuco_config_override is not None:
            self.board_config = self._charuco_config_override
        else:
            self.board_config = ChArUcoBoardConfig.from_dict(
                self.poses_data["charuco_config"]
            )
        print("✓ ChArUco board config loaded")

        # Create calibrator
        self.calibrator = CameraCalibrator(self.board_config)

    def initialize_robots(self, left_wrist: bool, right_wrist: bool):
        """Initialize robots based on which cameras to calibrate"""
        # Create robot controller (only followers, no leaders needed for calibration)
        self.robot_controller = RobotController(
            use_right_leader=False,
            use_left_leader=False,
            use_right_follower=right_wrist,
            use_left_follower=left_wrist,
        )

        # Check CAN interfaces
        print("\nChecking CAN interfaces...")
        self.robot_controller.check_can_interfaces()

        # Initialize robots
        print("\nInitializing robots...")
        self.robot_controller.initialize_robots(gravity_comp_mode=False)

        # Move to home positions
        self.robot_controller.move_to_home_positions(simultaneous=True)

    def initialize_cameras(self, camera_names: List[str], warmup_frames: int = 10):
        """Initialize cameras by name using the camera factory."""
        print("\nInitializing cameras...")

        for name in camera_names:
            cam_type = self.camera_config.get_camera_type(name)
            serial = self.camera_config.get_serial_by_name(name)
            if serial is None:
                raise ValueError(f"Camera '{name}' not found in config")

            print(f"  - Initializing {name} ({cam_type}, serial: {serial})...")
            camera = self.camera_config.create_camera(name)
            self.cameras[name] = camera
            camera.open()

        # Warm up all cameras together after all are open, so each one has had
        # time to stabilize regardless of how many cameras are connected.
        print(f"  Warming up cameras ({warmup_frames} frames)...")
        for _ in range(warmup_frames):
            for name, camera in self.cameras.items():
                if not camera.grab():
                    raise RuntimeError(f"Camera warmup failed for {name}")

        print("✓ Cameras initialized")

    def _compute_and_apply_bimanual_transform(
        self, camera_data: Dict, left_camera_id: str, right_camera_id: str
    ) -> bool:
        """Compute bimanual transform from hand-eye calibration results

        Strategy:
        1. Compute left hand-eye calibration (in left arm base frame)
        2. Compute right hand-eye calibration (in right arm base frame)
        3. Use both hand-eye results to compute bimanual base transform
        4. Update right arm poses to left base frame

        Args:
            camera_data: Dictionary with calibration data for all cameras
            left_camera_id: Name of left wrist camera
            right_camera_id: Name of right wrist camera

        Returns:
            True if transform was successfully computed, False otherwise
        """
        global T_RIGHT_BASE_TO_LEFT_BASE

        print("\nComputing bimanual base transform from hand-eye calibrations...")

        left_data = camera_data.get(left_camera_id)
        right_data = camera_data.get(right_camera_id)

        # Validate we have data from both cameras
        if not left_data or not right_data:
            print("  ✗ Missing camera data")
            return False

        if not left_data["corners"] or not right_data["corners"]:
            print("  ✗ No board detections found")
            return False

        # Step 1: Get camera intrinsics from SDK (factory calibrated)
        print("  Step 1: Getting camera intrinsics from SDK...")
        left_camera = self.cameras.get(left_camera_id)
        right_camera = self.cameras.get(right_camera_id)

        if not left_camera or not right_camera:
            print("  ✗ Camera objects not found")
            return False

        left_cam_matrix, left_dist_coeffs, left_img_size = left_camera.get_intrinsics()
        right_cam_matrix, right_dist_coeffs, right_img_size = (
            right_camera.get_intrinsics()
        )

        print(
            f"    ✓ Left camera: fx={left_cam_matrix[0, 0]:.1f}, fy={left_cam_matrix[1, 1]:.1f}"
        )
        print(
            f"    ✓ Right camera: fx={right_cam_matrix[0, 0]:.1f}, fy={right_cam_matrix[1, 1]:.1f}"
        )

        # Step 2: Estimate board poses from both cameras
        left_robot_poses, left_board_poses = self._valid_wrist_observations(
            left_data,
            left_cam_matrix.tolist(),
            left_dist_coeffs.tolist(),
        )
        right_robot_poses, right_board_poses = self._valid_wrist_observations(
            right_data,
            right_cam_matrix.tolist(),
            right_dist_coeffs.tolist(),
        )

        if len(left_robot_poses) < 3 or len(right_robot_poses) < 3:
            print("  ✗ Need at least 3 valid observations per arm")
            return False

        print(
            "  Step 2: Computing hand-eye calibrations..."
        )

        # Step 3: Compute left hand-eye calibration (in left arm base frame)
        print("    Computing left wrist hand-eye calibration...")
        left_hand_eye_result = self.calibrator.calibrate_hand_eye(
            left_robot_poses, left_board_poses
        )

        if not left_hand_eye_result["success"]:
            print(
                f"      ✗ Left hand-eye calibration failed: {left_hand_eye_result.get('error', 'Unknown error')}"
            )
            return False

        print(
            f"      ✓ Left hand-eye calibration complete (method: {left_hand_eye_result['method']})"
        )

        # Step 4: Compute right hand-eye calibration (in right arm base frame)
        print("    Computing right wrist hand-eye calibration...")
        right_hand_eye_result = self.calibrator.calibrate_hand_eye(
            right_robot_poses, right_board_poses
        )

        if not right_hand_eye_result["success"]:
            print(
                f"      ✗ Right hand-eye calibration failed: {right_hand_eye_result.get('error', 'Unknown error')}"
            )
            return False

        print(
            f"      ✓ Right hand-eye calibration complete (method: {right_hand_eye_result['method']})"
        )

        # Step 5: Compute bimanual base transform using both hand-eye results
        print("  Step 3: Computing bimanual base transform from hand-eye results...")

        T_RIGHT_BASE_TO_LEFT_BASE = compute_bimanual_base_transform_from_calibration(
            left_robot_poses,
            right_robot_poses,
            left_board_poses,
            right_board_poses,
            left_hand_eye_result,
            right_hand_eye_result,
        )

        print(
            "    ✓ Bimanual transform computed from both arms"
        )

        # Step 6: Transform right arm poses to left base frame
        T_left_base_to_right_base = np.linalg.inv(T_RIGHT_BASE_TO_LEFT_BASE)
        right_data["robot_poses"] = [
            T_left_base_to_right_base @ pose if pose is not None else None
            for pose in right_data["robot_poses"]
        ]
        print("  ✓ Right arm poses transformed to left base frame")

        return True

    def _estimate_board_poses(
        self, camera_data: Dict, camera_matrix: list, distortion_coeffs: list
    ) -> List:
        """Estimate board poses from corner detections"""
        camera_matrix_np = np.array(camera_matrix)
        dist_coeffs_np = np.array(distortion_coeffs)
        board_poses = []

        for corners, ids in zip(camera_data["corners"], camera_data["ids"]):
            rvec, tvec = self.calibrator.detector.estimate_pose(
                corners, ids, camera_matrix_np, dist_coeffs_np
            )
            board_poses.append((rvec, tvec) if rvec is not None else None)

        return board_poses

    def _valid_wrist_observations(self, data: Dict, camera_matrix, distortion_coeffs):
        """Filter FK and PnP together so a failed observation never shifts a pair."""
        boards = self._estimate_board_poses(data, camera_matrix, distortion_coeffs)
        if len(boards) != len(data["robot_poses"]):
            raise ValueError("Robot and board observation counts do not match")
        pairs = [
            (robot, board)
            for robot, board in zip(data["robot_poses"], boards)
            if robot is not None and board is not None
        ]
        return [robot for robot, _ in pairs], [board for _, board in pairs]

    @staticmethod
    def _validate_results(results: Dict, target_cameras: List[str], bimanual_required: bool):
        """Reject incomplete or invalid solves before replacing the result file."""
        if not target_cameras:
            raise ValueError("No calibration cameras selected")
        if bimanual_required and not results.get("bimanual_transform"):
            raise ValueError("Bimanual base calibration failed")
        for name in target_cameras:
            camera = results["cameras"].get(name, {})
            key = "hand_eye_calibration" if camera.get("type") == "hand_eye" else "extrinsics"
            extrinsics = camera.get(key, {})
            if not extrinsics.get("success"):
                raise ValueError(f"Missing successful {key} for {name}")
            transform = np.eye(4)
            try:
                transform[:3, :3] = np.asarray(extrinsics["rotation_matrix"], dtype=float).reshape(3, 3)
                transform[:3, 3] = np.asarray(extrinsics["translation_vector"], dtype=float).reshape(3)
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(f"Invalid extrinsics for {name}") from exc
            validate_transform(transform, name)
            if key == "extrinsics" and extrinsics.get("reference_frame") != results["coordinate_frame"]:
                raise ValueError(f"{name} extrinsics are not in {results['coordinate_frame']}")
        if bimanual_required:
            validate_transform(results["bimanual_transform"]["T_left_from_right"], "T_left_from_right")

    def _save_captures(self) -> Path:
        """Write each pose's images, measured joints and the camera intrinsics to disk.

        The calibration keeps nothing else from the capture, so these files are what let a bad
        solve be diagnosed and redone offline. They go next to the output file, under
        ``calibration_captures/<timestamp>/``.
        """
        out_dir = (
            Path(self.output_file).parent
            / "calibration_captures"
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        intrinsics = {}
        for name, camera in self.cameras.items():
            try:
                cam_matrix, dist_coeffs, image_size = camera.get_intrinsics()
                intrinsics[name] = {
                    "camera_matrix": np.asarray(cam_matrix).tolist(),
                    "distortion_coeffs": np.asarray(dist_coeffs).ravel().tolist(),
                    "image_size": list(image_size),
                }
            except Exception as e:
                print(f"  Warning: no intrinsics for {name}: {e}")

        poses = []
        for i, capture in enumerate(self._captures):
            files = {}
            for name, image in capture["images"].items():
                files[name] = f"{i:02d}_{name}.png"
                if not cv2.imwrite(str(out_dir / files[name]), image):
                    raise OSError(f"Could not save calibration image: {out_dir / files[name]}")
            poses.append(
                {
                    "id": capture["id"],
                    "pose": capture["pose"],
                    "joints_l": capture["joints_l"],
                    "joints_r": capture["joints_r"],
                    "images": files,
                }
            )

        meta = {
            "charuco_config": self.board_config.__dict__,
            "intrinsics": intrinsics,
            "poses": poses,
        }
        (out_dir / "captures.json").write_text(json.dumps(meta, indent=2))
        print(f"\n✓ Capture images and joints saved to: {out_dir}")
        return out_dir

    def collect_calibration_data(
        self,
    ) -> Dict[str, Dict[str, List]]:
        """Move through poses and collect calibration data.

        Returns:
            Dictionary mapping camera names to their calibration data.
            Each value is a dict with keys ``images``, ``corners``, ``ids``,
            and ``robot_poses`` (for hand-eye calibration).
        """
        print("\nStarting calibration sequence...")
        print("=" * 70)
        print(
            "Coordinate frame: All transforms will be expressed relative to left arm base"
        )
        print()

        # Initialize data storage
        camera_data = {
            name: {
                "images": [],
                "corners": [],
                "ids": [],
                "robot_poses": [],
                "observation_ids": [],
                "arm_type": self._wrist_arms.get(name) or {
                    "left_wrist": "left", "right_wrist": "right"
                }.get(self.camera_config.get_role(name)),
            }
            for name in self.cameras.keys()
        }

        poses = self.poses_data["poses"]
        self._captures = []

        for i, pose in enumerate(poses):
            print(f"\nPose {i + 1}/{len(poses)}: {pose['name']}")

            # Move robots to pose (only if we have robots)
            if self.robot_controller:
                threads = []
                move_errors = []

                def move(robot, target):
                    try:
                        smooth_move_joints(robot, target, stop_event=self._move_stop_event)
                    except BaseException as exc:
                        move_errors.append(exc)
                        self._move_stop_event.set()

                if self.robot_controller.follower_r and "follower_r" in pose:
                    target_r = np.array(pose["follower_r"])
                    threads.append(
                        threading.Thread(
                            target=move,
                            args=(self.robot_controller.follower_r, target_r),
                            daemon=True,
                        )
                    )
                if self.robot_controller.follower_l and "follower_l" in pose:
                    target_l = np.array(pose["follower_l"])
                    threads.append(
                        threading.Thread(
                            target=move,
                            args=(self.robot_controller.follower_l, target_l),
                            daemon=True,
                        )
                    )

                if threads:
                    print(
                        f"  Moving {'both arms' if len(threads) > 1 else 'arm'} simultaneously..."
                    )
                    try:
                        for t in threads:
                            t.start()
                        for t in threads:
                            t.join()
                    except BaseException:
                        self._move_stop_event.set()
                        for t in threads:
                            if t.ident is not None:
                                t.join()
                        raise
                    if move_errors:
                        raise RuntimeError("Calibration pose movement failed") from move_errors[0]

                # Wait for stabilization
                print("  Waiting for stabilization...")
                time.sleep(1.0)
            else:
                # No robots - just a short delay for scene camera
                time.sleep(0.5)

            # Capture images and detect board
            print("  Capturing images...")
            pose_images: Dict[str, np.ndarray] = {}
            capture = {
                "id": i,
                "pose": pose["name"],
                "images": pose_images,
                "joints_l": None,
                "joints_r": None,
            }
            self._captures.append(capture)
            for camera_name, camera in self.cameras.items():
                try:
                    # Capture frame
                    if not camera.grab():
                        raise RuntimeError(f"grab() failed for {camera_name}")
                    image = camera.get_frame().color

                    # Ensure image is contiguous and has correct dtype
                    image = np.ascontiguousarray(image, dtype=np.uint8)

                    camera_data[camera_name]["images"].append(image)
                    pose_images[camera_name] = image

                    # Retain the same measured joints used for this image's FK,
                    # including images where board detection subsequently fails.
                    arm_type = camera_data[camera_name]["arm_type"]
                    actual_joint_pos = None
                    if arm_type is not None and self.robot_controller:
                        side = "r" if arm_type == "right" else "l"
                        robot = getattr(self.robot_controller, f"follower_{side}", None)
                        if robot is not None:
                            try:
                                measured = np.asarray(robot.get_joint_pos(), dtype=float).reshape(-1)
                                if measured.size < 6 or not np.isfinite(measured).all():
                                    raise ValueError("Invalid measured joints")
                                actual_joint_pos = measured
                                capture[f"joints_{side}"] = measured.tolist()
                            except Exception as e:
                                print(f"    Warning: Failed to read {arm_type} joints: {e}")

                    # Detect ChArUco board
                    corners, ids = self.calibrator.detector.detect(image)

                    if corners is not None and len(corners) > 0:
                        camera_data[camera_name]["corners"].append(corners)
                        camera_data[camera_name]["ids"].append(ids)
                        camera_data[camera_name]["observation_ids"].append(i)
                        print(
                            f"    ✓ {camera_name}: ChArUco board detected ({len(corners)} corners)"
                        )

                        # Keep one FK slot per detection so failed reads cannot
                        # shift the robot/board correspondence at later poses.
                        if arm_type is not None:
                            robot_pose = None
                            try:
                                # Compute forward kinematics from actual joint positions
                                # For initial data collection, compute FK in local frames
                                # (we'll recompute after getting bimanual transform)
                                if actual_joint_pos is not None:
                                    global _kinematics_cache
                                    if _kinematics_cache is None:
                                        _kinematics_cache = Kinematics(
                                            _get_combined_xml_path(),
                                            site_name="grasp_site",
                                        )

                                    q = np.zeros(8)
                                    q[:6] = actual_joint_pos[:6]
                                    robot_pose = _kinematics_cache.fk(q)
                            except Exception as e:
                                print(
                                    f"    Warning: Failed to get actual robot pose: {e}"
                                )
                            camera_data[camera_name]["robot_poses"].append(robot_pose)
                    else:
                        print(f"    ✗ {camera_name}: Board not detected")

                except Exception as e:
                    print(f"    ✗ {camera_name}: Error - {e}")

        print("\n" + "=" * 70)
        print("Data collection complete!")
        return camera_data

    def run_calibration(
        self,
        left_wrist_camera_id: Optional[str] = None,
        right_wrist_camera_id: Optional[str] = None,
        scene_camera_ids: Optional[List[str]] = None,
    ) -> Dict:
        """Run the full calibration workflow

        Args:
            left_wrist_camera_id: Name of left wrist camera (e.g., "left_wrist")
            right_wrist_camera_id: Name of right wrist camera (e.g., "right_wrist")
            scene_camera_ids: Names of all scene cameras (multiple allowed)

        Returns:
            Calibration results dictionary
        """
        failure = None
        captures_saved = False
        self._captures = []
        self._move_stop_event.clear()
        try:
            # Load configuration
            self.load_configuration()

            # Determine which cameras to calibrate
            target_cameras = []
            calibrate_left = left_wrist_camera_id is not None
            calibrate_right = right_wrist_camera_id is not None
            calibrate_scene = bool(scene_camera_ids)
            self._wrist_arms = {
                name: side for name, side in (
                    (left_wrist_camera_id, "left"), (right_wrist_camera_id, "right")
                ) if name is not None
            }

            if calibrate_right:
                target_cameras.append(right_wrist_camera_id)
            if calibrate_left:
                target_cameras.append(left_wrist_camera_id)
            if calibrate_scene:
                target_cameras.extend(scene_camera_ids)

            # Check bimanual transform requirements
            global T_RIGHT_BASE_TO_LEFT_BASE
            if calibrate_left and calibrate_right:
                # Every two-arm run measures the current rig, including retries
                # made in the same Python process.
                T_RIGHT_BASE_TO_LEFT_BASE = None
            if (
                calibrate_right
                and not calibrate_left
                and T_RIGHT_BASE_TO_LEFT_BASE is None
            ):
                raise ValueError(
                    "Cannot calibrate right wrist camera without bimanual transform.\n"
                    "Please calibrate both left and right wrist cameras together to compute the transform."
                )

            print("\nCalibration targets:")
            for camera in target_cameras:
                camera_type = "hand-eye" if "wrist" in camera else "scene"
                print(f"  - {camera} ({camera_type})")

            # Initialize hardware
            # Initialize cameras first (quick check before robot initialization)
            self.initialize_cameras(target_cameras)

            # Only initialize robots if we're calibrating wrist cameras
            if calibrate_left or calibrate_right:
                self.initialize_robots(calibrate_left, calibrate_right)
            else:
                print("\nNo wrist cameras selected - skipping robot initialization")

            # Collect data
            camera_data = self.collect_calibration_data()
            self._save_captures()
            captures_saved = True

            # Compute bimanual transform if both arms are calibrated
            bimanual_transform_computed = False
            if calibrate_left and calibrate_right and T_RIGHT_BASE_TO_LEFT_BASE is None:
                bimanual_transform_computed = (
                    self._compute_and_apply_bimanual_transform(
                        camera_data, left_wrist_camera_id, right_wrist_camera_id
                    )
                )

            # Run calibration for each camera
            print("\nComputing calibrations...")
            print("=" * 70)

            results = {
                "version": "1.0",
                "timestamp": datetime.now().isoformat(),
                "coordinate_frame": "left_arm_base" if calibrate_left or calibrate_right else "board",
                "charuco_config": self.board_config.__dict__,
                "cameras": {},
                "quality_metrics": {},
            }

            # Add bimanual transform if computed
            if bimanual_transform_computed and T_RIGHT_BASE_TO_LEFT_BASE is not None:
                results["bimanual_transform"] = {
                    "right_base_to_left_base": T_RIGHT_BASE_TO_LEFT_BASE.tolist(),
                    "T_left_from_right": np.linalg.inv(T_RIGHT_BASE_TO_LEFT_BASE).tolist(),
                    "direction": "T_left_from_right maps right_arm_base points into left_arm_base",
                    "computed_from_calibration": True,
                    "description": "right_base_to_left_base is the legacy inverse (left to right)",
                }
            elif T_RIGHT_BASE_TO_LEFT_BASE is not None:
                results["bimanual_transform"] = {
                    "right_base_to_left_base": T_RIGHT_BASE_TO_LEFT_BASE.tolist(),
                    "T_left_from_right": np.linalg.inv(T_RIGHT_BASE_TO_LEFT_BASE).tolist(),
                    "direction": "T_left_from_right maps right_arm_base points into left_arm_base",
                    "computed_from_calibration": False,
                    "description": "Transform from right arm base frame to left arm base frame (from config)",
                }

            # Process cameras in order: left wrist first (needed for scene camera conversion), then others
            # This ensures left wrist hand-eye calibration is available for scene camera conversion
            cameras_to_process = []
            if left_wrist_camera_id and left_wrist_camera_id in target_cameras:
                cameras_to_process.append(left_wrist_camera_id)
            for camera_name in target_cameras:
                if camera_name != left_wrist_camera_id:
                    cameras_to_process.append(camera_name)

            for camera_name in cameras_to_process:
                print(f"\n{camera_name}:")
                data = camera_data[camera_name]

                if len(data["corners"]) < 3:
                    print(
                        f"  ✗ Insufficient data ({len(data['corners'])} views, need >= 3)"
                    )
                    continue

                # Get factory-calibrated intrinsics from the camera SDK
                cam_type = self.camera_config.get_camera_type(camera_name) or "unknown"
                print(
                    f"  Getting intrinsics from {cam_type} SDK (factory calibration)..."
                )
                camera = self.cameras.get(camera_name)
                if not camera:
                    print("  ✗ Camera object not found")
                    continue

                cam_matrix, dist_coeffs, image_size = camera.get_intrinsics()
                print(
                    f"    ✓ fx={cam_matrix[0, 0]:.1f}, fy={cam_matrix[1, 1]:.1f}, cx={cam_matrix[0, 2]:.1f}, cy={cam_matrix[1, 2]:.1f}"
                )

                intrinsics_result = {
                    "success": True,
                    "camera_matrix": cam_matrix.tolist(),
                    "distortion_coeffs": dist_coeffs.tolist(),
                }

                # Store intrinsics
                camera_result = {
                    "type": "hand_eye" if camera_name in (left_wrist_camera_id, right_wrist_camera_id) else "scene",
                    "serial_number": self.camera_config.get_serial_by_name(camera_name),
                    "intrinsics": {
                        "camera_matrix": intrinsics_result["camera_matrix"],
                        "distortion_coeffs": intrinsics_result["distortion_coeffs"],
                        "image_size": list(image_size),
                        "source": f"{cam_type}_sdk_factory_calibration",
                    },
                    "num_poses_used": len(data["corners"]),
                }

                # Perform extrinsics calibration
                if camera_name in (left_wrist_camera_id, right_wrist_camera_id):
                    # Hand-eye calibration
                    print("  Computing hand-eye calibration...")

                    # Check if we have robot poses
                    if not data["robot_poses"]:
                        print(
                            "  ✗ Forward kinematics not implemented - skipping hand-eye calibration"
                        )
                    else:
                        # Estimate camera poses (board-to-camera)
                        camera_matrix = np.array(intrinsics_result["camera_matrix"])
                        dist_coeffs = np.array(intrinsics_result["distortion_coeffs"])
                        robot_poses, camera_poses = self._valid_wrist_observations(
                            data, camera_matrix, dist_coeffs
                        )
                        camera_result["num_poses_used"] = len(camera_poses)

                        # Run hand-eye calibration
                        hand_eye_result = self.calibrator.calibrate_hand_eye(
                            robot_poses, camera_poses
                        )

                        if hand_eye_result["success"]:
                            camera_result["hand_eye_calibration"] = hand_eye_result
                            print(
                                f"    ✓ Hand-eye calibration complete (method: {hand_eye_result['method']})"
                            )
                        else:
                            print(
                                f"  ✗ Hand-eye calibration failed: {hand_eye_result.get('error', 'Unknown error')}"
                            )

                else:
                    # Scene camera calibration
                    print("  Computing scene camera extrinsics...")

                    # Estimate camera poses (T_board_to_scene_cam)
                    camera_matrix = np.array(intrinsics_result["camera_matrix"])
                    dist_coeffs = np.array(intrinsics_result["distortion_coeffs"])
                    camera_poses = []

                    for corners, ids in zip(data["corners"], data["ids"]):
                        rvec, tvec = self.calibrator.detector.estimate_pose(
                            corners, ids, camera_matrix, dist_coeffs
                        )
                        if rvec is not None:
                            camera_poses.append((rvec, tvec))

                    # Calibrate scene camera relative to board frame first
                    scene_result = self.calibrator.calibrate_scene_camera(camera_poses)
                    camera_result["num_poses_used"] = len(camera_poses)

                    if scene_result["success"]:
                        # Now convert to left arm base frame if we have left wrist camera data
                        # T_scene_cam_to_left_base = T_board_to_left_base @ T_scene_cam_to_board
                        # We can get T_board_to_left_base from the left wrist camera observations

                        if calibrate_left and left_wrist_camera_id:
                            left_data = camera_data.get(left_wrist_camera_id)
                            if left_data and "hand_eye_calibration" in results[
                                "cameras"
                            ].get(left_wrist_camera_id, {}):
                                print(
                                    "    Converting scene camera pose to left arm base frame..."
                                )

                                # Get left hand-eye calibration (T_left_cam_to_left_ee)
                                left_hand_eye = results["cameras"][
                                    left_wrist_camera_id
                                ]["hand_eye_calibration"]
                                T_left_cam_to_left_ee = np.eye(4)
                                T_left_cam_to_left_ee[:3, :3] = np.array(
                                    left_hand_eye["rotation_matrix"]
                                )
                                T_left_cam_to_left_ee[:3, 3] = np.array(
                                    left_hand_eye["translation_vector"]
                                )

                                # Get left intrinsics
                                left_camera = self.cameras.get(left_wrist_camera_id)
                                left_cam_matrix, left_dist_coeffs, _ = (
                                    left_camera.get_intrinsics()
                                )

                                # Estimate board poses from left camera
                                left_robot_poses, left_board_poses = self._valid_wrist_observations(
                                    left_data, left_cam_matrix, left_dist_coeffs
                                )

                                if left_board_poses:
                                    # Compute average T_board_to_left_base from left wrist observations
                                    print(
                                        f"      Computing T_board_to_left_base from {len(left_board_poses)} left wrist observations..."
                                    )
                                    board_to_left_base_transforms = []
                                    for robot_pose, (rvec, tvec) in zip(left_robot_poses, left_board_poses):
                                        # T_board_to_left_cam
                                        R, _ = cv2.Rodrigues(rvec)
                                        T_board_to_left_cam = np.eye(4)
                                        T_board_to_left_cam[:3, :3] = R
                                        T_board_to_left_cam[:3, 3] = tvec.flatten()

                                        # Correct chain: T^base_board = FK @ hand_eye @ ChArUco
                                        T_left_base_to_left_ee = robot_pose

                                        T_board_to_left_base = (
                                            T_left_base_to_left_ee
                                            @ T_left_cam_to_left_ee
                                            @ T_board_to_left_cam
                                        )

                                        board_to_left_base_transforms.append(
                                            T_board_to_left_base
                                        )

                                    if board_to_left_base_transforms:
                                        # Average the transforms
                                        from scipy.spatial.transform import Rotation

                                        rotations = [
                                            Rotation.from_matrix(T[:3, :3])
                                            for T in board_to_left_base_transforms
                                        ]
                                        translations = [
                                            T[:3, 3]
                                            for T in board_to_left_base_transforms
                                        ]

                                        avg_rotation = Rotation.concatenate(
                                            rotations
                                        ).mean()
                                        avg_translation = np.mean(translations, axis=0)

                                        T_board_to_left_base = np.eye(4)
                                        T_board_to_left_base[:3, :3] = (
                                            avg_rotation.as_matrix()
                                        )
                                        T_board_to_left_base[:3, 3] = avg_translation

                                        # Now compute T_scene_cam_to_left_base
                                        T_scene_cam_to_board = np.eye(4)
                                        T_scene_cam_to_board[:3, :3] = np.array(
                                            scene_result["rotation_matrix"]
                                        )
                                        T_scene_cam_to_board[:3, 3] = np.array(
                                            scene_result["translation_vector"]
                                        )

                                        # T_board_to_left_base is T^base_board (transforms board->base)
                                        # T_scene_cam_to_board is T^board_cam (transforms cam->board, from calibrate_scene_camera)
                                        # T^base_cam = T^base_board @ T^board_cam
                                        T_left_base_to_scene_cam = (
                                            T_board_to_left_base @ T_scene_cam_to_board
                                        )

                                        scene_cam_pos_in_left_base = (
                                            T_left_base_to_scene_cam[:3, 3]
                                        )
                                        print(
                                            f"      Scene camera position in left base: {scene_cam_pos_in_left_base}"
                                        )

                                        # Store T^base_cam: scene camera pose in left base frame
                                        # translation_vector = scene camera position in left base frame
                                        scene_result["rotation_matrix"] = (
                                            T_left_base_to_scene_cam[:3, :3].tolist()
                                        )
                                        scene_result["translation_vector"] = (
                                            T_left_base_to_scene_cam[:3, 3].tolist()
                                        )
                                        rvec_new, _ = cv2.Rodrigues(
                                            T_left_base_to_scene_cam[:3, :3]
                                        )
                                        scene_result["rotation_vector"] = (
                                            rvec_new.flatten().tolist()
                                        )
                                        scene_result["reference_frame"] = (
                                            "left_arm_base"
                                        )

                                        print(
                                            "      ✓ Scene camera pose converted to left arm base frame"
                                        )
                                    else:
                                        print(
                                            "      ! Warning: Could not convert to left arm base frame (no valid board observations)"
                                        )
                            else:
                                print(
                                    "      ! Warning: Cannot convert to left arm base frame (left wrist not calibrated)"
                                )
                        else:
                            print(
                                "      ! Warning: Scene camera extrinsics are in board frame (left wrist not calibrated)"
                            )

                        camera_result["extrinsics"] = scene_result
                        print("    ✓ Scene camera calibration complete")
                    else:
                        print("  ✗ Scene camera calibration failed")

                results["cameras"][camera_name] = camera_result

            # Compute quality metrics
            total_poses = len(self.poses_data["poses"])
            poses_with_detections = max(
                len(data["corners"]) for data in camera_data.values()
            )

            results["quality_metrics"] = {
                "overall_success": False,
                "poses_collected": total_poses,
                "poses_with_detections": poses_with_detections,
                "detection_rate": poses_with_detections / total_poses
                if total_poses > 0
                else 0,
            }

            self._validate_results(results, target_cameras, calibrate_left and calibrate_right)
            results["quality_metrics"]["overall_success"] = True

            print("\n" + "=" * 70)
            print("Calibration complete!")
            print("\nSummary:")
            print(f"  - Cameras calibrated: {len(results['cameras'])}")
            print(f"  - Poses used: {poses_with_detections}/{total_poses}")
            print(
                f"  - Detection rate: {results['quality_metrics']['detection_rate'] * 100:.1f}%"
            )
            if bimanual_transform_computed:
                print("  - Bimanual transform: Computed from calibration data ✓")
            elif T_RIGHT_BASE_TO_LEFT_BASE is not None:
                print("  - Bimanual transform: Loaded from configuration file")

            # Save results
            save_calibration_results(results, self.output_file)
            print(f"\n✓ Results saved to: {self.output_file}")

            # Store calibration result in DB
            try:
                from raiden.db.database import get_db

                get_db().add_calibration_result(results, self.output_file)
                print("✓ Calibration result recorded in DB")
            except Exception as _db_err:
                print(f"  Warning: could not record calibration in DB: {_db_err}")

            if self.robot_controller:
                self.robot_controller.move_to_home_positions(simultaneous=True)

            return results

        except KeyboardInterrupt as error:
            failure = error
            self._move_stop_event.set()
            print("\n\nCalibration interrupted by user")
            raise

        except Exception as e:
            failure = e
            self._move_stop_event.set()
            print(f"\nError during calibration: {e}")
            import traceback

            traceback.print_exc()
            raise

        finally:
            self._move_stop_event.set()
            cleanup_errors = []
            # Replay threads have stopped before collection raises. Keep their
            # completed images and measured joints available for offline diagnosis.
            if self._captures and not captures_saved:
                try:
                    self._save_captures()
                except BaseException as error:
                    cleanup_errors.append(("capture artifacts", error))
            for name, camera in self.cameras.items():
                try:
                    camera.close()
                except BaseException as error:
                    cleanup_errors.append((f"camera {name}", error))
            self.cameras.clear()
            if self.robot_controller:
                try:
                    if failure is not None and self.robot_controller.has_robots():
                        # The existing emergency stop exits the process, so
                        # cameras and capture artifacts must be handled first.
                        self.robot_controller.emergency_stop()
                except BaseException as error:
                    cleanup_errors.append(("emergency stop", error))
                for name in ("stop_teleoperation", "close"):
                    try:
                        getattr(self.robot_controller, name)()
                    except BaseException as error:
                        cleanup_errors.append((name, error))
            if cleanup_errors:
                primary = failure if failure is not None else cleanup_errors[0][1]
                for name, error in cleanup_errors:
                    primary.add_note(f"Calibration cleanup ({name}) failed: {error!r}")
                if failure is None:
                    raise primary
